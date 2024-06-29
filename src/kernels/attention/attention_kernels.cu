/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "src/quantization/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "src/quantization/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }

  if (lane == 0) {
    red_smem[warp] = sum;
  }

  __syncthreads();

  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }

  return __shfl_sync(0xffffffff, sum, 0);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    float* exp_sums, float* max_logits, scalar_t* out, const scalar_t* q,
    const cache_t* k_cache, const cache_t* v_cache, int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens, int max_num_blocks_per_seq,
    const float* alibi_slopes, int q_stride, int kv_block_stride, int kv_head_stride,
    float kv_scale, int tp_rank, int blocksparse_local_blocks,
    int blocksparse_vert_stride, int blocksparse_block_size,
    int blocksparse_head_sliding_step) {

  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  const int seq_len = seq_lens[seq_idx];

  if (PARTITION_SIZE > 0 && partition_idx * PARTITION_SIZE >= seq_len) {
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      PARTITION_SIZE > 0 ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  const int start_block_idx = PARTITION_SIZE > 0 ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  const int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  const int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  const int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  __shared__ float red_smem[2 * NUM_WARPS];

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  float qk_max = -FLT_MAX;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                               kv_head_idx * kv_head_stride + physical_block_offset * VEC_SIZE;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / VEC_SIZE;
        const int offset2 = (vec_idx * VEC_SIZE) % VEC_SIZE;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * VEC_SIZE + offset2);
        } else {
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + offset1 * BLOCK_SIZE * VEC_SIZE + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(k_vec_quant, kv_scale);
        }
      }

      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        logits[token_idx - start_token_idx] = qk;
        qk_max = fmaxf(qk_max, qk);
      }
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
  }
  qk_max = __shfl_sync(0xffffffff, qk_max, 0);

  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  const float inv_sum = 1.f / (exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  if (PARTITION_SIZE > 0 && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<float>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    V_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec = *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec, kv_scale);
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(0xffffffff, acc, mask);
    }
    accs[i] = acc;
  }

  __syncthreads();

  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(out_ptr[row_idx], accs[i]);
      }
    }
  }
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* out, const scalar_t* q, const cache_t* k_cache, const cache_t* v_cache,
    int num_kv_heads, float scale, const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq, const float* alibi_slopes, int q_stride,
    int kv_block_stride, int kv_head_stride, float kv_scale, int tp_rank,
    int blocksparse_local_blocks, int blocksparse_vert_stride,
    int blocksparse_block_size, int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
      nullptr, nullptr, out, q, k_cache, v_cache, num_kv_heads, scale, block_tables,
      seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, kv_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* exp_sums, float* max_logits, scalar_t* tmp_out, const scalar_t* q,
    const cache_t* k_cache, const cache_t* v_cache, int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens, int max_num_blocks_per_seq,
    const float* alibi_slopes, int q_stride, int kv_block_stride, int kv_head_stride,
    float kv_scale, int tp_rank, int blocksparse_local_blocks,
    int blocksparse_vert_stride, int blocksparse_block_size,
    int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale, block_tables,
      seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, kv_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step);
}

template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* out, const float* exp_sums, const float* max_logits, const scalar_t* tmp_out,
    const int* seq_lens, int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                                  head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    return;
  }

  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  extern __shared__ char shared_mem[];
  __shared__ float red_smem[2 * NUM_WARPS];

  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, mask));
  }
  max_logit = __shfl_sync(0xffffffff, max_logit, 0);

  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                                head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += static_cast<float>(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    out_ptr[i] = static_cast<scalar_t>(acc);
  }
}

}  // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,                    \
                                  BLOCK_SIZE, NUM_THREADS>                  \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          kv_scale, tp_rank, blocksparse_local_blocks,                      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

template <typename T, typename CACHE_T, int BLOCK_SIZE, int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    T* out_ptr, const T* query_ptr, const CACHE_T* key_cache_ptr, const CACHE_T* value_cache_ptr,
    int num_kv_heads, float scale, const int* block_tables_ptr, const int* seq_lens_ptr,
    int max_seq_len, const float* alibi_slopes_ptr, float kv_scale, int tp_rank,
    int blocksparse_local_blocks, int blocksparse_vert_stride, int blocksparse_block_size,
    int blocksparse_head_sliding_step, int num_seqs, int num_heads, int head_size,
    int max_num_blocks_per_seq, int q_stride, int kv_block_stride, int kv_head_stride,
    cudaStream_t stream) {

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);

  switch (head_size) {
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      printf("Unsupported head size: %d\n", head_size);
      break;
  }
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
  vllm::paged_attention_v2_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,           \
                                  NUM_THREADS, PARTITION_SIZE>                 \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr,              \
          seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
          kv_block_stride, kv_head_stride, kv_scale, tp_rank,                  \
          blocksparse_local_blocks, blocksparse_vert_stride,                   \
          blocksparse_block_size, blocksparse_head_sliding_step);              \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions);

template <typename T, typename CACHE_T, int BLOCK_SIZE, int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    T* out_ptr, float* exp_sums_ptr, float* max_logits_ptr, T* tmp_out_ptr, const T* query_ptr,
    const CACHE_T* key_cache_ptr, const CACHE_T* value_cache_ptr, int num_kv_heads, float scale,
    const int* block_tables_ptr, const int* seq_lens_ptr, int max_seq_len, const float* alibi_slopes_ptr,
    float kv_scale, int tp_rank, int blocksparse_local_blocks, int blocksparse_vert_stride,
    int blocksparse_block_size, int blocksparse_head_sliding_step, int num_seqs, int num_heads,
    int head_size, int max_num_blocks_per_seq, int q_stride, int kv_block_stride, int kv_head_stride,
    cudaStream_t stream) {

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);

  switch (head_size) {
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V2(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      printf("Unsupported head size: %d\n", head_size);
      break;
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
