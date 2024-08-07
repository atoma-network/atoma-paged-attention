pub mod cache_manager;
mod ffi;
pub mod ops;
pub use cache_manager::{copy_blocks, reshape_and_cache_flash, swap_blocks};

use std::mem::MaybeUninit;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

/// Flash-attention v2 layer.
pub struct FlashAttention {
    /// Softmax scale
    pub softmax_scale: f32,
    /// Alibi slopes,
    /// see https://nn.labml.ai/transformers/alibi/index.html
    pub alibi_slopes: Option<Tensor>,
    /// Window size for left sided local attention
    pub window_size_left: Option<usize>,
    /// Window size for right sided local attention
    pub window_size_right: Option<usize>,
    /// Softcap parameter, used in Grok and Gemma2 models
    pub softcap: Option<f32>,
}

impl FlashAttention {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/7551202cb2dd245432bc878447e19015c0af3c22/csrc/flash_attn/flash_api.cpp#L341
        let device = q.device();

        utils::check_gpu_compatibility(device.ordinal())?;

        if q.dtype() != k.dtype() {
            candle_core::bail!("query and key must have the same dtype");
        }

        if q.dtype() != v.dtype() {
            candle_core::bail!("query and value must have the same dtype");
        }

        // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;

        let seqlenq_ngroups_swapped = seqlen_q == 1
            && num_heads > num_heads_k
            && self.window_size_left.is_none()
            && self.window_size_right.is_none()
            && head_size_og % 8 == 0
            && self.alibi_slopes.is_none();
        // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
        let (q_l, out_l, out_shape, seqlen_q, num_heads) = if seqlenq_ngroups_swapped {
            let ngroups = num_heads / num_heads_k;
            let new_shape = Shape::from((b_sz, ngroups, num_heads_k, head_size_og));

            // Create new layout for q, maintaining the original start_offset
            let new_q_l = Layout::contiguous_with_offset(&new_shape, q_l.start_offset());

            (
                new_q_l,
                Layout::contiguous(&new_shape),
                new_shape,
                ngroups,
                num_heads_k,
            )
        } else {
            let out_shape = q_l.shape().clone();
            (
                q_l.clone(),
                Layout::contiguous(&out_shape),
                out_shape,
                seqlen_q,
                num_heads,
            )
        };

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle_core::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank})"
            )
        }

        if q_stride[q_rank - 1] != 1 {
            candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle_core::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle_core::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);

        if expected_kv != k_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle_core::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle_core::bail!(
                "only supports head sizes that are a multiple of 8 (got {head_size_og})"
            )
        }
        if num_heads % num_heads_k != 0 {
            candle_core::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let (alibi_slopes_ptr, alibi_slopes_batch_stride) =
            if let Some(alibi_slopes) = &self.alibi_slopes {
                if alibi_slopes.dtype() != DType::F32 {
                    candle_core::bail!(
                        "DType mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes.dtype(),
                        DType::F32
                    );
                }

                let alibi_slopes_batch_stride = if alibi_slopes.dims().len() == 2 {
                    alibi_slopes.stride()[0]
                } else {
                    0
                };

                let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

                if num_heads != alibi_slopes_layout.shape().dims1()? {
                    candle_core::bail!(
                        "shape mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes_layout.shape(),
                        (num_heads)
                    );
                }

                let alibi_slopes = match &*alibi_slopes {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle_core::bail!("alibi_slopes must be a cuda tensor"),
                };

                (
                    *alibi_slopes.device_ptr() as *const core::ffi::c_void,
                    alibi_slopes_batch_stride,
                )
            } else {
                (std::ptr::null(), 0)
            };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let mut is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if seqlen_q == 1 && !self.alibi_slopes.is_some() {
            is_causal = 0;
        }

        let head_size = utils::round_multiple(head_size_og, 8);
        let head_size_rounded = utils::round_multiple(head_size, 32);
        let seqlen_q_rounded = utils::round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = utils::round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { device.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = device
            .alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        let num_splits = utils::compute_num_splits(
            b_sz,
            num_heads,
            head_size,
            seqlen_k,
            seqlen_q,
            device.ordinal(),
        )?;

        let mut softcap = self.softcap.unwrap_or(0.0);
        let (softmax_scale, scale_softmatx_log2) = if softcap > 0.0 {
            softcap = self.softmax_scale / softcap;
            (softcap, softcap * std::f32::consts::LOG2_E)
        } else {
            // Remove potential NaN
            softcap = 0.0;
            (
                self.softmax_scale,
                self.softmax_scale * std::f32::consts::LOG2_E,
            )
        };

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            let (q_batch_stride, o_batch_stride) = if !seqlenq_ngroups_swapped {
                (q_stride[0] as u32, o_stride[0] as u32)
            } else {
                (
                    (q_stride[0] * seqlen_q) as u32,
                    (o_stride[0] * seqlen_q) as u32,
                )
            };
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* is_seqlens_k_cumulative */ true,
                /* q_batch_stride */ q_batch_stride,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_batch_stride,
                /* alibi_slopes_batch_stride */ alibi_slopes_batch_stride as u32,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* num_splits */ num_splits,
                /* b */ b_sz as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ softmax_scale,
                /* scale_softmax_log2 */ scale_softmatx_log2,
                /* block_table */ std::ptr::null(),
                /* block_table_batch_stride */ 0,
                /* page_block_size */ 0,
                /* seqused_k */ std::ptr::null(),
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ softcap,
                /* unpadded_lse */ true,
                /* force_split_kernel */ false,
            )
        }

        let out_shape = if seqlenq_ngroups_swapped {
            Shape::from((b_sz, 1, num_heads_k * seqlen_q, head_size_og))
        } else {
            out_shape
        };

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, device.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttention {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            candle_core::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle_core::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle_core::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// # Softcap
///
/// `softcap` is applied to the softmax output. Softcap is a multiplicative factor that is applied to the
/// softmax output before the softmax is applied. Softcap is used in Grok and Gemma2 models.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed_with_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer, with variable sequence lengths.
struct FlashAttentionVarLen {
    /// Softmax scale
    pub softmax_scale: f32,
    /// Maximum sequence length of Query tensor
    pub max_seqlen_q: usize,
    /// Maximum sequence length of Key tensor
    pub max_seqlen_k: usize,
    /// Cumulative sequence lengths for the query tensor,
    /// of shape `[batch_size + 1, ]`
    pub seqlens_q: Tensor,
    /// Cumulative sequence lengths for the key tensor,
    /// of shape `[batch_size + 1, ]`
    pub seqlens_k: Tensor,
    /// The sequence used for keys tensor. If given,
    /// only this many elements of each batch element's keys are used,
    /// of shape `[batch_size, ]`
    pub seqused_k: Option<Tensor>,
    /// Block table, used for paged attention algorithm
    /// of shape [batch_size, max_num_block_per_sequence]
    pub block_table: Option<Tensor>,
    /// Alibi slopes, see https://nn.labml.ai/transformers/alibi/index.html,
    /// of shape `[num_heads, ]` or `[batch_size, num_heads]`
    pub alibi_slopes: Option<Tensor>,
    /// Window size for left sided local attention
    pub window_size_left: Option<usize>,
    /// Window size for right sided local attention
    pub window_size_right: Option<usize>,
    /// Softcap parameter, used in Grok and Gemma2 models
    pub softcap: Option<f32>,
}

impl FlashAttentionVarLen {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout, // shape: `[total_q,  num_heads, head_size]`, total_q := \sum_{i=0}^{b} s_i
        k: &candle_core::CudaStorage,
        k_l: &Layout, // shape: `[total_k, num_heads_k, head_size]`, total_k := \sum_{i=0}^{b} s_i or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
        v: &candle_core::CudaStorage,
        v_l: &Layout, // shape: `[total_k, num_heads_k, head_size]`, total_k := \sum_{i=0}^{b} s_i or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
        is_bf16: bool,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/7551202cb2dd245432bc878447e19015c0af3c22/csrc/flash_attn/flash_api.cpp#L528
        let dev = q.device();

        // Check GPU device compatibility
        utils::check_gpu_compatibility(dev.ordinal())?;

        if q.dtype() != k.dtype() {
            candle_core::bail!("query and key must have the same dtype");
        }

        if q.dtype() != v.dtype() {
            candle_core::bail!("query and value must have the same dtype");
        }

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle_core::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle_core::bail!("seqlens_q has to be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle_core::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle_core::bail!("seqlens_k has to be contiguous"),
        };

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        let batch_size = nseqlens_q - 1;
        let (total_q, num_heads, head_size_og) = q_l.shape().dims3()?;

        let (block_table_ptr, block_table_layout) = if let Some(block_table) = &self.block_table {
            let (block_table_storage, block_table_layout) = block_table.storage_and_layout();
            let block_table_ptr = match &*block_table_storage {
                candle_core::Storage::Cuda(c) => {
                    let cuda_slice = c.as_cuda_slice::<i64>()?;
                    let block_table = cuda_slice.slice(block_table_layout.start_offset()..);
                    let block_table_stride = block_table_layout.stride();
                    let block_table_rank = block_table_stride.len();
                    if block_table_stride[block_table_rank - 1] != 1 {
                        candle_core::bail!("block_table must be contiguous")
                    }
                    *block_table.device_ptr() as *const i32
                }
                _ => candle_core::bail!("block_table must be a cuda tensor"),
            };
            // Clone block_table_storage to extend its lifetime
            (block_table_ptr, Some(block_table_layout))
        } else {
            (std::ptr::null(), None)
        };

        let (num_blocks, total_k, num_heads_k, head_size_og) = if !block_table_ptr.is_null() {
            k_l.shape().dims4()?
        } else {
            let (total_k, num_heads_k, _head_size_og) = k_l.shape().dims3()?;
            (0, total_k, num_heads_k, head_size_og)
        };

        let seqlenq_ngroups_swapped = self.max_seqlen_q == 1
            && num_heads > num_heads_k
            && self.window_size_left.is_none()
            && self.window_size_right.is_none()
            && head_size_og % 8 == 0
            && self.alibi_slopes.is_none();
        // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
        let (q_l, out_l, out_shape, max_seqlen_q, num_heads) = if seqlenq_ngroups_swapped {
            let ngroups = num_heads / num_heads_k;
            let new_shape = Shape::from((batch_size, ngroups, num_heads_k, head_size_og));

            // Create new layout for q, maintaining the original start_offset
            let new_q_l = Layout::contiguous_with_offset(&new_shape, q_l.start_offset());
            // TODO: use `Layout` reshape
            (
                new_q_l,
                Layout::contiguous(&new_shape),
                new_shape,
                ngroups,
                num_heads_k,
            )
        } else {
            let out_shape = q_l.shape().clone();
            (
                q_l.clone(),
                Layout::contiguous(&out_shape),
                out_shape,
                self.max_seqlen_q,
                num_heads,
            )
        };

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if block_table_ptr.is_null() && (q_rank != 3 || k_rank != 3 || v_rank != 3) {
            candle_core::bail!(
                "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        } else if !block_table_ptr.is_null() && (q_rank != 3 || k_rank != 4 || v_rank != 4) {
            candle_core::bail!(
                "flash-attn-varlen expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle_core::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle_core::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let max_num_blocks_per_sequence = if let Some(layout) = block_table_layout {
            let (b_sz, max_num_blocks_per_sequence) = layout.shape().dims2()?;
            if b_sz != batch_size {
                candle_core::bail!(
                    "shape mismatch of block_table (got {:?}) expected {:?})",
                    layout.shape(),
                    (batch_size, max_num_blocks_per_sequence)
                )
            }
            max_num_blocks_per_sequence
        } else {
            0
        };

        let page_block_size = if block_table_layout.is_some() {
            total_k
        } else {
            1
        };

        if !block_table_ptr.is_null() && page_block_size % 16 != 0 {
            // NOTE: We are following the vLLM flash attention fork, where the paged
            // block size must be divisible by 16. In the actual flash attention
            // repository, the paged block size must be divisible by 256, instead.
            // TODO: benchmark the performance of block sizes such as
            // [16, 32, 64, 128, 256]
            candle_core::bail!("page_block_size must be a multiple of 16, got {page_block_size}")
        }

        if batch_size <= 0 {
            candle_core::bail!("batch_size must be > 0")
        }
        if head_size_og > 256 {
            candle_core::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle_core::bail!(
                "only supports head sizes that are a multiple of 8 (got {head_size_og})"
            )
        }
        if num_heads % num_heads_k != 0 {
            candle_core::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        if let Some(layout) = block_table_layout {
            if k_l.shape().dims4()? != (num_blocks, page_block_size, num_heads_k, head_size_og) {
                candle_core::bail!(
                    "shape mismatch of k (got {:?}) expected {:?})",
                    k_l.shape(),
                    (num_blocks, page_block_size, num_heads_k, head_size_og)
                )
            }
            if v_l.shape().dims4()? != (num_blocks, page_block_size, num_heads_k, head_size_og) {
                candle_core::bail!(
                    "shape mismatch of v (got {:?}) expected {:?})",
                    v_l.shape(),
                    (num_blocks, page_block_size, num_heads_k, head_size_og)
                )
            }
            if layout.shape().dims2()? != (batch_size, max_num_blocks_per_sequence) {
                candle_core::bail!(
                    "shape mismatch of block_table (got {:?}) expected {:?})",
                    layout.shape(),
                    (batch_size, max_num_blocks_per_sequence)
                )
            }
        } else {
            if k_l.shape().dims3()? != (total_k, num_heads_k, head_size_og) {
                candle_core::bail!(
                    "shape mismatch of k (got {:?}) expected {:?})",
                    k_l.shape(),
                    (total_k, num_heads_k, head_size_og)
                )
            }
            if v_l.shape().dims3()? != (total_k, num_heads_k, head_size_og) {
                candle_core::bail!(
                    "shape mismatch of v (got {:?}) expected {:?})",
                    v_l.shape(),
                    (total_k, num_heads_k, head_size_og)
                )
            }
        }

        if seqlens_k_layout.shape().dims1()? != batch_size + 1 {
            candle_core::bail!(
                "shape mismatch of seqlens_k (got {:?}) expected {:?})",
                seqlens_k_layout.shape(),
                (batch_size + 1)
            )
        }
        if seqlens_q_layout.shape().dims1()? != batch_size + 1 {
            candle_core::bail!(
                "shape mismatch of seqlens_q (got {:?}) expected {:?})",
                seqlens_q_layout.shape(),
                (batch_size + 1)
            )
        }

        if nseqlens_q < 2 {
            candle_core::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
        }
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_k != nseqlens_q {
            candle_core::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
        }

        let (alibi_slopes_ptr, alibi_slopes_batch_stride) =
            if let Some(alibi_slopes) = &self.alibi_slopes {
                if alibi_slopes.dtype() != DType::F32 {
                    candle_core::bail!(
                        "DType mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes.dtype(),
                        DType::F32
                    );
                }

                let alibi_slopes_batch_stride = if alibi_slopes.dims().len() == 2 {
                    alibi_slopes.stride()[0]
                } else {
                    0
                };

                let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

                if num_heads != alibi_slopes_layout.shape().dims1()? {
                    candle_core::bail!(
                        "shape mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes_layout.shape(),
                        (num_heads)
                    );
                }

                let alibi_slopes = match &*alibi_slopes {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle_core::bail!("alibi_slopes must be a cuda tensor"),
                };

                let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

                (
                    *alibi_slopes.device_ptr() as *const core::ffi::c_void,
                    alibi_slopes_batch_stride,
                )
            } else {
                (std::ptr::null(), 0)
            };

        let seqused_k = if let Some(seqused_k) = &self.seqused_k {
            let (seqused_k_storage, seqused_k_layout) = seqused_k.storage_and_layout();
            let seqused_k_ptr = match &*seqused_k_storage {
                candle_core::Storage::Cuda(c) => {
                    let cuda_slice = c.as_cuda_slice::<u32>()?;
                    let seqused_k = cuda_slice.slice(seqused_k_layout.start_offset()..);
                    let seqused_k_stride = seqused_k_layout.stride();
                    let seqused_k_rank = seqused_k_stride.len();
                    if seqused_k_stride[seqused_k_rank - 1] != 1 {
                        candle_core::bail!("block_table must be contiguous")
                    }
                    *seqused_k.device_ptr() as *const i32
                }
                _ => candle_core::bail!("block_table must be a cuda tensor"),
            };
            seqused_k_ptr
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = utils::round_multiple(head_size_og, 8);
        let head_size_rounded = utils::round_multiple(head_size, 32);
        let seqlen_q_rounded = utils::round_multiple(self.max_seqlen_q, 128);
        let seqlen_k_rounded = utils::round_multiple(self.max_seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = dev.alloc_zeros::<f32>(total_q * num_heads).w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = self.max_seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = self.max_seqlen_k as i32;
        }

        let num_splits = if seqlenq_ngroups_swapped {
            // Only apply split-k for decoding
            utils::compute_num_splits(
                batch_size,
                num_heads,
                head_size,
                self.max_seqlen_k,
                max_seqlen_q,
                dev.ordinal(),
            )?
        } else {
            0
        };

        let mut softcap = self.softcap.unwrap_or(0.0);
        let (softmax_scale, scale_softmatx_log2) = if softcap > 0.0 {
            softcap = self.softmax_scale / softcap;
            (softcap, softcap * std::f32::consts::LOG2_E)
        } else {
            // Remove potential NaN
            softcap = 0.0;
            (
                self.softmax_scale,
                self.softmax_scale * std::f32::consts::LOG2_E,
            )
        };

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let block_table_batch_stride = if let Some(layout) = block_table_layout {
                layout.stride()[0] as u32
            } else {
                0
            };
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            let seqlens_q_ptr = if !seqlenq_ngroups_swapped {
                *seqlens_q.device_ptr() as *const core::ffi::c_int
            } else {
                std::ptr::null()
            };
            let seqlens_k_ptr = *seqlens_k.device_ptr() as *const core::ffi::c_int;
            let (q_batch_stride, o_batch_stride) =
                match (seqlens_q_ptr.is_null(), seqlenq_ngroups_swapped) {
                    (false, _) => (0, 0),
                    (true, true) => (
                        (q_stride[0] * max_seqlen_q) as u32,
                        (o_stride[0] * max_seqlen_q) as u32,
                    ),
                    (true, false) => (q_stride[0] as u32, o_stride[0] as u32),
                };
            let (k_batch_stride, v_batch_stride) = block_table_layout
                .as_ref()
                .map(|_| (k_stride[0] as u32, v_stride[0] as u32))
                .unwrap_or((0, 0));
            // TODO: handle case where max_seqlen_q == 0, separately
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ seqlens_q_ptr,
                /* cu_seqlens_k_ptr */ seqlens_k_ptr,
                /* is_seqlens_k_cumulative */ true,
                /* q_batch_stride */ q_batch_stride,
                /* k_batch_stride */ k_batch_stride,
                /* v_batch_stride */ v_batch_stride,
                /* o_batch_stride */ o_batch_stride,
                /* alibi_slopes_batch_stride */ alibi_slopes_batch_stride as u32,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* num_splits */ num_splits,
                /* b */ batch_size as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ softmax_scale,
                /* scale_softmatx_log2 */ scale_softmatx_log2,
                /* block_table */ block_table_ptr,
                /* block_table_batch_stride */ block_table_batch_stride,
                /* page_block_size */ page_block_size as i32,
                /* seqused_k */ seqused_k,
                /* seqlen_q */ self.max_seqlen_q as u32,
                /* seqlen_k */ self.max_seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ softcap,
                /* unpadded_lse */ true,
                /* force_split_kernel */ !block_table_ptr.is_null(),
            )
        }

        let out_shape = if seqlenq_ngroups_swapped {
            Shape::from((batch_size, 1, num_heads_k * max_seqlen_q, head_size_og))
        } else {
            out_shape
        };

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttentionVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            candle_core::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle_core::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle_core::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqused_k: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqused_k: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqused_k: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        block_table: None,
        seqused_k: None,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// # Block table
///
/// Enables the paged attention algorithm. The block table is a tensor of shape `[batch_size, max_num_block_per_sequence]`
/// that contains the block table for each sequence in the batch. The block table is used to determine the
/// the number of blocks per sequence.
pub fn flash_attn_varlen_with_block_table(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    block_table: Option<&Tensor>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        block_table: block_table.cloned(),
        seqused_k: None,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// # Block table
///
/// Enables the paged attention algorithm. The block table is a tensor of shape `[batch_size, max_num_block_per_sequence]`
/// that contains the block table for each sequence in the batch. The block table is used to determine the
/// the number of blocks per sequence.
///
/// # Softcap
///
/// `softcap` is applied to the softmax output. Softcap is a multiplicative factor that is applied to the
/// softmax output before the softmax is applied. Softcap is used in Grok and Gemma2 models.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    block_table: Option<&Tensor>,
    seqused_k: Option<&Tensor>,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        block_table: block_table.cloned(),
        seqused_k: seqused_k.cloned(),
        softcap,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer, with Key-Value cache.
///
/// NOTE: We are not passing in each Key and Value tensor for the decoding phase. This is
/// because we plan to use paged attention with flash attention.
/// In that case, the key and value tensors at each decoding phase are stored within the
/// kv cache tensor, separately. So we don't need to pass the key and value tensors to the
/// flash attention kernel, directly.
struct FlashAttentionKvCache {
    /// Softmax scale
    pub softmax_scale: f32,
    /// Block table, used for paged attention algorithm
    /// of shape [batch_size, max_num_block_per_sequence]
    pub block_table: Option<Tensor>,
    /// Alibi slopes, see https://nn.labml.ai/transformers/alibi/index.html,
    /// of shape `[num_heads, ]` or `[batch_size, num_heads]`
    pub alibi_slopes: Option<Tensor>,
    /// Window size for left sided slicing attention
    pub window_size_left: Option<usize>,
    /// Window size for right sided slicing attention
    pub window_size_right: Option<usize>,
    /// Sequence lengths for the key tensor,
    /// of shape `[batch_size, ]`
    pub seqlens_k: Option<Tensor>,
    /// Softcap parameter, used in Grok and Gemma2 models
    pub softcap: Option<f32>,
}

impl FlashAttentionKvCache {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout, // shape: `[batch_size, seqlen_q, num_heads, head_size]`, total_q := \sum_{i=0}^{b} s_i
        kc: &candle_core::CudaStorage,
        kc_l: &Layout, // shape: `[batch_size_cache, seqlen_k, num_heads_k, head_size]`, or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
        vc: &candle_core::CudaStorage,
        vc_l: &Layout, // shape: `[batch_size_cache, seqlen_k, num_heads_k, head_size]`, or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
        is_bf16: bool,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/7551202cb2dd245432bc878447e19015c0af3c22/csrc/flash_attn/flash_api.cpp#L1284
        let dev = q.device();

        // Check GPU device compatibility
        utils::check_gpu_compatibility(dev.ordinal())?;

        if q.dtype() != kc.dtype() {
            candle_core::bail!("query and key must have the same dtype");
        }

        if q.dtype() != vc.dtype() {
            candle_core::bail!("query and value must have the same dtype");
        }

        let (block_table_ptr, block_table_layout) = if let Some(block_table) = &self.block_table {
            let (block_table_storage, block_table_layout) = block_table.storage_and_layout();
            let block_table_ptr = match &*block_table_storage {
                candle_core::Storage::Cuda(c) => {
                    let cuda_slice = c.as_cuda_slice::<i64>()?;
                    let block_table = cuda_slice.slice(block_table_layout.start_offset()..);
                    let block_table_stride = block_table_layout.stride();
                    let block_table_rank = block_table_stride.len();
                    if block_table_stride[block_table_rank - 1] != 1 {
                        candle_core::bail!("block_table must be contiguous")
                    }
                    *block_table.device_ptr() as *const i32
                }
                _ => candle_core::bail!("block_table must be a cuda tensor"),
            };
            // Clone block_table_storage to extend its lifetime
            (block_table_ptr, Some(block_table_layout))
        } else {
            (std::ptr::null(), None)
        };

        let (batch_size, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;

        let max_num_blocks_per_sequence = if let Some(layout) = block_table_layout {
            let (b_sz, max_num_blocks_per_sequence) = layout.shape().dims2()?;
            if b_sz != batch_size {
                candle_core::bail!(
                    "shape mismatch of block_table (got {:?}) expected {:?})",
                    layout.shape(),
                    (batch_size, max_num_blocks_per_sequence)
                )
            }
            max_num_blocks_per_sequence
        } else {
            0
        };
        let (batch_size_cache, num_blocks, page_block_size, seqlens_k, num_heads_k, _head_size) =
            if !block_table_ptr.is_null() {
                let (num_blocks, page_block_size, num_heads_k, _head_size) =
                    kc_l.shape().dims4()?;
                (
                    batch_size,
                    num_blocks,
                    page_block_size,
                    max_num_blocks_per_sequence * page_block_size,
                    num_heads_k,
                    _head_size,
                )
            } else {
                let (batch_size_cache, seqlen_k, num_heads_k, _head_size) = kc_l.shape().dims4()?;
                (batch_size_cache, 0, 1, seqlen_k, num_heads_k, _head_size)
            };

        match block_table_layout {
            Some(_) => {
                let expected_shape = (num_blocks, page_block_size, num_heads_k, head_size_og);
                if kc_l.shape().dims4()? != expected_shape {
                    candle_core::bail!(
                        "shape mismatch of k_cache (got {:?}) expected {:?})",
                        kc_l.shape(),
                        expected_shape
                    )
                }
                if vc_l.shape().dims4()? != expected_shape {
                    candle_core::bail!(
                        "shape mismatch of v_cache (got {:?}) expected {:?})",
                        vc_l.shape(),
                        expected_shape
                    )
                }
            }
            None => {
                let expected_shape = (batch_size_cache, seqlens_k, num_heads_k, head_size_og);
                if kc_l.shape().dims4()? != expected_shape {
                    candle_core::bail!(
                        "shape mismatch of k_cache (got {:?}) expected {:?})",
                        kc_l.shape(),
                        expected_shape
                    )
                }
                if vc_l.shape().dims4()? != expected_shape {
                    candle_core::bail!(
                        "shape mismatch of v_cache (got {:?}) expected {:?})",
                        vc_l.shape(),
                        expected_shape
                    )
                }
            }
        }

        if batch_size <= 0 {
            candle_core::bail!("batch_size must be > 0")
        }
        if head_size_og > 256 {
            candle_core::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle_core::bail!(
                "only supports head sizes that are a multiple of 8 (got {head_size_og})"
            )
        }
        if num_heads % num_heads_k != 0 {
            candle_core::bail!("number of k_cache/v_cache heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        if !block_table_ptr.is_null() && page_block_size % 16 != 0 {
            // NOTE: We are following the vLLM flash attention fork, where the paged
            // block size must be divisible by 16. In the actual flash attention
            // repository, the paged block size must be divisible by 256, instead.
            // TODO: benchmark the performance of block sizes such as
            // [16, 32, 64, 128, 256]
            candle_core::bail!("page_block_size must be a multiple of 16, got {page_block_size}")
        }

        let _seqlenq_ngroups_swapped = seqlen_q == 1
            && num_heads > num_heads_k
            && self.window_size_left.is_none()
            && self.window_size_right.is_none()
            && head_size_og % 8 == 0
            && self.alibi_slopes.is_none();
        let seqlenq_ngroups_swapped = false; // TODO: remove this
        let (q_l, out_l, out_shape, seqlen_q, num_heads) = if seqlenq_ngroups_swapped {
            let ngroups = num_heads / num_heads_k;
            let new_shape = Shape::from((batch_size, ngroups, num_heads_k, head_size_og));

            // Create new layout for q, maintaining the original start_offset
            let new_q_l = Layout::contiguous_with_offset(&new_shape, q_l.start_offset());
            // TODO: use `Layout` reshape
            (
                new_q_l,
                Layout::contiguous(&new_shape),
                new_shape,
                ngroups,
                num_heads_k,
            )
        } else {
            let out_shape = q_l.shape().clone();
            (
                q_l.clone(),
                Layout::contiguous(&out_shape),
                out_shape,
                seqlen_q,
                num_heads,
            )
        };

        let q = q.as_cuda_slice::<T>()?;
        let kc = kc.as_cuda_slice::<T>()?;
        let vc = vc.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let kc = kc.slice(kc_l.start_offset()..);
        let vc = vc.slice(vc_l.start_offset()..);

        let q_stride = q_l.stride();
        let kc_stride = kc_l.stride();
        let vc_stride = vc_l.stride();

        let q_rank = q_stride.len();
        let kc_rank = kc_stride.len();
        let vc_rank = vc_stride.len();

        if q_rank != 4 || kc_rank != 4 || vc_rank != 4 {
            candle_core::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {kc_rank}, v: {vc_rank})"
            )
        }

        if q_stride[q_rank - 1] != 1 {
            candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if kc_stride[kc_rank - 1] != 1 {
            candle_core::bail!("the last dim of k must be contiguous {kc_stride:?}")
        }
        if vc_stride[vc_rank - 1] != 1 {
            candle_core::bail!("the last dim of v must be contiguous {vc_stride:?}")
        }

        let (alibi_slopes_ptr, alibi_slopes_batch_stride) =
            if let Some(alibi_slopes) = &self.alibi_slopes {
                if alibi_slopes.dtype() != DType::F32 {
                    candle_core::bail!(
                        "DType mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes.dtype(),
                        DType::F32
                    );
                }

                let alibi_slopes_batch_stride = if alibi_slopes.dims().len() == 2 {
                    alibi_slopes.stride()[0]
                } else {
                    0
                };

                let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

                if num_heads != alibi_slopes_layout.shape().dims1()? {
                    candle_core::bail!(
                        "shape mismatch alibi_slopes {:?}, expected {:?}",
                        alibi_slopes_layout.shape(),
                        (num_heads)
                    );
                }

                let alibi_slopes = match &*alibi_slopes {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle_core::bail!("alibi_slopes must be a cuda tensor"),
                };

                let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

                (
                    *alibi_slopes.device_ptr() as *const core::ffi::c_void,
                    alibi_slopes_batch_stride,
                )
            } else {
                (std::ptr::null(), 0)
            };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlens_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlens_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = utils::round_multiple(head_size_og, 8);
        let head_size_rounded = utils::round_multiple(head_size, 32);
        let seqlen_q_rounded = utils::round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = utils::round_multiple(seqlens_k, 128);

        let cu_seqlens_k_ptr = if let Some(seqlens_k) = &self.seqlens_k {
            if seqlens_k.dims() != &[batch_size] {
                candle_core::bail!(
                    "shape mismatch of seqlens_k (got {:?}) expected {:?})",
                    seqlens_k.dims(),
                    [batch_size]
                )
            }
            if seqlens_k.dtype() != DType::U32 {
                candle_core::bail!(
                    "DType mismatch seqlens_k {:?}, expected {:?}",
                    seqlens_k.dtype(),
                    DType::U32
                );
            }
            let (seqlens_k, seqlens_k_layout) = seqlens_k.storage_and_layout();
            let seqlens_k = match &*seqlens_k {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle_core::bail!("seqlens_k must be a cuda tensor"),
            };
            let seqlens_k = seqlens_k.slice(seqlens_k_layout.start_offset()..);
            let seqlens_k_stride = seqlens_k_layout.stride();
            let seqlens_k_rank = seqlens_k_stride.len();
            if seqlens_k_stride[seqlens_k_rank - 1] != 1 {
                candle_core::bail!(
                    "the last dim of seqlens_k must be contiguous {seqlens_k_stride:?}"
                )
            }
            *seqlens_k.device_ptr() as *const core::ffi::c_int
        } else {
            std::ptr::null()
        };
        let is_seqlens_k_cumulative = self.seqlens_k.is_none();

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(batch_size * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let mut is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };

        if seqlen_q == 1 && self.alibi_slopes.is_none() {
            // is_causal = true is the same as is_causal = false, in this case
            is_causal = 0;
        }

        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlens_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlens_k as i32;
        }

        let num_splits = utils::compute_num_splits(
            batch_size,
            num_heads,
            head_size,
            seqlens_k,
            seqlen_q,
            dev.ordinal(),
        )?;

        let mut softcap = self.softcap.unwrap_or(0.0);
        let (softmax_scale, scale_softmatx_log2) = if softcap > 0.0 {
            softcap = self.softmax_scale / softcap;
            (softcap, softcap * std::f32::consts::LOG2_E)
        } else {
            // Remove potential NaN
            softcap = 0.0;
            (
                self.softmax_scale,
                self.softmax_scale * std::f32::consts::LOG2_E,
            )
        };

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let kc_ptr = *kc.device_ptr() as *const core::ffi::c_void;
            let vc_ptr = *vc.device_ptr() as *const core::ffi::c_void;
            let block_table_batch_stride = if let Some(layout) = block_table_layout {
                layout.stride()[0] as u32
            } else {
                0
            };
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            let (k_batch_stride, v_batch_stride) = block_table_layout
                .as_ref()
                .map(|_| (kc_stride[0] as u32, vc_stride[0] as u32))
                .unwrap_or((0, 0));

            let o_stride = out_l.stride();
            let o_rank = o_stride.len();
            let (q_batch_stride, o_batch_stride) = if !seqlenq_ngroups_swapped {
                (q_stride[0] as u32, o_stride[0] as u32)
            } else {
                (
                    (q_stride[0] * seqlen_q) as u32,
                    (o_stride[0] * seqlen_q) as u32,
                )
            };
            ffi::run_mha(
                q_ptr,
                kc_ptr,
                vc_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ cu_seqlens_k_ptr,
                /* is_seqlens_k_cumulative */ is_seqlens_k_cumulative,
                /* q_batch_stride */ q_batch_stride,
                /* k_batch_stride */ k_batch_stride,
                /* v_batch_stride */ v_batch_stride,
                /* o_batch_stride */ o_batch_stride,
                /* alibi_slopes_batch_stride */ alibi_slopes_batch_stride as u32,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ kc_stride[kc_rank - 3] as u32,
                /* v_row_stride   */ vc_stride[vc_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ kc_stride[kc_rank - 2] as u32,
                /* v_head_stride  */ vc_stride[vc_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* num_splits */ num_splits,
                /* b */ batch_size as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ softmax_scale,
                /* scale_softmatx_log2 */ scale_softmatx_log2,
                /* block_table */ block_table_ptr,
                /* block_table_batch_stride */ block_table_batch_stride,
                /* page_block_size */ page_block_size as i32,
                /* seqused_k */ std::ptr::null(),
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlens_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ softcap,
                /* unpadded_lse */ false,
                /* force_split_kernel */ !block_table_ptr.is_null(),
            )
        }

        let out_shape = if seqlenq_ngroups_swapped {
            Shape::from((batch_size, 1, num_heads_k * seqlen_q, head_size_og))
        } else {
            out_shape
        };

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttentionKvCache {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        kc: &candle_core::CudaStorage,
        kc_l: &Layout,
        vc: &candle_core::CudaStorage,
        vc_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            candle_core::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, kc, kc_l, vc, vc_l, false),
            candle_core::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, kc, kc_l, vc, vc_l, true),
            dt => candle_core::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with key and value tensors cached.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
/// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
///
/// The resulting tensor has dimensions `[batch_size, seqlen_q, num_heads, head_size]`.
pub fn flash_attn_kv_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionKvCache {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqlens_k: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with key and value tensors cached.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
/// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
///
/// The resulting tensor has dimensions `[batch_size, seqlen_q, num_heads, head_size]`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_kv_cache_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_k: Option<&Tensor>,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionKvCache {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqlens_k: seqlens_k.cloned(),
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with key and value tensors cached.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
/// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_kv_cache_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_k: Option<&Tensor>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionKvCache {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
        block_table: None,
        seqlens_k: seqlens_k.cloned(),
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with key and value tensors cached.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
/// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_kv_cache_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionKvCache {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        block_table: None,
        seqlens_k: None,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with key and value tensors cached.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
/// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
/// * `block_table` - Block table tensor with shape `[batch_size, max_num_block_per_sequence]`.
/// * `softcap` - Softcap parameter, used in Grok and Gemma2 models.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_kv_cache_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    softmax_scale: f32,
    block_table: Option<&Tensor>,
    seqlens_k: Option<&Tensor>,
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionKvCache {
        softmax_scale,
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        block_table: block_table.cloned(),
        seqlens_k: seqlens_k.cloned(),
        softcap,
    };
    q.apply_op3(k, v, op)
}

pub(crate) mod utils {

    use cuda_runtime_sys::*;

    use super::*;
    pub(crate) fn round_multiple(x: usize, m: usize) -> usize {
        (x + m - 1) / m * m
    }

    /// Find the number of splits that maximizes the occupancy. For example, if we have
    /// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
    /// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
    /// splits as that would incur more HBM reads/writes.
    /// So we find the best efficiency, then find the smallest number of splits that gets 85%
    /// of the best efficiency.
    pub(crate) fn num_splits_heuristic(
        batch_nheads_mblocks: usize,
        num_sms: usize,
        num_n_blocks: usize,
        max_splits: usize,
    ) -> usize {
        // If we have enough to almost fill the SMs, then just use 1 split
        if (batch_nheads_mblocks as f32) >= 0.8 * (num_sms as f32) {
            return 1;
        }

        let max_splits = max_splits.min(num_sms).min(num_n_blocks);
        let mut max_efficiency = 0.0;
        let mut efficiency = Vec::with_capacity(max_splits as usize);

        let ceil_div = |a: usize, b: usize| -> usize { (a + b - 1) / b };

        let is_split_eligible = |num_splits: usize| -> bool {
            num_splits == 1
                || ceil_div(num_n_blocks, num_splits) != ceil_div(num_n_blocks, num_splits - 1)
        };

        for num_splits in 1..=max_splits {
            if !is_split_eligible(num_splits) {
                efficiency.push(0.0);
            } else {
                let n_waves = (batch_nheads_mblocks * num_splits) as f32 / num_sms as f32;
                let eff = n_waves / n_waves.ceil();
                if eff > max_efficiency {
                    max_efficiency = eff;
                }
                efficiency.push(eff);
            }
        }

        for num_splits in 1..=max_splits {
            if !is_split_eligible(num_splits) {
                continue;
            }
            if efficiency[(num_splits - 1) as usize] >= 0.85 * max_efficiency {
                return num_splits;
            }
        }

        1
    }

    pub(crate) fn compute_num_splits(
        batch_size: usize,
        num_heads: usize,
        head_size: usize,
        max_seqlen_k: usize,
        max_seqlen_q: usize,
        device_ordinal: usize,
    ) -> Result<u32> {
        let block_n = if head_size <= 64 {
            256
        } else if head_size <= 128 {
            128
        } else {
            64
        };
        let num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
        // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
        // In any case we don't expect seqlen_q to be larger than 64 for inference.
        let num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
        let cuda_multiprocessor_count = get_multiprocessor_count(device_ordinal)?;
        let num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks,
            cuda_multiprocessor_count,
            num_n_blocks,
            128,
        );
        if num_splits > 128 {
            candle_core::bail!("num_splits > 128 not supported")
        }
        Ok(num_splits as u32)
    }

    pub(crate) fn get_multiprocessor_count(device_index: usize) -> Result<usize> {
        unsafe {
            let mut count = MaybeUninit::uninit();
            let error = cudaDeviceGetAttribute(
                count.as_mut_ptr(),
                cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                device_index as i32,
            );
            if error != cudaError::cudaSuccess {
                candle_core::bail!("CUDA error: {:?}", error)
            }
            Ok(count.assume_init() as usize)
        }
    }

    pub(crate) fn check_gpu_compatibility(device_index: usize) -> Result<()> {
        use core::ffi::c_int;
        let mut props = cudaDeviceProp::default();
        unsafe {
            let error =
                cudaGetDeviceProperties(&mut props as *mut cudaDeviceProp, device_index as c_int);
            if error != cudaError::cudaSuccess {
                candle_core::bail!("CUDA error: {:?}", error)
            }
            let is_sm8x = props.major == 8 && props.minor >= 0;
            let is_sm90 = props.major == 9 && props.minor == 0;

            if !(is_sm90 || is_sm8x) {
                candle_core::bail!("FlashAttention only supports Ampere GPUs or newer.")
            }
        }
        Ok(())
    }
}
