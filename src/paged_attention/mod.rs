use crate::backend::{paged_attention, reshape_and_cache};
use candle_core::{
    cuda::cudarc::driver::DevicePtr,
    cuda_backend::{cudarc::driver::DeviceRepr, CudaDType},
    DType, Device, Error as CandleError, IndexOp, Storage, Tensor, WithDType, D,
};
use csrc::ffi::{copy_blocks, swap_blocks};
use half::{bf16, f16};

/// `PagedAttentionMetadata` - Structure wrapping the metadata
/// required for paged attention
pub struct PagedAttentionMetadata {
    /// Lengths of prompts
    pub prompt_lengths: Vec<usize>,
    /// The maximum sequence length
    pub max_sequence_length: usize,
    /// The block tables. (sequence_id -> vector of physical blocks)
    pub block_tables: Tensor,
    /// The length of attention context for each generation token
    pub sequence_lengths: Vec<usize>,
    /// The sequence lengths tensor
    pub sequence_lens_tensor: Tensor,
    /// The address to write the new KV to of each token
    pub slot_mapping: Tensor,
    /// Is a prefill prompt
    pub is_prompt: bool,
    /// KV cache datatype (auto or fp8_e5m2)
    pub kv_cache_dtype: String,
    /// Attention bias, one per input prompt
    pub attention_bias: Vec<Option<Tensor>>,
}

impl PagedAttentionMetadata {
    /// Constructor
    pub fn new(
        prompt_lengths: Vec<usize>,
        max_sequence_length: usize,
        block_tables: Tensor,
        sequence_lengths: Vec<usize>,
        sequence_lens_tensor: Tensor,
        slot_mapping: Tensor,
        kv_cache_dtype: String,
    ) -> Self {
        let is_prompt = !prompt_lengths.is_empty();
        Self {
            prompt_lengths,
            max_sequence_length,
            block_tables,
            sequence_lengths,
            sequence_lens_tensor,
            slot_mapping,
            is_prompt,
            kv_cache_dtype,
            attention_bias: vec![],
        }
    }
}

/// `PagedAttention` - Structure wrapping the CUDA
/// kernels implementing the paged attention memory
/// management algorithm
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    scale: f64,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    alibi_slopes: Option<Tensor>,
}

impl PagedAttention {
    /// Constructor
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f64,
        num_kv_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: &Device,
        alibi_slopes: Option<Vec<f64>>,
    ) -> Result<Self, CandleError> {
        let num_kv_heads = num_kv_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_kv_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_kv_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            alibi_slopes,
        })
    }

    /// Available supported head sizes
    pub fn supported_head_sizes() -> Vec<u32> {
        vec![64, 80, 96, 112, 128, 192, 256]
    }

    /// Returns the KV cache shape for the given model
    /// configurations.
    pub fn get_kv_cache_shape(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_size: usize,
    ) -> Vec<usize> {
        vec![2, num_blocks, block_size * num_kv_heads * head_size]
    }

    /// Splits the KV cache
    pub fn split_kv_cache(
        kv_cache: &Tensor,
        num_kv_heads: usize,
        head_size: usize,
    ) -> Result<(Tensor, Tensor), CandleError> {
        let x = 16 / kv_cache.dtype().size_in_bytes();
        let num_blocks = kv_cache.dim(1)?;

        let key_cache = kv_cache
            .i(0)?
            .reshape((num_blocks, num_kv_heads, head_size / x, (), x))?;
        let value_cache = kv_cache
            .i(1)?
            .reshape((num_blocks, num_kv_heads, head_size, ()))?;

        Ok((key_cache, value_cache))
    }

    /// Initiates a swap blocks operation on the current CUDA device
    pub fn swap_blocks(
        src_kv_cache: Tensor,
        dst_kv_cache: Tensor,
        src_to_dst: Tensor,
    ) -> Result<(), CandleError> {
        match src_kv_cache.dtype() {
            DType::F16 => swap_blocks_t::<f16>(src_kv_cache, dst_kv_cache, src_to_dst),
            DType::BF16 => swap_blocks_t::<bf16>(src_kv_cache, dst_kv_cache, src_to_dst),
            DType::F32 => swap_blocks_t::<f32>(src_kv_cache, dst_kv_cache, src_to_dst),
            _ => candle_core::bail!(
                "Only f16, bf16 and f32 is supported for paged attention `swap_blocks`"
            ),
        }
    }

    pub fn copy_blocks(kv_caches: Vec<Tensor>, block_mapping: Tensor) -> Result<(), CandleError> {
        match kv_caches[0].dtype() {
            DType::F16 => copy_blocks_t::<f16>(kv_caches, block_mapping),
            DType::BF16 => copy_blocks_t::<bf16>(kv_caches, block_mapping),
            DType::F32 => copy_blocks_t::<f32>(kv_caches, block_mapping),
            _ => candle_core::bail!(
                "Only f16, bf16 and f32 is supported for paged attention `copy_blocks`"
            ),
        }
    }

    /// Forward pass with Paged Attention.
    ///
    /// Arguments shapes:
    /// - `query`: [num_tokens, num_heads * head_size]
    /// - `key`: [num_tokens, num_kv_heads * head_size]
    /// - `value`: [num_tokens, num_kv_heads * head_size]
    /// - `kv_cache`: [2, num_blocks, block_size * num_kv_heads * head_size]
    ///
    /// Output shape:
    /// [num_tokens, num_heads * head_size]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        kv_cache: &Tensor,
        attention_metadata: &mut PagedAttentionMetadata,
    ) -> Result<Tensor, CandleError> {
        let (num_tokens, _hidden_size) = query.dims2()?;
        // Reshape the query, key and value tensors to [num_tokens, num_heads, head_size]
        let query = query.reshape((num_tokens, self.num_attention_heads, self.head_dim))?;
        let key = key.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;
        let value = value.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;

        let (key_cache, value_cache) =
            Self::split_kv_cache(kv_cache, self.num_kv_heads, self.head_dim)?;

        reshape_and_cache(
            &key,
            &value,
            &key_cache,
            &value_cache,
            &attention_metadata.slot_mapping,
            1.0,
        )?;

        let output = if attention_metadata.is_prompt {
            debug_assert!(
                attention_metadata.sequence_lengths.len() > 0,
                "sequence_lengths should be available, for prefill sequences"
            );
            debug_assert!(attention_metadata
                .block_tables
                .elem_count() == 0,
                "block_tables should be empty, for prefill sequences as we do not support prefix decoding");

            let (key, value) = if self.num_kv_heads != self.num_attention_heads {
                (
                    key.repeat(&[1, self.num_queries_per_kv])?,
                    value.repeat(&[1, self.num_queries_per_kv])?,
                )
            } else {
                (key, value)
            };

            if attention_metadata.attention_bias.is_empty() {
                let attention_masks = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
                    utils::make_alibi_bias(
                        alibi_slopes,
                        query.dtype(),
                        &attention_metadata.sequence_lengths,
                    )?
                } else if let Some(sliding_window) = self.sliding_window {
                    utils::make_sliding_window_bias(
                        &attention_metadata.sequence_lengths,
                        sliding_window,
                        query.dtype(),
                        &query.device(),
                    )?
                } else {
                    vec![None; attention_metadata.sequence_lengths.len()]
                };
                attention_metadata.attention_bias = attention_masks;
            }

            let query = utils::move_dim_to_second_last(&query)?;
            let key = utils::move_dim_to_second_last(&key)?;
            let value = utils::move_dim_to_second_last(&value)?;

            let mut start_index = 0;
            let output = Tensor::zeros(
                &[num_tokens, self.num_attention_heads, self.head_dim],
                query.dtype(),
                query.device(),
            )?;

            for (seq_len, mask) in attention_metadata
                .sequence_lengths
                .iter()
                .zip(attention_metadata.attention_bias.iter())
            {
                let end_index = start_index + seq_len;

                // Scaled dot product attention
                let attention = (query
                    .narrow(1, start_index, *seq_len)?
                    .unsqueeze(0)?
                    .matmul(&key.narrow(1, start_index, *seq_len)?.unsqueeze(0)?.t()?)?
                    * self.scale as f64)?;
                let attention = if let Some(mask) = mask {
                    attention.broadcast_add(&mask)?
                } else {
                    attention
                };
                let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
                let attention = attention.matmul(&value)?;
                output.slice_assign(
                    &[start_index..end_index, 0..output.dim(1)?, 0..output.dim(2)?],
                    &attention,
                )?;

                start_index = end_index;
            }

            output
        } else {
            paged_attention(
                &query,
                &key_cache,
                &value_cache,
                &attention_metadata.block_tables,
                &attention_metadata.sequence_lens_tensor,
                attention_metadata.max_sequence_length,
                attention_metadata.kv_cache_dtype.clone(),
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes.clone(),
                1.0, // TODO: support kv_scale in the future, for quantized KV cache
            )?
        };

        output.reshape(((), self.num_attention_heads * self.head_dim))
    }

    // pub fn forward(
    //     &self,
    //     query: &Tensor,
    //     key: &Tensor,
    //     value: &Tensor,
    //     attention_mask: Option<&Tensor>,
    //     mut key_cache: Option<&Tensor>,
    //     mut value_cache: Option<&Tensor>,
    //     attention_metadata: &mut PagedAttentionMetadata,
    // ) -> Result<Tensor, CandleError> {
    //     let dims = attention_metadata.slot_mapping.dims();

    //     let slot_mapping = if dims.len() > 1 {
    //         attention_metadata
    //             .slot_mapping
    //             .flatten(0, attention_metadata.slot_mapping.dims().len())?
    //     } else {
    //         attention_metadata.slot_mapping.clone()
    //     };

    //     let attention = match attention_mask {
    //         None => None,
    //         Some(attention_mask) => {
    //             let attention = (query.matmul(&key.t()?)? * self.scale as f64)?;
    //             let attention = attention.broadcast_add(attention_mask)?;
    //             let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
    //             Some(attention.matmul(&value)?)
    //         }
    //     };

    //     // Attention has been already computed
    //     if let Some(computed_attention) = attention {
    //         // prefill prompts, as both the key and value caches are None
    //         return Ok(computed_attention);
    //     }

    //     // paged attention expects [b_sz, seq_len, nheads, head_dim]
    //     let query = query.transpose(1, 2)?.contiguous()?;
    //     let key = key.transpose(1, 2)?.contiguous()?;
    //     let value = value.transpose(1, 2)?.contiguous()?;

    //     // format [batch_size, num_tokens, num_heads, head_size]
    //     let (batch_size, seq_len, attention_heads, head_size) = query.shape().dims4()?;
    //     let (_, _, key_value_heads, _) = key.shape().dims4()?;
    //     let query = query.reshape(((), attention_heads, head_size))?;
    //     let key = key.reshape(((), key_value_heads, head_size))?;
    //     let value = value.reshape(((), key_value_heads, head_size))?;

    //     // key: Tensor,              // [num_tokens, num_heads, head_size]
    //     // value: Tensor,            // [num_tokens, num_heads, head_size]
    //     // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
    //     // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
    //     // slot_mapping: Tensor,     // [num_tokens]
    //     if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
    //         let _ = reshape_and_cache(
    //             &key,
    //             &value,
    //             &key_cache.as_mut().unwrap(),
    //             &value_cache.as_mut().unwrap(),
    //             &slot_mapping,
    //             1.0, // TODO: support kv_scale in the future, for quantized KV cache
    //         )?;
    //     }

    //     // At this point, we are
    //     let key_cache =
    //         key_cache.expect("key_cache should be available, for decode only sequences");
    //     let value_cache =
    //         value_cache.expect("value_cache should be available, for decode only sequences");
    //     let block_tables = attention_metadata
    //         .block_tables
    //         .as_ref()
    //         .expect("block_tables should be available, for decode only sequences");
    //     let sequence_lengths = attention_metadata
    //         .sequence_lengths
    //         .as_ref()
    //         .expect("sequence_lengths should be available, for decode only sequences");
    //     let max_sequence_length = attention_metadata
    //         .max_sequence_length
    //         .expect("Max sequence length should be available, for decode only sequences");

    //     paged_attention(
    //         &query,
    //         key_cache,
    //         value_cache,
    //         block_tables,
    //         sequence_lengths,
    //         max_sequence_length,
    //         attention_metadata.kv_cache_dtype.clone(),
    //         self.num_kv_heads,
    //         self.scale,
    //         self.alibi_slopes.clone(),
    //         1.0, // TODO: support kv_scale in the future, for quantized KV cache
    //     )
    // }
}

/// Swaps blocks in the key and value cache from CPU to GPU,
/// or GPU to CPU.
fn swap_blocks_t<T: CudaDType + DeviceRepr + WithDType>(
    src_kv_cache: Tensor,
    dst_kv_cache: Tensor,
    src_to_dst: Tensor,
) -> Result<(), CandleError> {
    let (block_mapping_storage, _block_mapping_layout) = src_to_dst.storage_and_layout();
    let block_mapping = match &*block_mapping_storage {
        Storage::Cpu(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    let block_mapping_slice = block_mapping.as_slice::<i64>()?;
    let block_mapping_ptr = block_mapping_slice.as_ptr() as *const core::ffi::c_void;

    match (src_kv_cache.device(), dst_kv_cache.device()) {
        (Device::Cuda(src_device), Device::Cuda(dst_device)) => {
            if src_device.ordinal() != dst_device.ordinal() {
                candle_core::bail!(
                    "Source and destiny KV cache tensors must be on the same GPU device"
                );
            }
            let (source_storage, source_layout) = src_kv_cache.storage_and_layout();
            let source = match &*source_storage {
                Storage::Cuda(storage) => storage,
                _ => candle_core::bail!("Only CUDA storage is supported"),
            };

            let (destiny_storage, destiny_layout) = dst_kv_cache.storage_and_layout();
            let destiny = match &*destiny_storage {
                Storage::Cuda(storage) => storage,
                _ => candle_core::bail!("Only CUDA storage is supported"),
            };

            let source_slice = source.as_cuda_slice::<T>()?;
            let destiny_slice = destiny.as_cuda_slice::<T>()?;

            let source_view = source_slice.slice(source_layout.start_offset()..);
            let destiny_view = destiny_slice.slice(destiny_layout.start_offset()..);

            let source_ptr = *source_view.device_ptr() as *mut core::ffi::c_void;
            let destiny_ptr = *destiny_view.device_ptr() as *mut core::ffi::c_void;

            unsafe { swap_blocks(source_ptr, destiny_ptr, block_mapping_ptr) }
        }
        (Device::Cpu, Device::Cuda(_)) => {
            let (source_storage, _) = src_kv_cache.storage_and_layout();
            let source = match &*source_storage {
                Storage::Cpu(storage) => storage,
                _ => candle_core::bail!("Source tensor storage should be available on CUDA device"),
            };

            let (destiny_storage, destiny_layout) = dst_kv_cache.storage_and_layout();
            let destiny = match &*destiny_storage {
                Storage::Cuda(storage) => storage,
                _ => candle_core::bail!("Destiny tensor storage should be available on CPU device"),
            };

            let source_slice = source.as_slice::<T>()?;
            let destiny_slice = destiny.as_cuda_slice::<T>()?;

            let destiny_view = destiny_slice.slice(destiny_layout.start_offset()..);

            let source_ptr = source_slice.as_ptr() as *mut T as *mut core::ffi::c_void;
            let destiny_ptr = *destiny_view.device_ptr() as *mut core::ffi::c_void;

            unsafe { swap_blocks(source_ptr, destiny_ptr, block_mapping_ptr) }
        }
        (Device::Cuda(_), Device::Cpu) => {
            let (source_storage, source_layout) = src_kv_cache.storage_and_layout();
            let source = match &*source_storage {
                Storage::Cuda(storage) => storage,
                _ => candle_core::bail!("Source tensor storage should be available on CUDA device"),
            };

            let (destiny_storage, _) = dst_kv_cache.storage_and_layout();
            let destiny = match &*destiny_storage {
                Storage::Cpu(storage) => storage,
                _ => candle_core::bail!("Destiny tensor storage should be available on CPU device"),
            };

            let source_slice = source.as_cuda_slice::<T>()?;
            let destiny_slice = destiny.as_slice::<T>()?;

            let source_view = source_slice.slice(source_layout.start_offset()..);

            let source_ptr = *source_slice.device_ptr() as *mut core::ffi::c_void;
            let destiny_ptr = destiny_slice.as_ptr() as *mut T as *mut core::ffi::c_void;

            unsafe { swap_blocks(source_ptr, destiny_ptr, block_mapping_ptr) }
        }
        _ => candle_core::bail!("Only CPU and CUDA devices are supported"),
    }
    Ok(())
}

fn copy_blocks_t<T: CudaDType + DeviceRepr>(
    kv_caches: Vec<Tensor>,
    block_mapping: Tensor,
) -> Result<(), CandleError> {
    if kv_caches.len() == 0 {
        return Ok(());
    }

    // 1. Handle block mapping tensor
    let (block_mapping, block_mapping_layout) = block_mapping.storage_and_layout();
    let block_mapping = match &*block_mapping {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    // Get CUDA slices for block_mapping tensor
    let block_mapping_slice = block_mapping.as_cuda_slice::<T>()?;
    let block_mapping_view = block_mapping_slice.slice(block_mapping_layout.start_offset()..);

    // Extract block_mapping pointer
    let block_mapping_ptr = *block_mapping_view.device_ptr() as *const core::ffi::c_void;

    let key_caches = kv_caches
        .iter()
        .map(|t| t.i(0))
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches = kv_caches
        .iter()
        .map(|t| t.i(1))
        .collect::<Result<Vec<_>, _>>()?;

    // 2. Handle key_caches and value_caches tensors
    let key_caches = key_caches
        .iter()
        .map(|t| t.storage_and_layout())
        .collect::<Vec<_>>();
    let value_caches = value_caches
        .iter()
        .map(|t| t.storage_and_layout())
        .collect::<Vec<_>>();

    // Get CUDA slices for all tensors
    let key_caches_slices = key_caches
        .iter()
        .map(|(storage, layout)| match &**storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>().map(|s| (s, layout)),
            _ => candle_core::bail!("Only CUDA storage is supported"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches_slices = value_caches
        .iter()
        .map(|(storage, layout)| match &**storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>().map(|s| (s, layout)),
            _ => candle_core::bail!("Only CUDA storage is supported"),
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Get CUDA views for all tensors
    let key_caches_views = key_caches_slices
        .iter()
        .map(|(slice, layout)| slice.slice(layout.start_offset()..))
        .collect::<Vec<_>>();
    let value_caches_views = value_caches_slices
        .iter()
        .map(|(slice, layout)| slice.slice(layout.start_offset()..))
        .collect::<Vec<_>>();

    // Get pointers to key_caches and value_caches
    let key_caches_ptrs = key_caches_views
        .iter()
        .map(|v| *v.device_ptr() as *const core::ffi::c_void)
        .collect::<Vec<_>>()
        .as_ptr() as *const *const core::ffi::c_void;
    let value_caches_ptrs = value_caches_views
        .iter()
        .map(|v| *v.device_ptr() as *const core::ffi::c_void)
        .collect::<Vec<_>>()
        .as_ptr() as *const *const core::ffi::c_void;

    unsafe {
        copy_blocks(
            key_caches_ptrs as *const *const core::ffi::c_void,
            value_caches_ptrs as *const *const core::ffi::c_void,
            block_mapping_ptr,
        )
    }

    Ok(())
}

mod utils {
    use super::*;

    pub(crate) fn make_alibi_bias(
        alibi_slopes: &Tensor,
        dtype: DType,
        sequence_lengths: &Vec<usize>,
    ) -> Result<Vec<Option<Tensor>>, CandleError> {
        let mut biases = vec![];
        for &seq_len in sequence_lengths {
            let bias =
                Tensor::arange(0, seq_len as i64, alibi_slopes.device())?.to_dtype(DType::I64)?;
            let bias_2d = bias.unsqueeze(0)?.broadcast_sub(&bias.unsqueeze(1)?)?;
            let num_heads = alibi_slopes.dim(0)?;
            let bias_3d = bias_2d.unsqueeze(0)?.repeat((num_heads, 1, 1))?;
            let slopes_3d = alibi_slopes.unsqueeze(1)?.unsqueeze(2)?;
            let bias_with_slopes = bias_3d.mul(&slopes_3d)?;
            let inf_mask = Tensor::full(
                f32::NEG_INFINITY,
                &[1, seq_len, seq_len],
                alibi_slopes.device(),
            )?
            .to_dtype(dtype)?;
            let inf_mask = Tensor::triu2(seq_len, dtype, &alibi_slopes.device())?
                .unsqueeze(0)?
                .broadcast_mul(&Tensor::new(f32::NEG_INFINITY, &alibi_slopes.device())?)?;
            let final_bias = bias_with_slopes.broadcast_add(&inf_mask)?;
            biases.push(Some(final_bias.to_dtype(dtype)?))
        }
        Ok(biases)
    }

    pub(crate) fn make_sliding_window_bias(
        sequence_lengths: &Vec<usize>,
        window_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Vec<Option<Tensor>>, CandleError> {
        let mut biases = vec![];
        for &seq_len in sequence_lengths {
            let bias = Tensor::full(1.0, &[1, seq_len, seq_len], device)?.to_dtype(dtype)?;
            let bias_tril = Tensor::tril2(seq_len, dtype, device)?
                .broadcast_mul(&Tensor::new(f32::NEG_INFINITY, device)?)?;
            let mask = Tensor::tril2(seq_len, dtype, device)?
                .unsqueeze(0)?
                .broadcast_mul(&bias_tril)?;
            let mask = mask.broadcast_mul(&Tensor::triu2(seq_len, dtype, device)?.unsqueeze(0)?)?;
            let mask = mask.log()?;
            biases.push(Some(mask.to_dtype(dtype)?))
        }
        Ok(biases)
    }

    pub(crate) fn move_dim_to_second_last(tensor: &Tensor) -> Result<Tensor, CandleError> {
        let dims = tensor.dims();
        let ndim = dims.len();

        if ndim <= 2 {
            return Ok(tensor.clone());
        }

        let mut permutation: Vec<usize> = (1..ndim).collect();
        permutation.swap(0, ndim - 2);

        // Apply the permutation
        tensor.permute(permutation)
    }
}
