use crate::{
    backend::reshape_and_cache,
    kernels::ffi::{copy_blocks, swap_blocks},
};
use candle_core::{
    cuda::cudarc::driver::CudaSlice,
    cuda_backend::{cudarc::driver::DeviceRepr, CudaDType},
    DType, Device, Error as CandleError, IndexOp, Layout, Storage, Tensor, D,
};
use half::{bf16, f16};

/// `PagedAttentionMetadata` - Structure wrapping the metadata
/// required for paged attention
pub struct PagedAttentionMetadata {
    /// Lengths of prompts
    pub prompt_lengths: Vec<usize>,
    /// The maximum sequence length
    pub max_sequence_length: Option<usize>,
    /// The block tables. (sequence_id -> vector of physical blocks)
    pub block_tables: Option<Tensor>,
    /// The length of attention context for each generation token
    pub sequence_lengths: Option<Tensor>,
    /// The address to write the new KV to of each token
    pub slot_mapping: Tensor,
    // /// The attention bias
    // pub attn_bias: Option<Box<dyn AttentionBiasBlockDiagonal>>,
    /// Is a prefill prompt
    pub is_prompt: bool,
    /// KV cache datatype (auto or fp8_e5m2)
    pub kv_cache_dtype: String,
}

impl PagedAttentionMetadata {
    /// Constructor
    pub fn new(
        prompt_lengths: Vec<usize>,
        max_sequence_length: Option<usize>,
        block_tables: Option<Tensor>,
        sequence_lengths: Option<Tensor>,
        slot_mapping: Tensor,
        kv_cache_dtype: String,
    ) -> Self {
        let is_prompt = !prompt_lengths.is_empty();
        Self {
            prompt_lengths,
            max_sequence_length,
            block_tables,
            sequence_lengths,
            slot_mapping,
            // attn_bias: None,
            is_prompt,
            kv_cache_dtype,
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
    pub fn split_kv_cache() {
        todo!()
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

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<&Tensor>,
        mut value_cache: Option<&Tensor>,
        attention_metadata: &mut PagedAttentionMetadata,
    ) -> Result<Tensor, CandleError> {
        let dims = attention_metadata.slot_mapping.dims();

        let slot_mapping = if dims.len() > 1 {
            attention_metadata
                .slot_mapping
                .flatten(0, attention_metadata.slot_mapping.dims().len())?
        } else {
            attention_metadata.slot_mapping.clone()
        };

        let attention = match attention_mask {
            None => None,
            Some(attention_mask) => {
                let attention = (query.matmul(&key.t()?)? * self.scale as f64)?;
                let attention = attention.broadcast_add(attention_mask)?;
                let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
                Some(attention.matmul(&value)?)
            }
        };

        // paged attention expects [b_sz, seq_len, nheads, head_dim]
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value = value.transpose(1, 2)?.contiguous()?;

        // format [batch_size, num_tokens, num_heads, head_size]
        let (batch_size, seq_len, attention_heads, head_size) = query.shape().dims4()?;
        let (_, _, key_value_heads, _) = key.shape().dims4()?;
        let query = query.reshape(((), attention_heads, head_size))?;
        let key = key.reshape(((), key_value_heads, head_size))?;
        let value = value.reshape(((), key_value_heads, head_size))?;

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let _ = reshape_and_cache(
                &key,
                &value,
                &key_cache.as_mut().unwrap(),
                &value_cache.as_mut().unwrap(),
                &slot_mapping,
                self.scale,
            )?;
        }

        // Attention has been already computed
        if let Some(computed_attention) = attention {
            // prefill prompts
            return Ok(computed_attention);
        }
    }
}

fn swap_blocks_t<T: CudaDType + DeviceRepr>(
    src_kv_cache: Tensor,
    dst_kv_cache: Tensor,
    src_to_dst: Tensor,
) -> Result<(), CandleError> {
    // 1. Handle block mapping tensor
    let (block_mapping, block_mapping_layour) = src_to_dst.storage_and_layout();
    let block_mapping = match block_mapping {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    // Get CUDA slices for block_mapping tensor
    let block_mapping_slice = block_mapping.as_cuda_slice::<T>()?;
    let block_mapping_view = block_mapping_slice.slice(block_mapping_layour.start_offset()..)?;

    // 2. Handle source and destination key_cache tensor
    let src_key_cache = src_kv_cache.i(0)?;
    let dst_key_cache = dst_kv_cache.i(0)?;

    let (src_key_cache_storage, src_key_cache_layout) = src_key_cache.storage_and_layout();
    let src_key_cache = match src_key_cache_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    let (dst_key_cache_storage, dst_key_cache_layout) = dst_key_cache.storage_and_layout();
    let dst_key_cache = match dst_key_cache_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    // Get CUDA slices for both source and destiny key_cache tensors
    let src_key_cache_slice = src_key_cache.as_cuda_slice::<T>()?;
    let dst_key_cache_slice = dst_key_cache.as_cuda_slice::<T>()?;

    // Get CUDA views for all tensors
    let src_key_cache_view = src_key_cache_slice.slice(src_key_cache_layout.start_offset()..)?;
    let dst_key_cache_view = dst_key_cache_slice.slice(dst_key_cache_layout.start_offset()..)?;

    unsafe {
        swap_blocks(
            src_key_cache_view as *const core::ffi::c_void,
            dst_key_cache_view as *const core::ffi::c_void,
            block_mapping_view as *const core::ffi::c_void,
        )
    };

    // 3. Handle source and destination value_cache tensor
    let src_value_cache = src_kv_cache.i(1)?;
    let dst_value_cache = dst_kv_cache.i(1)?;

    let (src_value_cache_storage, src_value_cache_layout) = src_value_cache.storage_and_layout();
    let src_value_cache = match src_value_cache_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    let (dst_value_cache_storage, dst_value_cache_layout) = dst_value_cache.storage_and_layout();
    let dst_value_cache = match dst_value_cache_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    // Get CUDA slices for both source and destiny value_cache tensors
    let src_value_cache_slice = src_value_cache_storage.as_cuda_slice::<T>()?;
    let dst_value_cache_slice = dst_value_cache_storage.as_cuda_slice::<T>()?;

    // Get CUDA views for all tensors
    let src_value_cache_view =
        src_value_cache_slice.slice(src_value_cache_layout.start_offset()..)?;
    let dst_value_cache_view =
        dst_value_cache_slice.slice(dst_value_cache_layout.start_offset()..)?;

    unsafe {
        swap_blocks(
            src_value_cache_view as *const core::ffi::c_void,
            dst_value_cache_view as *const core::ffi::c_void,
            block_mapping_view as *const core::ffi::c_void,
        )
    };

    Ok(())
}

fn copy_blocks_t<T: CudaDType + DeviceRepr>(
    kv_caches: Vec<Tensor>,
    block_mapping: Tensor,
) -> Result<(), CandleError> {
    // 1. Handle block mapping tensor
    let (block_mapping, block_mapping_layout) = block_mapping.storage_and_layout();
    let block_mapping = match block_mapping {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("Only CUDA storage is supported"),
    };

    // Get CUDA slices for block_mapping tensor
    let block_mapping_slice = block_mapping.as_cuda_slice::<T>()?;
    let block_mapping_view = block_mapping_slice.slice(block_mapping_layout.start_offset()..)?;
    let key_caches = kv_caches
        .iter()
        .map(|t| t.i(0))
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches = kv_caches
        .iter()
        .map(|t| t.i(1))
        .collect::<Result<Vec<_>, _>>()?;

    let key_caches_length = key_caches.len();
    let value_caches_length = value_caches.len();

    // 2. Handle key_caches and value_caches tensors
    let key_caches = key_caches
        .iter()
        .map(|t| t.storage_and_layout())
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches = value_caches
        .iter()
        .map(|t| t.storage_and_layout())
        .collect::<Result<Vec<_>, _>>()?;

    // Get CUDA slices for all tensors
    let key_caches_slice = key_caches
        .iter()
        .map(|(storage, layout): &(Storage, Layout)| match storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>().map(|s| (s, layout)),
            _ => candle_core::bail!("Only CUDA storage is supported"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches_slice = value_caches
        .iter()
        .map(|(storage, layout): &(Storage, Layout)| match storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>().map(|s| (s, layout)),
            _ => candle_core::bail!("Only CUDA storage is supported"),
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Get CUDA views for all tensors
    let key_caches_view = key_caches_slice
        .iter()
        .map(|(slice, layout): &(CudaSlice<_>, Layout)| slice.slice(layout.start_offset()..))
        .collect::<Result<Vec<_>, _>>()?;
    let value_caches_view = value_caches_slice
        .iter()
        .map(|(slice, layout): &(CudaSlice<_>, Layout)| slice.slice(layout.start_offset()..))
        .collect::<Result<Vec<_>, _>>()?;

    unsafe {
        copy_blocks(
            key_caches_view as *const *const core::ffi::c_void,
            value_caches_view as *const *const core::ffi::c_void,
            block_mapping_view as *const core::ffi::c_void,
        )
    }

    Ok(())
}