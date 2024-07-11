use std::ffi::{c_int, CString};

use candle_core::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{DevicePtr, DeviceRepr},
        CudaDType, WrapErr,
    },
    CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor,
};
use csrc::ffi::{paged_attention_v1, paged_attention_v2};
use half::{bf16, f16};

const PARTITION_SIZE: usize = 512;

/// `PagedAttention` - Backend to run
/// Paged Attention based attention cuda kernels
pub struct PagedAttention {
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    sequence_lengths: Tensor,
    max_sequence_length: usize,
    kv_cache_dtype: String,
    num_kv_heads: i64,
    scale: f64,
    alibi_slopes: Option<Tensor>,
    kv_scale: f64,
}

impl CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("PagedAttention is not implemented for CPU");
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        match storage.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(storage, layout),
            DType::F16 => self.cuda_fwd_t::<f16>(storage, layout),
            DType::BF16 => self.cuda_fwd_t::<bf16>(storage, layout),
            dtype => candle_core::bail!("Unsupported dtype for paged attention: {dtype:?}"),
        }
    }
}

impl PagedAttention {
    // #[cfg(feature = "cuda")]
    fn cuda_fwd_t<T: CudaDType + DeviceRepr>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = storage.dtype();
        let internal_type = match dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            _ => candle_core::bail!("Unsupported dtype for paged attention: {dtype:?}"),
        };

        let device = storage.device();
        let output_shape = layout.shape();

        let (key_cache, key_cache_layout) = self.key_cache.storage_and_layout();
        let key_cache = match &*key_cache {
            Storage::Cuda(kc) => kc,
            _ => candle_core::bail!("key_cache must be a Cuda tensor"),
        };

        let (value_cache, value_cache_layout) = self.value_cache.storage_and_layout();
        let value_cache = match &*value_cache {
            Storage::Cuda(vc) => vc,
            _ => candle_core::bail!("value_cache must be a Cuda tensor"),
        };

        let (block_tables, block_tables_layout) = self.block_tables.storage_and_layout();
        let block_tables = match &*block_tables {
            Storage::Cuda(bt) => bt,
            _ => candle_core::bail!("block_tables must be a Cuda tensor"),
        };

        let (sequence_lengths, sequence_lengths_layout) =
            self.sequence_lengths.storage_and_layout();
        let sequence_lengths = match &*sequence_lengths {
            Storage::Cuda(sl) => sl,
            _ => candle_core::bail!("sequence_lengths must be a Cuda tensor"),
        };

        // let (query, query_layout) = self.query.storage_and_layout();
        // let query = match &*&query {
        //     Storage::Cuda(q) => q,
        //     _ => candle_core::bail!("query must be a Cuda tensor"),
        // };

        let q_rank = layout.stride().len();
        let key_cache_rank = key_cache_layout.stride().len();
        let value_cache_rank = value_cache_layout.stride().len();

        if q_rank != 3 {
            candle_core::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {layout:?})"
            )
        }

        if key_cache_rank != 5 {
            candle_core::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {key_cache_layout:?})"
            )
        }

        if value_cache_rank != 4 {
            candle_core::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {value_cache_layout:?})"
            )
        }

        // Get cuda slice for all tensors
        let q = storage.as_cuda_slice::<T>()?;
        let key_cache = key_cache.as_cuda_slice::<T>()?;
        let value_cache = value_cache.as_cuda_slice::<T>()?;
        // TODO: can we downcast to i32/u32 to reduce memory usage?
        let block_tables = block_tables.as_cuda_slice::<i64>()?;
        // TODO: can we downcast to i32/u32 to reduce memory usage?
        let sequence_lengths = sequence_lengths.as_cuda_slice::<i64>()?;

        // Get cuda views for all tensors
        let q = q.slice(layout.start_offset()..);
        let key_cache = key_cache.slice(key_cache_layout.start_offset()..);
        let value_cache = value_cache.slice(value_cache_layout.start_offset()..);
        let block_tables = block_tables.slice(block_tables_layout.start_offset()..);
        let sequence_lengths = sequence_lengths.slice(sequence_lengths_layout.start_offset()..);

        let (num_sequences, num_heads, head_size) = layout.shape().dims3()?;

        if !matches!(head_size, 64 | 80 | 96 | 112 | 128 | 256) {
            candle_core::bail!("`head_size` must be one of 64, 80, 96, 112, 128 or 256");
        }

        let (num_sequences_block_table, _max_num_blocks_per_sequence) =
            (block_tables_layout.dims()[0], block_tables_layout.dims()[1]);

        if num_sequences_block_table != num_sequences {
            candle_core::bail!(
                "block_tables shape mismatch {:?}, expected {:?}",
                block_tables_layout.shape(),
                (num_sequences, num_sequences_block_table)
            );
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) =
            key_cache_layout.shape().dims5()?;
        if head_size_kc != head_size / x {
            candle_core::bail!(
                "key_cache shape mismatch {:?}, expected {:?}",
                key_cache_layout.shape(),
                (num_blocks, num_kv_heads, head_size / x, block_size, x)
            );
        }

        if (num_blocks, num_kv_heads, head_size, block_size)
            != value_cache_layout.shape().dims4()?
        {
            candle_core::bail!(
                "value_cache shape mismatch {:?} key_cache shape {:?}",
                value_cache_layout.shape(),
                key_cache_layout.shape()
            );
        }

        if num_sequences != sequence_lengths_layout.shape().dims1()? {
            candle_core::bail!(
                "sequence_lengths shape mismatch {:?}, expected {:?}",
                sequence_lengths_layout.shape(),
                num_sequences
            );
        }

        let max_num_partitions = (self.max_sequence_length + PARTITION_SIZE - 1) / PARTITION_SIZE;

        // We use a simple heuristic to decide whether to use
        // PagedAttention V1 or V2. If the number of partitions is 1, we use
        // V1 to avoid the overhead of reduction. Also, if the number of
        // sequences or heads is large, we use V1 since there is enough work
        // to parallelize.
        let use_v1 = (max_num_partitions == 1 || num_sequences * num_heads > PARTITION_SIZE)
            && PARTITION_SIZE % block_size == 0;

        let elem_count = output_shape.elem_count();
        let out = unsafe { device.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *mut core::ffi::c_void;
        let query_ptr = *q.device_ptr() as *mut core::ffi::c_void;
        let key_cache_ptr = *key_cache.device_ptr() as *mut core::ffi::c_void;
        let value_cache_ptr = *value_cache.device_ptr() as *mut core::ffi::c_void;
        let block_tables_ptr = *block_tables.device_ptr() as *mut core::ffi::c_void;
        let sequence_lengths_ptr = *sequence_lengths.device_ptr() as *mut core::ffi::c_void;

        let alibi_slopes_ptrs = self
            .alibi_slopes
            .as_ref()
            .map(|t| {
                let (t, layout) = t.storage_and_layout();
                let t = match &*t {
                    Storage::Cuda(s) => s.as_cuda_slice::<T>(),
                    _ => Err(candle_core::Error::Msg(format!(
                        "Only CUDA storage is supported"
                    ))),
                }?;
                let t = t.slice(layout.start_offset()..);
                Ok::<_, candle_core::Error>(*t.device_ptr() as *const core::ffi::c_void)
            })
            .transpose()?;

        if use_v1 {
            unsafe {
                paged_attention_v1(
                    out_ptr,
                    query_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    self.num_kv_heads,
                    self.scale,
                    block_tables_ptr,
                    sequence_lengths_ptr,
                    block_size as i64,
                    self.max_sequence_length as i64,
                    alibi_slopes_ptrs,
                    internal_type as *const i8,
                    self.kv_scale,
                    0,
                    0,
                    64,
                    0,
                    0,
                )
            };
        } else {
            let temp_out_shape =
                Shape::from((num_sequences, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_sequences, num_heads, max_num_partitions));

            let tmp_out = unsafe { device.alloc::<T>(temp_out_shape.elem_count()) }.w()?;
            let exp_sums = unsafe { device.alloc::<T>(exp_sums_shape.elem_count()) }.w()?;
            let max_logits = unsafe { device.alloc::<T>(exp_sums_shape.elem_count()) }.w()?;

            let tmp_out_ptr = *tmp_out.device_ptr() as *mut core::ffi::c_void;
            let exp_sums_ptr = *exp_sums.device_ptr() as *mut core::ffi::c_void;
            let max_logits_ptr = *max_logits.device_ptr() as *mut core::ffi::c_void;

            unsafe {
                paged_attention_v2(
                    out_ptr,
                    exp_sums_ptr,
                    max_logits_ptr,
                    tmp_out_ptr,
                    query_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    self.num_kv_heads,
                    self.scale,
                    block_tables_ptr,
                    sequence_lengths_ptr,
                    block_size as i64,
                    self.max_sequence_length as i64,
                    alibi_slopes_ptrs,
                    internal_type as *const i8,
                    self.kv_scale,
                    0,
                    0,
                    64,
                    0,
                    0,
                )
            };
        }

        let output = CudaStorage::wrap_cuda_slice(out, device.clone());
        Ok((output, output_shape.clone()))
    }
}

/// Computes a forward pass of the PagedAttention operator. The latter
/// is a scaled dot product `softmax(Q @ K^T * scale) @ V` where `Q`, `K`
/// and`V` are the query, key and value tensors respectively.
///
/// Multi-query and grouped-query attention is supported by using `key_cache`
/// and `value_cache` tensors with fewer heads than `Q`. The number of heads
/// in `K` and `V` has to be divisible by the number of heads in `Q`.
///
/// Arguments:
///
/// `query` - Query tensor with shape `[num_sequences, num_heads_q, head_size]`.
/// `key_cache` - Key cache paged tensor of shape `[num_blocks, num_heads_kv, head_size / x, block_size, x]`
///     with `x` being the size of an element in bytes.
/// `value_cache` - Value cache paged tensor of shape `[num_blocks, num_heads_kv, head_size, block_size]`.
/// `block_tables` - Padded table associating blocks to each sequence of shape `[num_sequences, max_context_len // block_size]`
/// `sequence_lengths` - Tensor associating lengths to each sequence of shape `[num_sequences]`
/// `max_sequence_length` - Maximum value in `sequence_lengths`
/// `scale` - Softmax scaling factor
///
/// The resulting tensor has dimensions `[num_sequences, num_heads_q, head_size]`.
pub fn paged_attention(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    sequence_lengths: &Tensor,
    max_sequence_length: usize,
    kv_cache_dtype: String,
    num_kv_heads: usize,
    scale: f64,
    alibi_slopes: Option<Tensor>,
    kv_scale: f64,
) -> Result<Tensor> {
    let attention = PagedAttention {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        sequence_lengths: sequence_lengths.clone(),
        max_sequence_length,
        kv_cache_dtype,
        num_kv_heads: num_kv_heads as i64,
        scale,
        alibi_slopes,
        kv_scale,
    };
    query.apply_op1(attention)
}

/// Updates the intermediate Key and Value cache
/// results for paged attention forward pass.
fn reshape_and_cache_t<T: CudaDType + DeviceRepr>(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    kv_scale: f64,
) -> Result<()> {
    let (key_storage, key_layout) = key.storage_and_layout();
    let key = match &*key_storage {
        Storage::Cuda(k) => k,
        _ => candle_core::bail!("key_cache must be a Cuda tensor"),
    };

    let (value_storage, value_layout) = value.storage_and_layout();
    let value = match &*value_storage {
        Storage::Cuda(v) => v,
        _ => candle_core::bail!("value_cache must be a Cuda tensor"),
    };

    let (key_cache_storage, key_cache_layout) = key_cache.storage_and_layout();
    let key_cache = match &*key_cache_storage {
        Storage::Cuda(kc) => kc,
        _ => candle_core::bail!("key_cache must be a Cuda tensor"),
    };

    let (value_cache_storage, value_cache_layout) = value_cache.storage_and_layout();
    let value_cache = match &*value_cache_storage {
        Storage::Cuda(vc) => vc,
        _ => candle_core::bail!("value_cache must be a Cuda tensor"),
    };

    let (slot_mapping, slot_mapping_layout) = slot_mapping.storage_and_layout();
    let slot_mapping = match &*slot_mapping {
        Storage::Cuda(sm) => sm,
        _ => candle_core::bail!("slot_mapping must be a Cuda tensor"),
    };

    let key_rank = key_layout.stride().len();
    let value_rank = value_layout.stride().len();
    let key_cache_rank = key_cache_layout.stride().len();
    let value_cache_rank = value_cache_layout.stride().len();

    if key_rank != 3 || value_rank != 3 {
        candle_core::bail!(
            "paged-attention expects `key` tensor to be of rank 3 \
            (key: {key_layout:?}, value: {value_layout:?})"
        )
    }

    if key_cache_rank != 5 {
        candle_core::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
            (key_cache: {key_cache_layout:?})"
        )
    }

    if value_cache_rank != 4 {
        candle_core::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
            (value_cache: {value_cache_layout:?})"
        )
    }

    // Get CUDA slices for all tensors
    let key_slice = key.as_cuda_slice::<T>()?;
    let value_slice = value.as_cuda_slice::<T>()?;
    let key_cache_slice = key_cache.as_cuda_slice::<T>()?;
    let value_cache_slice = value_cache.as_cuda_slice::<T>()?;
    let slot_mapping_slice = slot_mapping.as_cuda_slice::<i64>()?;

    // Get CUDA views for all tensors
    let key_view = key_slice.slice(key_layout.start_offset()..);
    let value_view = value_slice.slice(value_layout.start_offset()..);
    let key_cache_view = key_cache_slice.slice(key_cache_layout.start_offset()..);
    let value_cache_view = value_cache_slice.slice(value_cache_layout.start_offset()..);
    let slot_mapping_view = slot_mapping_slice.slice(slot_mapping_layout.start_offset()..);

    let (num_tokens, num_heads, head_size) = key_layout.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != (value_layout.shape().dims3()?) {
        candle_core::bail!(
            "paged-attention expects `key` and `value` tensors to have the same shape \
            (key: {key_layout:?}, value: {value_layout:?})"
        )
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) =
        key_cache_layout.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        candle_core::bail!(
            "paged-attention shape mismatch value_cache {:?}, expected {:?}",
            value_cache_layout,
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_heads, head_size, block_size) != value_cache_layout.shape().dims4()? {
        candle_core::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            key_cache_layout.shape(),
            value_cache_layout.shape()
        )
    }

    if num_tokens != slot_mapping_layout.shape().dims1()? {
        candle_core::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            slot_mapping_layout.shape(),
            (num_tokens)
        )
    }

    let k_ptr = *key_view.device_ptr() as *mut core::ffi::c_void;
    let v_ptr = *value_view.device_ptr() as *mut core::ffi::c_void;
    let kc_ptr = *key_cache_view.device_ptr() as *mut core::ffi::c_void;
    let vc_ptr = *value_cache_view.device_ptr() as *mut core::ffi::c_void;
    let s_ptr = *slot_mapping_view.device_ptr() as *mut core::ffi::c_void;
    // TODO: allow for different dtypes
    let kv_cache_dtype = CString::new("auto").expect("CString::new failed");
    let kv_cache_dtype = kv_cache_dtype.as_ptr();

    unsafe {
        csrc::ffi::reshape_and_cache(
            k_ptr,
            v_ptr,
            kc_ptr,
            vc_ptr,
            s_ptr,
            kv_cache_dtype,
            kv_scale,
        );
    };

    Ok(())
}

/// Inserts key and value blocks at the provided slot mapping inside the key value paged attention
/// cache
///
/// Arguments:
///
/// `key` - Key tensor of shape `(num_tokens, num_heads, head_size)`.
/// `value` - Value tensor of shape `(num_tokens, num_heads, head_size)`.
/// `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads, head_size / x, block_size, x)`
///     with `x` being the size of an element in bytes.
/// `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads, head_size, block_size)`.
/// `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    kv_scale: f64,
) -> Result<()> {
    match key_cache.dtype() {
        DType::F16 => {
            reshape_and_cache_t::<f16>(key, value, key_cache, value_cache, slot_mapping, kv_scale)
        }
        DType::BF16 => {
            reshape_and_cache_t::<bf16>(key, value, key_cache, value_cache, slot_mapping, kv_scale)
        }
        DType::F32 => {
            reshape_and_cache_t::<f32>(key, value, key_cache, value_cache, slot_mapping, kv_scale)
        }
        _ => candle_core::bail!(
            "Unsupported data type of key_cache: {:?}",
            key_cache.dtype()
        ),
    }
}
