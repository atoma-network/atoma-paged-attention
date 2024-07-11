pub const ATTENTION_KERNELS: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "./attention/attention_kernels.ptx"
));
pub const CACHE_KERNELS: &str = include_str!(concat!(env!("OUT_DIR"), "./cache_kernels.ptx"));
pub mod ffi;
