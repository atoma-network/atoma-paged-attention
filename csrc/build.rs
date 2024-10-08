// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the ATOMA_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const KERNEL_FILES: [&str; 66] = [
    "kernels/cache_manager.cu",
    "kernels/flash_api.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_sm80.cu",
];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    println!("cargo:rerun-if-changed=kernels/rotary.h");
    println!("cargo:rerun-if-changed=kernels/alibi.h");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("ATOMA_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => PathBuf::from(build_dir)
            .canonicalize()
            .context("Failed to canonicalize build directory")?,
    };
    println!("cargo:warning=Build directory: {:?}", build_dir.display());

    compile_cuda_files(&build_dir)?;

    // Link libraries
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=msvcprt");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}

fn compile_cuda_files(build_dir: &Path) -> Result<()> {
    let kernels: Vec<_> = KERNEL_FILES.iter().map(|&s| s.to_string()).collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.to_path_buf())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Icutlass/include")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-w");

    println!("cargo:info={builder:?}");

    let out_file = if cfg!(target_os = "windows") {
        build_dir.join("flashattention.lib")
    } else {
        build_dir.join("libflashattention.a")
    };

    builder.build_lib(&out_file);

    Ok(())
}
