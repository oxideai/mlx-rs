use std::{path::{PathBuf, Path}, process::Command};

const MLX_DIR: &str = "mlx";

const FILES_MLX: &[&str] = &[
    "src/mlx.cpp", // TODO: remove this later if everything works
    "mlx/mlx/allocator.cpp",
    "mlx/mlx/array.cpp",
    "mlx/mlx/device.cpp",
    "mlx/mlx/dtype.cpp",
    "mlx/mlx/fft.cpp",
    "mlx/mlx/graph_utils.cpp",
    "mlx/mlx/load.cpp",
    "mlx/mlx/ops.cpp",
    "mlx/mlx/primitives.cpp",
    "mlx/mlx/random.cpp",
    "mlx/mlx/scheduler.cpp",
    "mlx/mlx/transforms.cpp",
    "mlx/mlx/utils.cpp",
];

/// Common files to compile for all backends
const FILES_MLX_BACKEND_COMMON: &[&str] = &[
    "mlx/mlx/backend/common/arg_reduce.cpp",
    "mlx/mlx/backend/common/binary.cpp",
    "mlx/mlx/backend/common/conv.cpp",
    "mlx/mlx/backend/common/copy.cpp",
    "mlx/mlx/backend/common/default_primitives.cpp",
    "mlx/mlx/backend/common/erf.cpp",
    "mlx/mlx/backend/common/fft.cpp",
    "mlx/mlx/backend/common/indexing.cpp",
    "mlx/mlx/backend/common/load.cpp",
    "mlx/mlx/backend/common/primitives.cpp",
    "mlx/mlx/backend/common/quantized.cpp",
    "mlx/mlx/backend/common/reduce.cpp",
    "mlx/mlx/backend/common/scan.cpp",
    "mlx/mlx/backend/common/softmax.cpp",
    "mlx/mlx/backend/common/sort.cpp",
    "mlx/mlx/backend/common/threefry.cpp",
];

/// Files to compile for accelerate backend
const FILES_MLX_BACKEND_ACCELERATE: &[&str] = &[
    "mlx/mlx/backend/accelerate/conv.cpp",
    "mlx/mlx/backend/accelerate/matmul.cpp",
    "mlx/mlx/backend/accelerate/primitives.cpp",
    "mlx/mlx/backend/accelerate/quantized.cpp",
    "mlx/mlx/backend/accelerate/reduce.cpp",
    "mlx/mlx/backend/accelerate/softmax.cpp",
];

const METAL_KERNEL_DIR: &str = "mlx/mlx/backend/metal/kernels";
const KERNEL_HEADERS: &[&str] = &[
    "bf16.h",
    "bf16_math.h",
    "complex.h",
    "defines.h",
    "erf.h",
    "reduce.h",
    "utils.h",
];
const KERNELS: &[&str] = &[
    "arange",
    "arg_reduce",
    "binary",
    "conv",
    "copy",
    "gemm",
    "gemv",
    "quantized",
    "random",
    "reduce",
    "scan",
    "softmax",
    "sort",
    "unary",
    "indexing",
];

fn main() {
    // TODO: conditionally compile based on if accelerate is available

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:warning=out_dir: {}", out_dir.display());

    #[cfg(feature = "metal")]
    {
        build_metal_kernels(&out_dir);
    }

    let mut build = cxx_build::bridge("src/main.rs");
        
    build.include(MLX_DIR)
        .include("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers")
        .flag("-std=c++17")
        .flag("-framework").flag("vecLib")
        .files(FILES_MLX)
        .files(FILES_MLX_BACKEND_COMMON);

    // TODO: check if accelerate is available
    #[cfg(feature = "accelerate")]
    {
        build
            .flag("-framework").flag("Accelerate")
            // mlx uses new lapack api if accelerate is available
            .flag("-DACCELERATE_NEW_LAPACK") 
            .files(FILES_MLX_BACKEND_ACCELERATE);
    }
        
    build.compile("mlx");

    println!("cargo:rerun-if-changed=src/main.rs");
    // println!("cargo:rustc-link-lib=dylib=mlx");
}

fn build_metal_kernels(out_dir: &PathBuf) {
    // test build kernel air
    let kernel_build_dir = PathBuf::from(out_dir).join("kernels");
    let _ = std::fs::create_dir_all(&kernel_build_dir);
    let kernel_src_dir = PathBuf::from(METAL_KERNEL_DIR);
    let kernel_airs = KERNELS.iter().map(|&k| {
        build_kernel_air(
            &kernel_build_dir, 
            &kernel_src_dir,
            MLX_DIR,
            k
        )
    }).collect::<Vec<_>>();
    build_kernel_metallib(&kernel_airs, out_dir);
}

fn build_kernel_air(
    kernel_build_dir: &PathBuf, 
    kernel_src_dir: &PathBuf,
    mlx_dir: impl AsRef<Path>,
    kernel_name: &str
) -> PathBuf {
    let kernel_src = kernel_src_dir.join(format!("{}.metal", kernel_name));
    let kernel_air = kernel_build_dir.join(format!("{}.air", kernel_name));
    // let kernel_metallib = kernel_build_dir.join(format!("{}.metallib", kernel_name));

    Command::new("xcrun")
        .arg("-sdk").arg("macosx").arg("metal")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-fno-fast-math")
        .arg("-c").arg(kernel_src)
        .arg(format!("-I{}", mlx_dir.as_ref().display()))
        .arg("-o").arg(&kernel_air)
        .status()
        .unwrap();

    kernel_air
}

fn build_kernel_metallib(
    kernel_airs: &[PathBuf],
    out_dir: &PathBuf,
) {
    // Note that the built ``mlx.metallib`` file should be either at the same
    // directory as the executable statically linked to ``libmlx.a`` or the
    // preprocessor constant ``METAL_PATH`` should be defined at build time and it
    // should point to the path to the built metal library.
    let kernel_metallib = out_dir.join("mlx.metallib");
    Command::new("xcrun")
        .arg("-sdk").arg("macosx").arg("metallib")
        .args(kernel_airs)
        .arg("-o").arg(&kernel_metallib)
        .status()
        .unwrap();
}