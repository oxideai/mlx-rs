use std::path::PathBuf;

const MLX_DIR: &str = "mlx";
#[cfg(feature = "metal")]
const METAL_CPP_MACOS_14_2_DIR: &str = "metal-cpp_macOS14.2_iOS17.2";
#[cfg(feature = "metal")]
const METAL_CPP_MACOS_14_0_DIR: &str = "metal-cpp_macOS14_iOS17-beta";
#[cfg(feature = "metal")]
const METAL_CPP_MACOS_13_3_DIR: &str = "metal-cpp_macOS13.3_iOS16.4";

const FILES_MLX: &[&str] = &[
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
#[cfg(feature = "accelerate")]
const FILES_MLX_BACKEND_ACCELERATE: &[&str] = &[
    "mlx/mlx/backend/accelerate/conv.cpp",
    "mlx/mlx/backend/accelerate/matmul.cpp",
    "mlx/mlx/backend/accelerate/primitives.cpp",
    "mlx/mlx/backend/accelerate/quantized.cpp",
    "mlx/mlx/backend/accelerate/reduce.cpp",
    "mlx/mlx/backend/accelerate/softmax.cpp",
];

#[cfg(feature = "metal")]
const FILES_MLX_BACKEND_METAL: &[&str] = &[
    "mlx/mlx/backend/metal/allocator.cpp",
    "mlx/mlx/backend/metal/conv.cpp",
    "mlx/mlx/backend/metal/copy.cpp",
    "mlx/mlx/backend/metal/device.cpp",
    "mlx/mlx/backend/metal/fft.cpp",
    "mlx/mlx/backend/metal/indexing.cpp",
    "mlx/mlx/backend/metal/matmul.cpp",
    "mlx/mlx/backend/metal/metal.cpp",
    "mlx/mlx/backend/metal/primitives.cpp",
    "mlx/mlx/backend/metal/quantized.cpp",
    "mlx/mlx/backend/metal/scan.cpp",
    "mlx/mlx/backend/metal/softmax.cpp",
    "mlx/mlx/backend/metal/sort.cpp",
    "mlx/mlx/backend/metal/reduce.cpp",
];

#[cfg(not(feature = "metal"))]
const FILES_MLX_BACKEND_NO_METAL: &[&str] = &[
    "mlx/mlx/backend/no_metal/allocator.cpp",
    "mlx/mlx/backend/no_metal/metal.cpp",
    "mlx/mlx/backend/no_metal/primitives.cpp",
];

#[cfg(feature = "metal")]
const METAL_KERNEL_DIR: &str = "mlx/mlx/backend/metal/kernels";
// const KERNEL_HEADERS: &[&str] = &[
//     "bf16.h",
//     "bf16_math.h",
//     "complex.h",
//     "defines.h",
//     "erf.h",
//     "reduce.h",
//     "utils.h",
// ];
#[cfg(feature = "metal")]
const METAL_KERNELS: &[&str] = &[
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

const WRAPPER_DIR: &str = "wrapper";

const FILES_WRAPPER_MLX: &[&str] = &[
    "wrapper/mlx-cxx/array.cpp",
];

fn main() {
    // TODO: conditionally compile based on if accelerate is available

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:warning=out_dir: {}", out_dir.display());
    

    let mut build = cxx_build::bridge("src/lib.rs");
        
    build.include(MLX_DIR)
        .include(WRAPPER_DIR)
        .include("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers")
        .flag("-std=c++17")
        .files(FILES_MLX)
        .files(FILES_MLX_BACKEND_COMMON)
        .files(FILES_WRAPPER_MLX);

    // TODO: check if accelerate is available
    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        build
            // mlx uses new lapack api if accelerate is available
            .flag("-DACCELERATE_NEW_LAPACK") 
            .files(FILES_MLX_BACKEND_ACCELERATE);
    }

    #[cfg(feature = "metal")]
    {
        let macos_version = get_macos_version();
        if macos_version >= 14.2 {
            build.include(METAL_CPP_MACOS_14_2_DIR);
        } else if macos_version >= 14.0 {
            build.include(METAL_CPP_MACOS_14_0_DIR);
        } else if macos_version >= 13.3 {
            build.include(METAL_CPP_MACOS_13_3_DIR);
        } else {
            panic!("MLX requires macOS >= 13.4 to be built with MLX_BUILD_METAL=ON");
        }

        let metallib = build_metal_kernels(&out_dir);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        build
            .files(FILES_MLX_BACKEND_METAL)
            .flag("-D").flag(&format!("METAL_PATH=\"{}\"", metallib.display()));
    }
    #[cfg(not(feature = "metal"))]
    {
        build
            .files(FILES_MLX_BACKEND_NO_METAL);
    }
        
    build.compile("mlx");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rustc-link-lib=mlx");
}

#[cfg(feature = "metal")]
fn build_metal_kernels(out_dir: &PathBuf) -> PathBuf {
    // test build kernel air
    let kernel_build_dir = PathBuf::from(out_dir).join("kernels"); // MLX_METAL_PATH
    let _ = std::fs::create_dir_all(&kernel_build_dir);
    let kernel_src_dir = PathBuf::from(METAL_KERNEL_DIR);
    let kernel_airs = METAL_KERNELS.iter().map(|&k| {
        build_kernel_air(
            &kernel_build_dir, 
            &kernel_src_dir,
            MLX_DIR,
            k
        )
    }).collect::<Vec<_>>();
    build_kernel_metallib(&kernel_airs, out_dir)
}

#[cfg(feature = "metal")]
fn build_kernel_air(
    kernel_build_dir: &PathBuf, 
    kernel_src_dir: &PathBuf,
    mlx_dir: impl AsRef<std::path::Path>,
    kernel_name: &str
) -> PathBuf {
    let kernel_src = kernel_src_dir.join(format!("{}.metal", kernel_name));
    let kernel_air = kernel_build_dir.join(format!("{}.air", kernel_name));
    // let kernel_metallib = kernel_build_dir.join(format!("{}.metallib", kernel_name));

    std::process::Command::new("xcrun")
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

#[cfg(feature = "metal")]
fn build_kernel_metallib(
    kernel_airs: &[PathBuf],
    out_dir: &PathBuf,
) -> PathBuf {
    // Note that the built ``mlx.metallib`` file should be either at the same
    // directory as the executable statically linked to ``libmlx.a`` or the
    // preprocessor constant ``METAL_PATH`` should be defined at build time and it
    // should point to the path to the built metal library.
    let kernel_metallib = out_dir.join("mlx.metallib");
    std::process::Command::new("xcrun")
        .arg("-sdk").arg("macosx").arg("metallib")
        .args(kernel_airs)
        .arg("-o").arg(&kernel_metallib)
        .status()
        .unwrap();
    kernel_metallib
}

#[cfg(feature = "metal")]
fn get_macos_version() -> f32 {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg("-sdk").arg("macosx")
        .arg("--show-sdk-version")
        .output()
        .expect("Failed to get macos version");
    String::from_utf8(output.stdout).unwrap().trim()
        .parse().unwrap()
}