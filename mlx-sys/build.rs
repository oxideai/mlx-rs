use std::{path::PathBuf, process::Command};

const MLX_DIR: &str = "mlx";
#[cfg(feature = "metal")]
const METAL_CPP_MACOS_14_2_DIR: &str = "metal-cpp_macOS14.2_iOS17.2";
#[cfg(feature = "metal")]
const METAL_CPP_MACOS_14_0_DIR: &str = "metal-cpp_macOS14_iOS17-beta";

const FILES_MLX: &[&str] = &[
    "mlx/mlx/allocator.cpp",
    "mlx/mlx/array.cpp",
    "mlx/mlx/compile.cpp",
    "mlx/mlx/device.cpp",
    "mlx/mlx/dtype.cpp",
    "mlx/mlx/fast.cpp",
    "mlx/mlx/fft.cpp",
    "mlx/mlx/ops.cpp",
    "mlx/mlx/graph_utils.cpp",
    "mlx/mlx/primitives.cpp",
    "mlx/mlx/random.cpp",
    "mlx/mlx/scheduler.cpp",
    "mlx/mlx/transforms.cpp",
    "mlx/mlx/utils.cpp",
    "mlx/mlx/linalg.cpp",
];

/// Common files to compile for all backends
const FILES_MLX_BACKEND_COMMON: &[&str] = &[
    "mlx/mlx/backend/common/arg_reduce.cpp",
    "mlx/mlx/backend/common/binary.cpp",
    "mlx/mlx/backend/common/compiled.cpp",
    "mlx/mlx/backend/common/conv.cpp",
    "mlx/mlx/backend/common/copy.cpp",
    "mlx/mlx/backend/common/erf.cpp",
    "mlx/mlx/backend/common/fft.cpp",
    "mlx/mlx/backend/common/primitives.cpp",
    "mlx/mlx/backend/common/quantized.cpp",
    "mlx/mlx/backend/common/reduce.cpp",
    "mlx/mlx/backend/common/scan.cpp",
    "mlx/mlx/backend/common/select.cpp",
    "mlx/mlx/backend/common/softmax.cpp",
    "mlx/mlx/backend/common/sort.cpp",
    "mlx/mlx/backend/common/threefry.cpp",
    "mlx/mlx/backend/common/indexing.cpp",
    "mlx/mlx/backend/common/load.cpp",
    "mlx/mlx/backend/common/qrf.cpp",
    "mlx/mlx/backend/common/svd.cpp",
    "mlx/mlx/backend/common/inverse.cpp",
];

#[cfg(target_os = "ios")]
const FILES_MLX_BACKEND_COMPILED: &str = "mlx/mlx/backend/common/compiled_nocpu.cpp";

#[cfg(not(target_os = "ios"))]
const FILES_MLX_BACKEND_COMPILED: &str = "mlx/mlx/backend/common/compiled_cpu.cpp";

#[cfg(not(feature = "accelerate"))]
const FILE_MLX_BACKEND_COMMON_DEFAULT_PRIMITIVES: &str =
    "mlx/mlx/backend/common/default_primitives.cpp";

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
    "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/metal/metal.cpp",
    "mlx/mlx/backend/metal/primitives.cpp",
    "mlx/mlx/backend/metal/quantized.cpp",
    "mlx/mlx/backend/metal/normalization.cpp",
    "mlx/mlx/backend/metal/rope.cpp",
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
    "binary_two",
    "conv",
    "copy",
    "gemv",
    "quantized",
    "random",
    "rms_norm",
    "layer_norm",
    "rope",
    "scan",
    "scaled_dot_product_attention",
    "softmax",
    "sort",
    "ternary",
    "unary",
    "gather",
    "scatter",
];

const METAL_STEEL_KERNEL_DIR: &str = "mlx/mlx/backend/metal/kernels/steel";
const METAL_REDUCTION_KERNEL_DIR: &str = "mlx/mlx/backend/metal/kernels/reduction";

const SHIM_DIR: &str = "shim";

const FILES_SHIM_MLX: &[&str] = &[
    "src/array.cpp",
    "src/types.cpp",
    "src/dtype.cpp",
    "src/fft.cpp",
    "src/mlx_cxx.cpp",
    "src/utils.cpp",
    "src/linalg.cpp",
    "src/ops.cpp",
    "src/transforms.cpp",
    "src/random.cpp",
    "src/compile.cpp",
    "src/compat.cpp",
    "src/fast.cpp",
    "src/backend/metal.cpp",
];

const RUST_SOURCE_FILES: &[&str] = &[
    "src/lib.rs",
    "src/array.rs",
    "src/dtype.rs",
    "src/types/float16.rs",
    "src/types/bfloat16.rs",
    "src/types/complex64.rs",
    "src/backend/metal.rs",
    "src/device.rs",
    "src/stream.rs",
    "src/fft.rs",
    "src/utils.rs",
    "src/linalg.rs",
    "src/ops.rs",
    "src/transforms.rs",
    "src/compat.rs",
    "src/random.rs",
    "src/compile.rs",
    "src/fast.rs",
];

const MAKE_COMPILED_PREAMBLE_SH: &str = "mlx/mlx/backend/common/make_compiled_preamble.sh";

fn main() {
    // TODO: conditionally compile based on if accelerate is available
    // TODO: how to use MLX_METAL_DEBUG flag?

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    panic!("Building for x86_64 on macOS is not supported.");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:warning=out_dir: {}", out_dir.display());

    // let mut build = cxx_build::bridge("src/lib.rs");
    let mut build = cxx_build::bridges(RUST_SOURCE_FILES);

    build
        .include(MLX_DIR)
        .include(SHIM_DIR)
        .flag("-std=c++17")
        .files(FILES_MLX)
        .files(FILES_MLX_BACKEND_COMMON)
        .file(FILES_MLX_BACKEND_COMPILED)
        .files(FILES_SHIM_MLX);

    let compiled_preamble_cpp = cpu_compiled_preamble(&out_dir);
    build.file(compiled_preamble_cpp);

    // TODO: check if accelerate is available
    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        build
            .include("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers")
            // mlx uses new lapack api if accelerate is available
            .flag("-D")
            .flag("ACCELERATE_NEW_LAPACK")
            .files(FILES_MLX_BACKEND_ACCELERATE);
    }

    #[cfg(not(feature = "accelerate"))]
    {
        // Find the system's BLAS and LAPACK
        let blas_include_dirs = find_non_accelerate_blas().expect("Must have BLAS installed");
        build.include(blas_include_dirs);
        println!("cargo:rustc-link-lib=cblas");

        let lapack_include_dirs = find_non_accelerate_lapack().expect("Must have LAPACK installed");
        build.include(lapack_include_dirs);
        println!("cargo:rustc-link-lib=lapacke");

        build.file(FILE_MLX_BACKEND_COMMON_DEFAULT_PRIMITIVES);
    }

    #[cfg(feature = "metal")]
    {
        build.flag("-D").flag("_METAL_");

        let macos_version = get_macos_version();
        if macos_version >= 14.2 {
            build.include(METAL_CPP_MACOS_14_2_DIR);
        } else if macos_version >= 14.0 {
            build.include(METAL_CPP_MACOS_14_0_DIR);
        } else {
            panic!("MLX requires macOS >= 14.0 to be built with MLX_BUILD_METAL=ON");
        }

        let metallib = build_metal_kernels(&out_dir);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        build
            .files(FILES_MLX_BACKEND_METAL)
            .flag("-D")
            .flag(&format!("METAL_PATH=\"{}\"", metallib.display()));
    }
    #[cfg(not(feature = "metal"))]
    {
        build.files(FILES_MLX_BACKEND_NO_METAL);
    }

    build.compile("mlx");

    println!("cargo:rerun-if-changed=shim/mlx-cxx/array.hpp");
    println!("cargo:rerun-if-changed=shim/mlx-cxx/array.cpp");
    println!("cargo:rustc-link-lib=mlx");
}

/// Equivalent to the cpu_compiled_preamble in backend/common/CMakeLists.txt
fn cpu_compiled_preamble(out_dir: &PathBuf) -> PathBuf {
    generate_compiled_preamble_cpp(out_dir)
}

fn get_cxx_compiler() -> String {
    let output = Command::new("xcrun")
        .arg("-find")
        .arg("c++")
        .output().unwrap();

    assert!(output.status.success(), "Failed to find c++ compiler");

    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn generate_compiled_preamble_cpp(out_dir: &PathBuf) -> PathBuf {
    let source_dir = PathBuf::from(MLX_DIR);
    let source_dir = std::fs::canonicalize(source_dir).unwrap();
    let preamble_cpp = out_dir.join("compiled_preamble.cpp");
    let cxx_compiler = get_cxx_compiler();
    let status = Command::new("/bin/bash")
        .arg(MAKE_COMPILED_PREAMBLE_SH)
        .arg(&preamble_cpp)
        .arg(&cxx_compiler)
        .arg(&source_dir)
        .arg("TRUE")
        .status()
        .unwrap();

    assert!(status.success(), "Failed to generate compiled preamble");
    preamble_cpp
}

// TODO: recursively search for .metal files in steels directory and reduction directory
// and build them
#[cfg(feature = "metal")]
fn build_metal_kernels(out_dir: &PathBuf) -> PathBuf {
    let steel_kernels = recursive_search_metal_kernels(METAL_STEEL_KERNEL_DIR);
    let reduction_kernels = recursive_search_metal_kernels(METAL_REDUCTION_KERNEL_DIR);

    let kernel_build_dir = PathBuf::from(out_dir).join("kernels"); // MLX_METAL_PATH
    let _ = std::fs::create_dir_all(&kernel_build_dir);
    let kernel_src_dir = PathBuf::from(METAL_KERNEL_DIR);
    let mut kernel_airs = METAL_KERNELS
        .iter()
        .map(|&k| build_kernel_air(&kernel_build_dir, &kernel_src_dir, MLX_DIR, k))
        .collect::<Vec<_>>();

    for (kernel_src_dir, kernel_name) in steel_kernels {
        let kernel_air = build_kernel_air(&kernel_build_dir, &kernel_src_dir, MLX_DIR, &kernel_name);
        kernel_airs.push(kernel_air);
    }

    for (kernel_src_dir, kernel_name) in reduction_kernels {
        let kernel_air = build_kernel_air(&kernel_build_dir, &kernel_src_dir, MLX_DIR, &kernel_name);
        kernel_airs.push(kernel_air);
    }

    build_kernel_metallib(&kernel_airs, out_dir)
}

#[cfg(feature = "metal")]
fn recursive_search_metal_kernels(root: impl AsRef<std::path::Path>) -> Vec<(PathBuf, String)> {
    use walkdir::WalkDir;

    let mut kernels = Vec::new();
    for entry in WalkDir::new(root) 
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let filename = entry.file_name().to_string_lossy();
        if filename.ends_with(".metal") {
            let dir = entry.path().parent().unwrap().to_path_buf();
            let kernel_name = filename.trim_end_matches(".metal").to_string();
            kernels.push((dir, kernel_name));
        }
    }

    kernels
}

#[cfg(feature = "metal")]
fn build_kernel_air(
    kernel_build_dir: &PathBuf,
    kernel_src_dir: &PathBuf,
    mlx_dir: impl AsRef<std::path::Path>,
    kernel_name: &str,
) -> PathBuf {
    let kernel_src = kernel_src_dir.join(format!("{}.metal", kernel_name));
    let kernel_air = kernel_build_dir.join(format!("{}.air", kernel_name));
    // let kernel_metallib = kernel_build_dir.join(format!("{}.metallib", kernel_name));

    // TODO: what to do with MLX_METAL_DEBUG flag?

    let status = std::process::Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-fno-fast-math")
        .arg("-c")
        .arg(kernel_src)
        .arg(format!("-I{}", mlx_dir.as_ref().display()))
        .arg("-o")
        .arg(&kernel_air)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "Failed to build kernel air: {}",
        kernel_name
    );

    kernel_air
}

#[cfg(feature = "metal")]
fn build_kernel_metallib(kernel_airs: &[PathBuf], out_dir: &PathBuf) -> PathBuf {
    // Note that the built ``mlx.metallib`` file should be either at the same
    // directory as the executable statically linked to ``libmlx.a`` or the
    // preprocessor constant ``METAL_PATH`` should be defined at build time and it
    // should point to the path to the built metal library.
    let kernel_metallib = out_dir.join("mlx.metallib");
    println!(
        "cargo:warning=kernel_metallib: {}",
        kernel_metallib.display()
    );
    let status = std::process::Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("metallib")
        .args(kernel_airs)
        .arg("-o")
        .arg(&kernel_metallib)
        .status()
        .unwrap();
    assert!(status.success(), "Failed to build kernel metallib");
    kernel_metallib
}

#[cfg(feature = "metal")]
fn get_macos_version() -> f32 {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("--show-sdk-version")
        .output()
        .expect("Failed to get macos version");
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[cfg(not(feature = "accelerate"))]
const NON_ACCELERATE_BLAS_HEADER: &str = "cblas.h";

#[cfg(not(feature = "accelerate"))]
const NON_ACCELERATE_LAPACK_HEADER: &str = "lapacke.h";

#[cfg(not(feature = "accelerate"))]
const NON_ACCELERATE_BLAS_LAPACK_SEARCH_DIRS: &[&str] = &["/usr/include", "/usr/local/include"];

#[cfg(not(feature = "accelerate"))]
fn find_non_accelerate_blas() -> Option<PathBuf> {
    let blas_home = std::env::var("BLAS_HOME").ok().map(PathBuf::from);

    let mut blas_dirs = NON_ACCELERATE_BLAS_LAPACK_SEARCH_DIRS
        .iter()
        .map(|&dir| PathBuf::from(dir))
        .collect::<Vec<_>>();

    if let Some(blas_home) = blas_home {
        blas_dirs.push(blas_home.join("include"));
    }

    for dir in blas_dirs {
        let blas_header = dir.join(NON_ACCELERATE_BLAS_HEADER);
        if blas_header.exists() {
            return Some(dir);
        }
    }

    None
}

#[cfg(not(feature = "accelerate"))]
fn find_non_accelerate_lapack() -> Option<PathBuf> {
    for dir in NON_ACCELERATE_BLAS_LAPACK_SEARCH_DIRS {
        let lapack_header = PathBuf::from(dir).join(NON_ACCELERATE_LAPACK_HEADER);
        if lapack_header.exists() {
            return Some(PathBuf::from(dir));
        }
    }

    None
}
