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

const ENV_ACCELERATE_NEW_LAPACK: &str = "ACCELERATE_NEW_LAPACK";

/// Files to compile for accelerate backend
const FILES_MLX_BACKEND_ACCELERATE: &[&str] = &[
    "mlx/mlx/backend/accelerate/conv.cpp",
    "mlx/mlx/backend/accelerate/matmul.cpp",
    "mlx/mlx/backend/accelerate/primitives.cpp",
    "mlx/mlx/backend/accelerate/quantized.cpp",
    "mlx/mlx/backend/accelerate/reduce.cpp",
    "mlx/mlx/backend/accelerate/softmax.cpp",
];

fn main() {
    // TODO: conditionally compile based on if accelerate is available

    let mut build = cxx_build::bridge("src/main.rs");
        
    build.include("mlx")
        .include("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers")
        .flag("-std=c++17")
        .flag("-framework").flag("vecLib")
        .flag("-framework").flag("Accelerate")
        .files(FILES_MLX)
        .files(FILES_MLX_BACKEND_COMMON);


    // #[cfg(feature = "accelerate")] // TODO: check if accelerate works with the original cmake build
    // {
    //     std::env::set_var(ENV_ACCELERATE_NEW_LAPACK, "1");
    //     build.files(FILES_MLX_BACKEND_ACCELERATE); // TODO: this may require setting ACCELERATE_NEW_LAPACK
    // }
        
    build.compile("mlx");

    println!("cargo:rerun-if-changed=src/main.rs");
    // println!("cargo:rustc-link-lib=dylib=mlx");
}