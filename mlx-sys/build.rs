extern crate cmake;

use cmake::Config;
use std::{env, fs, path::PathBuf, process::Command};

fn use_prebuilt_mlx() -> bool {
    // Check if MLX_PREBUILT_PATH is set
    env::var("MLX_PREBUILT_PATH").is_ok()
}

/// Patch the MLX source files to work around macOS Tahoe beta issues
/// This is needed because Metal 4.0 and __builtin_available for macOS 26 are not fully supported
fn patch_metal_version(out_dir: &PathBuf) {
    // Patch device.cpp to force Metal 3.2
    let device_cpp = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.cpp");
    if device_cpp.exists() {
        if let Ok(content) = fs::read_to_string(&device_cpp) {
            if !content.contains("// PATCHED: Force Metal 3.2") {
                let old_code = r#"auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                let new_code = r#"// PATCHED: Force Metal 3.2 to work around Xcode beta Metal 4.0 issues
auto get_metal_version() {
  auto get_metal_version_ = []() {
    // Force Metal 3.2 - Metal 4.0 not supported in current Xcode beta
    if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    if fs::write(&device_cpp, patched).is_ok() {
                        println!("cargo:warning=Patched MLX device.cpp to force Metal 3.2");
                    }
                }
            }
        }
    }

    // Patch device.h to disable NAX (uses __builtin_available for macOS 26.2)
    let device_h = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.h");
    if device_h.exists() {
        if let Ok(content) = fs::read_to_string(&device_h) {
            if !content.contains("// PATCHED: Disable NAX") {
                let old_code = r#"inline bool is_nax_available() {
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    can_use_nax &=
        metal::device(mlx::core::Device::gpu).get_architecture_gen() >= 17;
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
}"#;

                let new_code = r#"// PATCHED: Disable NAX - __builtin_available for macOS 26.2 causes link errors
inline bool is_nax_available() {
  // NAX is not available on current Xcode beta
  return false;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    if fs::write(&device_h, patched).is_ok() {
                        println!("cargo:warning=Patched MLX device.h to disable NAX");
                    }
                }
            }
        }
    }
}

fn build_and_link_mlx_c() {
    if use_prebuilt_mlx() {
        // Use pre-built MLX library
        let mlx_path = env::var("MLX_PREBUILT_PATH").unwrap();
        println!("cargo:warning=Using pre-built MLX from: {}", mlx_path);
        println!("cargo:rustc-link-search=native={}", mlx_path);
        println!("cargo:rustc-link-lib=dylib=mlx");
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let build_dir = out_dir.join("build");

        // Create build directory
        fs::create_dir_all(&build_dir).ok();

        // Run CMake configure to fetch dependencies
        let src_dir = PathBuf::from("src/mlx-c").canonicalize().unwrap();

        let mut cmake_args = vec![
            format!("-S{}", src_dir.display()),
            format!("-B{}", build_dir.display()),
            "-DCMAKE_INSTALL_PREFIX=.".to_string(),
        ];

        #[cfg(debug_assertions)]
        cmake_args.push("-DCMAKE_BUILD_TYPE=Debug".to_string());

        #[cfg(not(debug_assertions))]
        cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string());

        #[cfg(feature = "metal")]
        cmake_args.push("-DMLX_BUILD_METAL=ON".to_string());

        #[cfg(not(feature = "metal"))]
        cmake_args.push("-DMLX_BUILD_METAL=OFF".to_string());

        #[cfg(feature = "accelerate")]
        cmake_args.push("-DMLX_BUILD_ACCELERATE=ON".to_string());

        #[cfg(not(feature = "accelerate"))]
        cmake_args.push("-DMLX_BUILD_ACCELERATE=OFF".to_string());

        cmake_args.push("-DCMAKE_METAL_COMPILER=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal".to_string());

        // MLX v0.30.1 requires macOS 15.0 for __builtin_available checks
        cmake_args.push("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0".to_string());

        // Set environment for cmake to use
        std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "15.0");

        // Run cmake configure
        let status = Command::new("cmake")
            .args(&cmake_args)
            .status()
            .expect("Failed to run cmake configure");

        if !status.success() {
            panic!("CMake configure failed");
        }

        // Apply Metal version patch after CMake fetches the sources
        patch_metal_version(&out_dir);

        // Run cmake build
        let status = Command::new("cmake")
            .args(["--build", &build_dir.to_string_lossy(), "--config", "Release", "-j"])
            .status()
            .expect("Failed to run cmake build");

        if !status.success() {
            panic!("CMake build failed");
        }

        // Link the libraries from the correct paths
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-search=native={}/_deps/mlx-build", build_dir.display());
        println!("cargo:rustc-link-lib=static=mlx");
        println!("cargo:rustc-link-lib=static=mlxc");
    }

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

fn main() {
    // Set macOS deployment target early for consistent linking
    std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "15.0");

    // Add linker flags for macOS minimum version (for __builtin_available)
    println!("cargo:rustc-link-arg=-mmacosx-version-min=15.0");

    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
