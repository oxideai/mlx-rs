extern crate cmake;
use cmake::Config;

fn main() {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")] {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")] {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")] {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")] {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}