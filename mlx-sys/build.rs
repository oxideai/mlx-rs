fn main() {
    cxx_build::bridge("src/main.rs")
        // .file("mlx/mlx/random.cpp")
        .include("mlx")
        .flag_if_supported("-std=c++17")
        .compile("mlx");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rustc-link-lib=dylib=mlx");
}