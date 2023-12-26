use std::path::Path;

fn main() -> miette::Result<()> {
    let dst = cmake::Config::new("mlx").build();
    println!("cargo:warning={}", dst.display());

    cxx_build::bridge("src/main.rs")
        .flag_if_supported("-std=c++17")
        .include(Path::new(&format!("{}/include", dst.display())))
        .compile("mlx");

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");

    Ok(())
}
