fn main() -> miette::Result<()> {
    // Directory containing the MLX source code
    let mlx_source_path = std::path::PathBuf::from("mlx");
    // Directory containing the MLX bindings where we may need to add custom C++ code
    // to make the opague types more useful
    let mlx_sys_path = std::path::PathBuf::from("src");

    let mut builder =
        autocxx_build::Builder::new("src/main.rs", &[&mlx_source_path, &mlx_sys_path])
        .extra_clang_args(&["-xc++", "-std=c++17"])
        .build()?;
    builder
        .cpp(true)
        .std("c++17")
        .compile("mlx");
    // println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=main.rs");
    Ok(())
}
