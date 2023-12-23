fn main() -> miette::Result<()> {
    // Directory containing the MLX source code
    let mlx_source_path = std::path::PathBuf::from("mlx/mlx");
    // Directory containing the MLX bindings where we may need to add custom C++ code
    // to make the opague types more useful
    let mlx_sys_path = std::path::PathBuf::from("src");

    let mut builder =
        autocxx_build::Builder::new("src/lib.rs", &[&mlx_source_path, &mlx_sys_path]).build()?;
    builder.flag_if_supported("-std=c++17").compile("mlx-sys");
    println!("cargo:rerun-if-changed=src/lib.rs");
    Ok(())
}
