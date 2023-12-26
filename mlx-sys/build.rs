fn main() -> miette::Result<()> {
    // It's necessary to use an absolute path here because the
    // C++ codegen and the macro codegen appears to be run from different
    // working directories.
    let path = std::path::PathBuf::from("mlx");
    let path2 = std::path::PathBuf::from("src");
    let mut b = autocxx_build::Builder::new("src/main.rs", &[&path, &path2])
        .extra_clang_args(&["-std=c++17"])
        .build()?;
    b
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .compile("mlx-binding");
    println!("cargo:rerun-if-changed=src/main.rs");
    Ok(())
}
