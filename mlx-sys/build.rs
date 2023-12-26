fn main() -> miette::Result<()> {
    let dst = cmake::Config::new("mlx").build();
    println!("cargo:warning={}", dst.display());

    println!("cargo:rustc-link-search={}/lib", dst.display());
    println!("cargo:rustc-link-lib=mlx");

    Ok(())
}
