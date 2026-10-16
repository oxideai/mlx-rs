# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

The crate version is independent of its native source tuple. This revision targets mlx-c
`c74db5307cc8ce122f48d97ef951b30578674e7f`, whose CMake configuration pins MLX `v0.32.2`.

## Metal library location

Metal builds place `mlx.metallib` in `~/.mlx/lib/<mlx-c-key>/`, where `<mlx-c-key>` is the
first 12 characters of the pinned mlx-c commit. Packaged source without Git metadata uses a
deterministic hash of the mlx-c headers and CMake configuration instead. The stable location
allows binaries produced by `cargo install` to keep loading the library after Cargo removes its
temporary build directory.

Set `MLX_RS_METAL_PATH` to use a different directory verbatim as CMake's `MLX_METAL_PATH`.
When this override is set, the build does not read or write `HOME`, which supports sandboxed and
Nix builds.

## Build metadata for dependents

`mlx-sys` declares `links = "mlx"`, so Cargo runs a dependent's build script after this one and
passes it the locations of the native sources this build compiled:

| Variable | Value |
|---|---|
| `DEP_MLX_INCLUDE` | the MLX source tree; include paths are `mlx/...` |
| `DEP_MLX_METAL_CPP` | the metal-cpp source tree |
| `DEP_MLX_MLX_C` | the mlx-c source tree |
| `DEP_MLX_DEFINES` | preprocessor definitions MLX was built with (`MLX_STATIC`) |
| `DEP_MLX_VERSION` | the MLX version mlx-c pins, when it can be read |

C++ that extends MLX, such as a custom `mlx::core::Primitive`, must compile against the same
headers `libmlx` was built from. A dependent can use these with the `cc` crate:

```rust
let mut build = cc::Build::new();
build.cpp(true).std("c++20");
for key in ["DEP_MLX_INCLUDE", "DEP_MLX_METAL_CPP", "DEP_MLX_MLX_C"] {
    build.include(std::env::var(key).unwrap());
}
for define in std::env::var("DEP_MLX_DEFINES").unwrap().split_whitespace() {
    build.define(define, None);
}
```
