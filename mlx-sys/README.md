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

The `metal-jit` feature builds MLX with `MLX_METAL_JIT`: the metallib shrinks from about 136 MB
to 1.5 MB at this pin, and most kernels compile from source the first time they are used. macOS
keeps the compiled shaders across runs, so the cost lands mostly on the first run: the `mlx-rs`
lib tests took 27 s instead of 5 s the first time, and the same as a normal build afterwards.
Results can differ from the precompiled kernels in the last bits. JIT builds use
`~/.mlx/lib/<mlx-c-key>-jit/`; with `MLX_RS_METAL_PATH`, give JIT and non-JIT builds different
directories.
