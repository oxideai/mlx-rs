# mlx-sys-burn

[![Crates.io](https://img.shields.io/crates/v/mlx-sys-burn.svg)](https://crates.io/crates/mlx-sys-burn)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Low-level Rust FFI bindings to the mlx-c API, generated using bindgen.

This is a fork of mlx-sys with additional bindings required by [burn-mlx](https://crates.io/crates/burn-mlx).

## Installation

```toml
[dependencies]
mlx-sys-burn = "0.2.1"
```

## Note

This crate provides raw FFI bindings. For a safe, idiomatic Rust interface, use [mlx-rs-burn](https://crates.io/crates/mlx-rs-burn) instead.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.82+

## License

MIT OR Apache-2.0
