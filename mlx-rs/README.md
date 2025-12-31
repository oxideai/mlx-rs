# mlx-rs-burn

[![Crates.io](https://img.shields.io/crates/v/mlx-rs-burn.svg)](https://crates.io/crates/mlx-rs-burn)
[![Documentation](https://docs.rs/mlx-rs-burn/badge.svg)](https://docs.rs/mlx-rs-burn)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Rust bindings for Apple's MLX framework** — a fork of [mlx-rs](https://github.com/oxideai/mlx-rs) with additional operations for [burn-mlx](https://crates.io/crates/burn-mlx).

## Why This Fork?

This crate extends the original mlx-rs with operations required by the burn-mlx deep learning backend:

- **`slice`** - Array slicing with start/stop indices
- **`slice_update`** - In-place slice updates
- **`scatter` / `scatter_add`** - Scatter operations for gradient computation
- **`flip`** - Array flipping along axes

These operations are essential for implementing neural network backward passes (e.g., pooling gradients).

## Installation

```toml
[dependencies]
mlx-rs-burn = "0.25.4"
```

## Quick Start

```rust
use mlx_rs::Array;
use mlx_rs::ops::{slice, scatter_add, flip};

// Create an array
let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

// Slice operations
let sliced = slice(&x, &[1], &[3], None).unwrap();
// Result: [2.0, 3.0]

// Scatter add (useful for gradient accumulation)
let zeros = Array::zeros::<f32>(&[4]);
let indices = Array::from_slice(&[0i32, 2], &[2]);
let updates = Array::from_slice(&[10.0f32, 30.0], &[2]);
let result = scatter_add(&zeros, &[&indices], &updates, &[0]).unwrap();
// Result: [10.0, 0.0, 30.0, 0.0]

// Flip along axis
let flipped = flip(&x, &[0]).unwrap();
// Result: [4.0, 3.0, 2.0, 1.0]
```

## Features

All features from the original mlx-rs are available:

- **Performance**: Optimized for Apple Silicon (M1/M2/M3/M4)
- **Lazy Evaluation**: Arrays are only materialized when needed
- **Dynamic Graphs**: Computation graphs constructed dynamically
- **Unified Memory**: Zero-copy data sharing between CPU and GPU
- **Metal GPU**: Native GPU acceleration via Metal

### Feature Flags

- `metal` (default) - Enable Metal GPU support
- `accelerate` (default) - Enable Accelerate framework
- `safetensors` - Enable safetensors file format support

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.82+

## Related Crates

| Crate | Description |
|-------|-------------|
| [burn-mlx](https://crates.io/crates/burn-mlx) | MLX backend for the Burn deep learning framework |
| [mlx-sys-burn](https://crates.io/crates/mlx-sys-burn) | Low-level FFI bindings |
| [mlx-macros-burn](https://crates.io/crates/mlx-macros-burn) | Procedural macros |
| [mlx-internal-macros-burn](https://crates.io/crates/mlx-internal-macros-burn) | Internal macros |

## Upstream

This is a fork of [oxideai/mlx-rs](https://github.com/oxideai/mlx-rs). We aim to contribute these additions back upstream. In the meantime, this fork is published to enable burn-mlx on crates.io.

## License

MIT OR Apache-2.0

## Acknowledgments

- [mlx-rs](https://github.com/oxideai/mlx-rs) - Original Rust bindings by OxideAI
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [Burn](https://github.com/tracel-ai/burn) - Rust deep learning framework
