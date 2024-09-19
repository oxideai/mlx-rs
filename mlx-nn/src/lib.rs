#![deny(missing_docs, missing_debug_implementations)]

//! Neural network support for MLX
//!
//! All modules provide a `new()` function that take mandatory parameters and other methods
//! to set optional parameters.

pub mod error;
pub mod macros;
pub mod optimizer;
pub mod utils;

mod activation;
mod container;
mod convolution;
mod dropout;
mod linear;
mod value_and_grad;

pub use activation::*;
pub use container::*;
pub use convolution::*;
pub use dropout::*;
pub use linear::*;
pub use value_and_grad::*;

/// Re-export of the `mlx-nn-module` crate.
pub mod module {
    pub use mlx_nn_module::*;
}
