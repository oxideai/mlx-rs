#![deny(missing_docs, missing_debug_implementations)]

//! Neural network support for MLX
//!
//! All modules provide a `new()` function that take mandatory parameters and other methods
//! to set optional parameters.

pub mod error;
pub mod losses;
pub mod macros;
pub mod utils;

mod activation;
mod container;
mod convolution;
mod convolution_transpose;
mod dropout;
mod embedding;
mod linear;
mod pooling;
mod normalization;
mod recurrent;
mod transformer;
mod upsample;
mod value_and_grad;

pub use activation::*;
pub use container::*;
pub use convolution::*;
pub use convolution_transpose::*;
pub use dropout::*;
pub use embedding::*;
pub use linear::*;
pub use pooling::*;
pub use normalization::*;
pub use recurrent::*;
pub use transformer::*;
pub use upsample::*;
pub use value_and_grad::*;

/// Re-export of the `mlx-nn-module` crate.
pub mod module {
    pub use mlx_rs::module::*;
}
