#![deny(missing_docs, missing_debug_implementations)]

//! Neural network support for MLX
//!
//! All modules provide a `new()` function that take mandatory parameters and other methods
//! to set optional parameters.

mod activation;
mod container;
mod convolution;
mod convolution_transpose;
mod dropout;
mod embedding;
mod linear;
mod module_value_and_grad;
mod normalization;
mod pooling;
mod positional_encoding;
mod quantized;
mod recurrent;
mod transformer;
mod upsample;

pub use activation::*;
pub use container::*;
pub use convolution::*;
pub use convolution_transpose::*;
pub use dropout::*;
pub use embedding::*;
pub use linear::*;
pub use module_value_and_grad::*;
pub use normalization::*;
pub use pooling::*;
pub use positional_encoding::*;
pub use quantized::*;
pub use recurrent::*;
pub use transformer::*;
pub use upsample::*;
