//! This mod defines the traits for neural network modules and parameters.
//!
//! This is to separate the trait definitions from the implementations, which are in the `mlx-nn`
//! crate. This also allows using the `mlx_macros::ModuleParameters` derive macro in crates other
//! than `mlx-nn`.

mod cache;
#[allow(clippy::module_inception)]
mod module;
mod param;

pub use cache::*;
pub use module::*;
pub use param::*;
