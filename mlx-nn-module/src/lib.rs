//! Trait definitions for mlx-nn
//! 
//! This is a separate crate from `mlx-nn` to ease the iplementation of the `ModuleParameters` macro

mod module;
mod param;

pub use module::*;
pub use param::*;
