//! Trait and implementations for optimizers.

use mlx_nn_module::{FlattenedModuleParam, ModuleParameters};

mod sgd;

pub use sgd::*;

/// Trait for optimizers.
pub trait Optimizer {
    /// Apply the gradients to the parameters of the model and update the model with the new
    /// parameters.
    fn update<M>(&mut self, model: &mut M, gradients: FlattenedModuleParam)
    where
        M: ModuleParameters;
}
