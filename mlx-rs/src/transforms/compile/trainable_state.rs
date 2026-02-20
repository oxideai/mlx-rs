//! Trainable state wrapper for efficient JIT compilation.
//!
//! This module provides [`TrainableState`], a wrapper that only exposes trainable
//! (non-frozen) parameters to the compile system. This is critical for JIT compilation
//! of large models like LLMs with LoRA, where only a small fraction of parameters
//! are trainable.
//!
//! # The Problem
//!
//! The standard [`Updatable`] implementation for modules returns ALL parameters.
//! For a model with 10M parameters but only 500K trainable (LoRA adapters):
//!
//! - `compile_with_state` captures all 10M arrays
//! - Training step only modifies 500K
//! - MLX compiler prunes 9.5M unchanged arrays from output
//! - State tracking breaks due to count mismatch
//!
//! # The Solution
//!
//! [`TrainableState`] wraps a model and optimizer, implementing [`Updatable`] to
//! return only trainable parameters plus optimizer state. This reduces state count
//! from millions to hundreds of thousands, enabling successful JIT compilation.
//!
//! # Example
//!
//! ```ignore
//! use mlx_rs::transforms::compile::{compile_with_state, TrainableState};
//!
//! let mut state = TrainableState::new(model, optimizer);
//!
//! let mut compiled_step = compile_with_state(
//!     |state: &mut TrainableState<M, O>, (inputs, labels): (&Array, &Array)| {
//!         // Training logic
//!     },
//!     None,
//! );
//!
//! // Run compiled training
//! let loss = compiled_step(&mut state, (&input_ids, &labels))?;
//! ```

use itertools::Itertools;

use crate::{
    module::{FlattenedModuleParamMut, ModuleParameters},
    optimizers::Optimizer,
    utils::Updatable,
    Array,
};

/// A wrapper that exposes only trainable parameters for JIT compilation.
///
/// This wrapper combines a model and optimizer, implementing [`Updatable`] to return
/// only the trainable (non-frozen) parameters plus optimizer state. This enables
/// efficient JIT compilation for large models with LoRA or other parameter-efficient
/// fine-tuning methods.
///
/// # State Ordering
///
/// The state arrays are returned in a consistent order:
/// 1. Trainable model parameters (sorted by name)
/// 2. Optimizer state arrays
///
/// This ordering is critical for compile correctness - inputs and outputs must match.
#[derive(Debug)]
pub struct TrainableState<M, O> {
    /// The model with trainable and frozen parameters.
    pub model: M,
    /// The optimizer holding state for trainable parameters.
    pub optimizer: O,
    /// Cached sorted keys for consistent ordering.
    /// We cache these to ensure identical ordering across calls.
    trainable_keys: Vec<String>,
}

impl<M, O> TrainableState<M, O>
where
    M: ModuleParameters,
    O: Optimizer,
{
    /// Create a new trainable state wrapper.
    ///
    /// This will cache the trainable parameter keys for consistent ordering.
    pub fn new(model: M, optimizer: O) -> Self {
        let trainable_keys = Self::compute_trainable_keys(&model);
        Self {
            model,
            optimizer,
            trainable_keys,
        }
    }

    /// Decompose the wrapper back into model and optimizer.
    pub fn into_parts(self) -> (M, O) {
        (self.model, self.optimizer)
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get a reference to the optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get a mutable reference to the optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get mutable references to both model and optimizer.
    ///
    /// This is useful when you need to borrow both at the same time,
    /// which the borrow checker won't allow with separate `model_mut()`
    /// and `optimizer_mut()` calls.
    pub fn as_parts_mut(&mut self) -> (&mut M, &mut O) {
        (&mut self.model, &mut self.optimizer)
    }

    /// Get the number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.trainable_keys.len()
    }

    /// Refresh the cached trainable keys.
    ///
    /// Call this if you've changed which parameters are frozen/unfrozen.
    pub fn refresh_trainable_keys(&mut self) {
        self.trainable_keys = Self::compute_trainable_keys(&self.model);
    }

    fn compute_trainable_keys(model: &M) -> Vec<String> {
        model
            .trainable_parameters()
            .flatten()
            .into_iter()
            .map(|(k, _)| k.to_string())
            .sorted()
            .collect()
    }
}

impl<M, O> Updatable for TrainableState<M, O>
where
    M: ModuleParameters,
    O: Updatable,
{
    fn updatable_states_len(&self) -> usize {
        self.trainable_keys.len() + self.optimizer.updatable_states_len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        // Get trainable parameters in cached order
        let params = self.model.trainable_parameters().flatten();
        let model_states: Vec<&Array> = self
            .trainable_keys
            .iter()
            .filter_map(|key| params.get(key.as_str()).copied())
            .collect();

        // Chain with optimizer state
        let opt_states: Vec<&Array> = self.optimizer.updatable_states().into_iter().collect();
        model_states.into_iter().chain(opt_states)
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        // Get trainable parameters in cached order
        let mut params: FlattenedModuleParamMut = self.model.parameters_mut().flatten();

        // Filter to only trainable keys and collect in order
        let model_states: Vec<&mut Array> = self
            .trainable_keys
            .iter()
            .filter_map(|key| params.remove(key.as_str()))
            .collect();

        // Collect optimizer state
        let opt_states: Vec<&mut Array> =
            self.optimizer.updatable_states_mut().into_iter().collect();

        // Chain them together
        model_states.into_iter().chain(opt_states)
    }
}

/// Extension trait for creating compiled training functions.
pub trait CompileTrainingExt: ModuleParameters + Sized {
    /// Wrap this model with an optimizer for JIT-compiled training.
    ///
    /// The returned [`TrainableState`] can be used with [`compile_with_state`] for
    /// efficient JIT compilation that only tracks trainable parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = model.with_optimizer(optimizer);
    /// let mut compiled = compile_with_state(training_step, None);
    /// compiled(&mut state, args)?;
    /// ```
    fn with_optimizer<O: Optimizer>(self, optimizer: O) -> TrainableState<Self, O> {
        TrainableState::new(self, optimizer)
    }
}

impl<T: ModuleParameters> CompileTrainingExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use crate::nn::Linear;
    use crate::optimizers::AdamWBuilder;

    #[test]
    fn test_trainable_state_creation() {
        // Create a simple model
        let linear = Linear::new(4, 4).unwrap();

        // Create optimizer
        let optimizer = AdamWBuilder::new(0.001).build().unwrap();

        let state = TrainableState::new(linear, optimizer);

        // Linear has 2 params: weight and bias (both trainable by default)
        assert_eq!(state.num_trainable_params(), 2);

        // updatable_states should return weight + bias + optimizer state
        // Before optimizer init, optimizer has 0 state
        let states: Vec<_> = state.updatable_states().into_iter().collect();
        assert_eq!(states.len(), 2); // weight + bias, optimizer not initialized yet
    }

    #[test]
    fn test_trainable_state_with_frozen_params() {
        // Create a simple model
        let mut linear = Linear::new(4, 4).unwrap();

        // Freeze all parameters first, then unfreeze just the weight
        linear.freeze_parameters(true);

        // Create optimizer
        let optimizer = AdamWBuilder::new(0.001).build().unwrap();

        let state = TrainableState::new(linear, optimizer);

        // All params are frozen, so no trainable params
        assert_eq!(state.num_trainable_params(), 0);

        let states: Vec<_> = state.updatable_states().into_iter().collect();
        assert_eq!(states.len(), 0); // No trainable params
    }
}
