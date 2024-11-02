//! Trait and implementations for optimizers.

use std::{borrow::Borrow, collections::HashMap, rc::Rc};

use crate::{
    error::Exception,
    module::{FlattenedModuleParam, ModuleParameters},
    Array,
};

mod adadelta;
mod adagrad;
mod adam;
mod rmsprop;
mod sgd;

pub use adadelta::*;
pub use adagrad::*;
pub use adam::*;
pub use rmsprop::*;
pub use sgd::*;

type OptimizerState<T = Array> = HashMap<Rc<str>, T>;

/// Trait for optimizers.
pub trait Optimizer {
    /// Update a single parameter with the given gradient.
    ///
    /// The implementation should look up the state for the parameter using the key and update the
    /// state and the parameter accordingly. The key is provided instead of the state because it
    /// would otherwise create a mutable borrow conflict with the rest of the optimizer fields.
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception>;

    /// Apply the gradients to the parameters of the model and update the model with the new
    /// parameters.
    fn apply<M>(
        &mut self,
        model: &mut M,
        gradients: impl Borrow<FlattenedModuleParam>,
    ) -> Result<(), Exception>
    where
        M: ModuleParameters,
    {
        let mut parameters = model.parameters_mut().flatten();

        for (key, gradient) in gradients.borrow().iter() {
            if let Some(parameter) = parameters.get_mut(key) {
                self.apply_single(key, gradient, parameter)?;
            }
        }

        Ok(())
    }
}
