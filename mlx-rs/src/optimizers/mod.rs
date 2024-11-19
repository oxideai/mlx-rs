//! Trait and implementations for optimizers.

#![deny(missing_docs)]

use std::{borrow::Borrow, collections::HashMap, rc::Rc};

use crate::{
    array, error::Exception, module::{FlattenedModuleParam, ModuleParameters}, utils::OwnedOrRef, Array
};

mod adadelta;
mod adafactor;
mod adagrad;
mod adam;
mod adamax;
mod adamw;
mod lion;
mod rmsprop;
mod sgd;

pub use adadelta::*;
pub use adafactor::*;
pub use adagrad::*;
pub use adam::*;
pub use adamax::*;
pub use adamw::*;
pub use lion::*;
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
    ) -> crate::error::Result<()>;

    /// Apply the gradients to the parameters of the model and update the model with the new
    /// parameters.
    fn apply<M>(
        &mut self,
        model: &mut M,
        gradients: impl Borrow<FlattenedModuleParam>,
    ) -> crate::error::Result<()>
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

/// Clips the global norm of the gradients
///
/// This function ensures that the global norm of the gradients does not exceed
/// `max_norm`. It scales down the gradients proportionally if their norm is
/// greater than `max_norm`.
pub fn clip_grad_norm<'a>(
    gradients: &'a FlattenedModuleParam,
    max_norm: f32,
) -> (HashMap<Rc<str>, OwnedOrRef<'a, Array>>, f32) {
    let total_norm: f32 = gradients
        .values()
        .fold(array!(0.0), |acc, grad| {
            acc + grad
                .square()
                .sum(None, None)
                .expect("Sum with default axes should not fail")
        })
        .sqrt()
        .item();
    let normalizer = array!(max_norm / (total_norm + 1e-6));

    let clipped_gradients: HashMap<_, _> = gradients
        .iter()
        .map(|(key, grad)| {
            let clipped_grad = if total_norm < max_norm {
                OwnedOrRef::Ref(grad)
            } else {
                OwnedOrRef::Owned(grad * &normalizer)
            };
            (key.clone(), clipped_grad)
        })
        .collect();
    (clipped_gradients, total_norm)
}
