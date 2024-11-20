//! Trait and implementations for optimizers.

#![deny(missing_docs)]

use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
    rc::Rc,
};

use crate::{
    array,
    module::{FlattenedModuleParam, ModuleParameters},
    Array,
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
pub fn clip_grad_norm(
    gradients: &FlattenedModuleParam,
    max_norm: f32,
) -> crate::error::Result<(HashMap<Rc<str>, Cow<'_, Array>>, f32)> {
    let total_norm: f32 = gradients
        .values()
        .fold(Ok(array!(0.0)), |acc, grad| {
            acc?.add(&grad.square()?.sum(None, None)?)
        })?
        .sqrt()?
        .item();
    let normalizer = array!(max_norm / (total_norm + 1e-6));

    let clipped_gradients: HashMap<_, _> = gradients
        .iter()
        .map(|(key, grad)| {
            let clipped_grad = if total_norm < max_norm {
                Cow::Borrowed(grad)
            } else {
                Cow::Owned(grad * &normalizer)
            };
            (key.clone(), clipped_grad)
        })
        .collect();
    Ok((clipped_gradients, total_norm))
}

#[cfg(test)]
mod tests {

    // def test_clip_grad_norm(self):
    //     # Test with small gradients that do not require clipping
    //     small_grads = {
    //         "first": [mx.array([0.1, 0.2]), mx.array([0.1])],
    //         "second": mx.array([0.3]),
    //     }
    //     max_norm = 10.0  # A large max_norm that shouldn't trigger clipping
    //     clipped_grads, total_norm = opt.clip_grad_norm(small_grads, max_norm)
    //     self.assertTrue(
    //         tree_equal(lambda x, y: mx.array_equal(x, y), small_grads, clipped_grads),
    //         "Gradients should not be modified when clipping is not necessary.",
    //     )

    //     # Test with large gradients that require clipping
    //     large_grads = {
    //         "first": [mx.array([10, 20]), mx.array([10])],
    //         "second": mx.array([30]),
    //     }
    //     max_norm = 1.0  # A small max_norm that should trigger clipping
    //     clipped_grads, total_norm = opt.clip_grad_norm(large_grads, max_norm)
    //     # Correctly extract only the gradient values for norm calculation
    //     clipped_values = [value for _, value in tree_flatten(clipped_grads)]
    //     norm_of_clipped = mx.sqrt(
    //         sum(mx.square(g).sum() for g in clipped_values)
    //     ).item()
    //     self.assertAlmostEqual(
    //         norm_of_clipped,
    //         max_norm,
    //         places=6,
    //         msg="Clipped gradients norm should be close to the specified max_norm.",
    //     )

    //     # Ensures that the scaling was done correctly
    //     scale = max_norm / total_norm
    //     expected_grads = tree_map(lambda g: g * scale, large_grads)
    //     self.assertTrue(
    //         tree_equal(
    //             lambda x, y: mx.allclose(x, y, atol=1e-6), expected_grads, clipped_grads
    //         ),
    //         "Gradients were not scaled correctly during clipping.",
    //     )

    #[test]
    fn test_clip_grad_norm() {
        todo!()
    }
}