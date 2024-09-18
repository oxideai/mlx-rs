//! Trait and implementations for optimizers.

use std::rc::Rc;

use mlx_nn_module::{FlattenedModuleParam, ModuleParameters};

mod rmsprop;
mod sgd;

use mlx_rs::Array;
pub use rmsprop::*;
pub use sgd::*;

type OptimizerState = FlattenedModuleParam;

/// Trait for optimizers.
pub trait Optimizer {
    /// Update a single parameter with the given gradient.
    fn update_single(&mut self, key: Rc<str>, gradient: Array, parameter: &mut Array);

    /// Apply the gradients to the parameters of the model and update the model with the new
    /// parameters.
    fn update<M>(&mut self, model: &mut M, gradients: FlattenedModuleParam)
    where
        M: ModuleParameters,
    {
        let mut parameters = model.parameters_mut().flatten();

        for (key, gradient) in gradients {
            if let Some(parameter) = parameters.get_mut(&key) {
                self.update_single(key, gradient, parameter);
            }
        }
    }
}

#[cfg(test)]
mod optim_test_util {
    use mlx_macros::ModuleParameters;
    use mlx_nn_module::{FlattenedModuleParam, ModuleParameters, Param};
    use mlx_rs::{
        ops::{ones, zeros},
        Array,
    };

    #[derive(Debug, ModuleParameters)]
    pub(super) struct First {
        #[param]
        pub a: Param<Array>,

        #[param]
        pub b: Param<Array>,
    }

    #[derive(Debug, ModuleParameters)]
    pub(super) struct Model {
        #[param]
        pub first: Param<First>,

        #[param]
        pub second: Param<Array>,
    }

    pub(super) type GradsMap = FlattenedModuleParam;

    pub(super) fn create_default_test_model_and_grads() -> (Model, GradsMap) {
        let first = First {
            a: Param::new(zeros::<f32>(&[10]).unwrap()),
            b: Param::new(zeros::<f32>(&[1]).unwrap()),
        };
        let model = Model {
            first: Param::new(first),
            second: Param::new(zeros::<f32>(&[1]).unwrap()),
        };

        let grads_map: GradsMap = model
            .parameters()
            .flatten()
            .iter()
            .map(|(k, v)| {
                let g = ones::<f32>(v.shape()).unwrap();
                (k.clone(), g)
            })
            .collect();

        (model, grads_map)
    }

    pub(super) const ATOL: f64 = 1e-5;
}
