//! Trait and implementations for optimizers.

use std::rc::Rc;

use mlx_rs::module::{FlattenedModuleParam, ModuleParameters};

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
    use mlx_rs::module::{FlattenedModuleParam, ModuleParameters, Param};
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

#[cfg(test)]
mod tests {
    use mlx_macros::ModuleParameters;
    use mlx_rs::{
        array,
        error::Exception,
        module::{Module, ModuleParameters, Param},
        random::uniform,
        transforms::{eval, eval_params},
        Array,
    };

    use crate::{
        losses::{mse_loss, LossReduction, MseLossOptions},
        module_value_and_grad,
    };

    use super::*;

    /// A helper model for testing optimizers.
    ///
    /// This is adapted from the swift binding tests in `mlx-swift/Tests/MLXTests/OptimizerTests.swift`.
    #[derive(Debug, ModuleParameters)]
    struct LinearFunctionModel {
        #[param]
        pub m: Param<Array>,

        #[param]
        pub b: Param<Array>,
    }

    impl Module for LinearFunctionModel {
        type Error = Exception;

        fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
            self.m.multiply(x)?.add(&self.b)
        }

        fn training_mode(&mut self, _mode: bool) {}
    }

    impl LinearFunctionModel {
        pub fn new() -> Result<Self, Exception> {
            let m = uniform::<_, f32>(-5.0, 5.0, None, None)?;
            let b = uniform::<_, f32>(-5.0, 5.0, None, None)?;
            Ok(Self {
                m: Param::new(m),
                b: Param::new(b),
            })
        }
    }

    pub fn train<F, O>(f: F, steps: usize) -> Result<Array, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> O,
        O: Optimizer,
    {
        let mut optimizer = f();

        let options = MseLossOptions::builder()
            .reduction(LossReduction::Mean)
            .build();
        let loss = |model: &LinearFunctionModel, (x, y): (&Array, &Array)| {
            mse_loss(model.forward(x)?, y, &options)
        };

        // TODO: check compiled model once we have it
        let mut model = LinearFunctionModel::new()?;
        eval_params(model.parameters())?;

        let m = array!(0.25);
        let b = array!(7.0);

        let mut lg = module_value_and_grad(loss);

        let mut last_loss = None;
        for _ in 0..steps {
            // println!("target: b = {}, m = {}", b, m);
            // println!("parameters: {:?}", model.parameters());

            // generate random training data along with the ground truth.
            // notice that the shape is [B, 1] where B is the batch
            // dimension -- this allows us to train on 10 samples simultaneously
            let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], None)?;
            let y = &m * &x + &b;
            eval([&x, &y])?;

            // compute the loss and gradients.  use the optimizer
            // to adjust the parameters closer to the target
            let (loss, g) = lg(&mut model, (&x, &y))?;
            optimizer.update(&mut model, g);

            eval_params(model.parameters())?;

            last_loss = Some(loss);
        }

        Ok(last_loss.unwrap())
    }

    const NUM_TRIALS: usize = 3;

    #[test]
    fn test_sgd_converges() {
        let mut total_loss = 0.0;
        for _ in 0..NUM_TRIALS {
            let loss = train(|| Sgd::new(0.1), 30).unwrap();
            total_loss += loss.item::<f32>();
        }
        // It sometimes doesn't converge that fast, so we take the average loss
        // across multiple trials
        let avg_loss = total_loss / NUM_TRIALS as f32;
        assert!(avg_loss < 0.1, "avg loss: {}", avg_loss);
    }

    #[test]
    fn test_rmsprop_converges() {
        let mut total_loss = 0.0;
        for _ in 0..NUM_TRIALS {
            // RMSProp doesn't seem to converge as fast as SGD
            let loss = train(|| RmsProp::new(0.1), 100).unwrap();
            total_loss += loss.item::<f32>();
        }
        // It sometimes doesn't converge that fast, so we take the average loss
        // across multiple trials
        let avg_loss = total_loss / NUM_TRIALS as f32;
        assert!(avg_loss < 0.1, "avg loss: {}", avg_loss);
    }
}
