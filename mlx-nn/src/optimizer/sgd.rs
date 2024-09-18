use std::rc::Rc;

use mlx_nn_module::FlattenedModuleParam;
use mlx_rs::{array, Array};

use crate::utils::get_mut_or_insert_with;

use super::Optimizer;

/// Stochastic gradient descent optimizer.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate
    pub lr: f32,

    /// Momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
    pub momentum: f32,

    /// Weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
    pub weight_decay: f32,

    /// Dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
    pub dampening: f32,

    /// Enables nesterov momentum. Default to [`Sgd::DEFAULT_NESTEROV`] if not specified.
    pub nesterov: bool,

    /// Inner state
    pub state: FlattenedModuleParam,
}

impl Sgd {
    /// Default momentum if not specified.
    pub const DEFAULT_MOMENTUM: f32 = 0.0;

    /// Default weight decay if not specified.
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;

    /// Default dampening if not specified.
    pub const DEFAULT_DAMPENING: f32 = 0.0;

    /// Default nesterov if not specified.
    pub const DEFAULT_NESTEROV: bool = false;

    /// Creates a new `Sgd` optimizer.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: Self::DEFAULT_MOMENTUM,
            weight_decay: Self::DEFAULT_WEIGHT_DECAY,
            dampening: Self::DEFAULT_DAMPENING,
            nesterov: Self::DEFAULT_NESTEROV,
            state: FlattenedModuleParam::new(),
        }
    }

    /// Sets the momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
    pub fn with_momentum(mut self, momentum: impl Into<Option<f32>>) -> Self {
        self.momentum = momentum.into().unwrap_or(Self::DEFAULT_MOMENTUM);
        self
    }

    /// Sets the weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
    pub fn with_weight_decay(mut self, weight_decay: impl Into<Option<f32>>) -> Self {
        self.weight_decay = weight_decay.into().unwrap_or(Self::DEFAULT_WEIGHT_DECAY);
        self
    }

    /// Sets the dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
    pub fn with_dampening(mut self, dampening: impl Into<Option<f32>>) -> Self {
        self.dampening = dampening.into().unwrap_or(Self::DEFAULT_DAMPENING);
        self
    }

    /// Enables nesterov momentum. Default to [`Sgd::DEFAULT_NESTEROV`] if not specified.
    pub fn with_nesterov(mut self, nesterov: impl Into<Option<bool>>) -> Self {
        self.nesterov = nesterov.into().unwrap_or(Self::DEFAULT_NESTEROV);
        self
    }

    /// Apply SGD to a single parameter. Returns the updated parameter and the updated state.
    #[inline]
    fn update_single(&mut self, key: Rc<str>, mut gradient: Array, parameter: &mut Array) {
        let state = get_mut_or_insert_with(&mut self.state, &key, || array!(0.0));

        // Apply weight decay
        if self.weight_decay != 0.0 {
            gradient = &gradient + array!(self.weight_decay) * &*parameter;
        }

        let lr = array!(self.lr);

        // Apply momentum
        if self.momentum <= 0.0 {
            *parameter = &*parameter - &lr * &gradient;
            return;
        }

        let momentum = array!(self.momentum);
        let mut v = &*state * &momentum;
        if self.dampening > 0.0 {
            v = &v + (&array!(1.0 - self.dampening) * &gradient);
        } else {
            v = &v + &gradient;
        }

        match self.nesterov {
            true => {
                let update = gradient + (&momentum * &v);
                *parameter = &*parameter - &lr * update;
                *state = v;
            }
            false => {
                let update = &v;
                *parameter = &*parameter - &lr * update;
                *state = v;
            }
        }
    }
}

impl Optimizer for Sgd {
    fn update<M>(&mut self, model: &mut M, gradients: mlx_nn_module::FlattenedModuleParam)
    where
        M: mlx_nn_module::ModuleParameters,
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
mod tests {

    use mlx_macros::ModuleParameters;
    use mlx_nn_module::{ModuleParameters, Param};
    use mlx_rs::ops::{ones, zeros};

    use super::*;

    #[derive(Debug, ModuleParameters)]
    struct First {
        #[param]
        a: Param<Array>,

        #[param]
        b: Param<Array>,
    }

    #[derive(Debug, ModuleParameters)]
    struct Model {
        #[param]
        first: Param<First>,

        #[param]
        second: Param<Array>,
    }

    #[test]
    fn test_sgd() {
        let first = First {
            a: Param::new(zeros::<f32>(&[10]).unwrap()),
            b: Param::new(zeros::<f32>(&[1]).unwrap()),
        };
        let mut model = Model {
            first: Param::new(first),
            second: Param::new(zeros::<f32>(&[1]).unwrap()),
        };
        // let param_map = param.parameters_mut().flatten();
        let grads_map: FlattenedModuleParam = model
            .parameters()
            .flatten()
            .iter()
            .map(|(k, v)| {
                let g = ones::<f32>(v.shape()).unwrap();
                (k.clone(), g)
            })
            .collect();

        let mut optim = Sgd::new(1e-2).with_momentum(0.9);
        optim.update(&mut model, grads_map);

        let expected_first_a = ones::<f32>(&[10]).unwrap() * array!(-0.01);
        let expected_first_b = ones::<f32>(&[1]).unwrap() * array!(-0.01);
        let expected_second = ones::<f32>(&[1]).unwrap() * array!(-0.01);
        assert_eq!(model.first.a.as_ref(), &expected_first_a);
        assert_eq!(model.first.b.as_ref(), &expected_first_b);
        assert_eq!(model.second.as_ref(), &expected_second);

        let expected_state_first_a = ones::<f32>(&[10]).unwrap();
        let expected_state_first_b = ones::<f32>(&[1]).unwrap();
        let expected_state_second = ones::<f32>(&[1]).unwrap();

        assert_eq!(optim.state.get("first.a"), Some(&expected_state_first_a));
        assert_eq!(optim.state.get("first.b"), Some(&expected_state_first_b));
        assert_eq!(optim.state.get("second"), Some(&expected_state_second));
    }
}
