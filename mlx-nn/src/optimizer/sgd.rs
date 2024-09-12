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

    /// Momentum strength. Default to `0` if not specified.
    pub momentum: f32,

    /// Weight decay (L2 penalty). Default to `0` if not specified.
    pub weight_decay: f32,

    /// Dampening for momentum. Default to `0` if not specified.
    pub dampening: f32,

    /// Enables nesterov momentum. Default to `false` if not specified.
    pub nesterov: bool,

    /// Inner state
    pub state: FlattenedModuleParam,
}

/// Optional configs for the SGD optimizer.
#[derive(Debug, Clone)]
pub struct SgdOptions {
    /// Momentum strength
    pub momentum: Option<f32>,

    /// Weight decay (L2 penalty)
    pub weight_decay: Option<f32>,

    /// Dampening for momentum
    pub dampening: Option<f32>,

    /// Enables nesterov momentum
    pub nesterov: Option<bool>,
}

impl Sgd {
    /// Creates a new `Sgd` optimizer.
    pub fn new(lr: f32, options: SgdOptions) -> Self {
        Self {
            lr,
            momentum: options.momentum.unwrap_or(0.0),
            weight_decay: options.weight_decay.unwrap_or(0.0),
            dampening: options.dampening.unwrap_or(0.0),
            nesterov: options.nesterov.unwrap_or(false),
            state: FlattenedModuleParam::new(),
        }
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
