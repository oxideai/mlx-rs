use std::rc::Rc;

use mlx_rs::{array, Array};

use crate::utils::get_mut_or_insert_with;

use super::*;

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
    pub state: OptimizerState,
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
            state: OptimizerState::new(),
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
}

impl Optimizer for Sgd {
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

#[cfg(test)]
mod tests {
    use mlx_rs::assert_array_eq;
    use mlx_rs::ops::ones;

    use super::optim_test_util::*;
    use super::*;

    // This unit test is adapted from the python unit test `test_sgd` in
    // `mlx/python/tests/test_optimizers.py`
    #[test]
    fn test_sgd() {
        let (mut model, gradients) = create_default_test_model_and_grads();

        let mut optim = Sgd::new(1e-2).with_momentum(0.9);
        optim.update(&mut model, gradients);

        let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.01;
        let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.01;
        let expected_second = ones::<f32>(&[1]).unwrap() * -0.01;

        assert_array_eq!(model.first.a.as_ref(), expected_first_a, ATOL);
        assert_array_eq!(model.first.b.as_ref(), expected_first_b, ATOL);
        assert_array_eq!(model.second.as_ref(), expected_second, ATOL);

        let expected_state_first_a = ones::<f32>(&[10]).unwrap();
        let expected_state_first_b = ones::<f32>(&[1]).unwrap();
        let expected_state_second = ones::<f32>(&[1]).unwrap();

        assert_array_eq!(
            optim.state["first.a"].as_ref(),
            expected_state_first_a,
            ATOL
        );
        assert_array_eq!(
            optim.state["first.b"].as_ref(),
            expected_state_first_b,
            ATOL
        );
        assert_array_eq!(optim.state["second"].as_ref(), expected_state_second, ATOL);
    }
}
