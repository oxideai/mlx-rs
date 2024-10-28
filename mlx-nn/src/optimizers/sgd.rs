use std::rc::Rc;

use mlx_internal_macros::generate_builder;
use mlx_rs::{array, Array};

use crate::utils::get_mut_or_insert_with;

use super::*;

generate_builder! {
    /// Stochastic gradient descent optimizer.
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Sgd {
        /// Learning rate
        pub lr: f32,

        /// Momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
        #[optional]
        pub momentum: f32,

        /// Weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
        #[optional]
        pub weight_decay: f32,

        /// Dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
        #[optional]
        pub dampening: f32,

        /// Enables nesterov momentum. Default to [`Sgd::DEFAULT_NESTEROV`] if not specified.
        #[optional]
        pub nesterov: bool,

        /// Inner state
        pub state: OptimizerState,
    }
}

impl SgdBuilder {
    /// Builds a new [`Sgd`].
    pub fn build(self, lr: f32) -> Sgd {
        let momentum = self.momentum.unwrap_or(Sgd::DEFAULT_MOMENTUM);
        let weight_decay = self.weight_decay.unwrap_or(Sgd::DEFAULT_WEIGHT_DECAY);
        let dampening = self.dampening.unwrap_or(Sgd::DEFAULT_DAMPENING);
        let nesterov = self.nesterov.unwrap_or(Sgd::DEFAULT_NESTEROV);

        Sgd {
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            state: OptimizerState::new(),
        }
    }
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
        Self::builder().build(lr)
    }
}

impl Optimizer for Sgd {
    /// Apply SGD to a single parameter. Returns the updated parameter and the updated state.
    #[inline]
    fn update_single(&mut self, key: Rc<str>, mut gradient: Array, parameter: &mut Array) -> Result<(), Exception> {
        let state = get_mut_or_insert_with(&mut self.state, &key, || array!(0.0));

        // Apply weight decay
        if self.weight_decay != 0.0 {
            let weight_decay = array!(self.weight_decay);
            gradient = weight_decay.multiply(&*parameter)?.add(&gradient)?;
        }

        let lr = array!(self.lr);

        // Apply momentum
        if self.momentum <= 0.0 {
            *parameter = parameter.subtract(lr.multiply(&gradient)?)?;
            return Ok(());
        }

        let momentum = array!(self.momentum);
        let mut v = state.multiply(&momentum)?;
        if self.dampening > 0.0 {
            let one_minus_dampening = array!(1.0 - self.dampening);
            v = v.add(&one_minus_dampening.multiply(&gradient)?)?;
        } else {
            v = v.add(&gradient)?;
        }

        match self.nesterov {
            true => {
                let update = gradient.add(&momentum.multiply(&v)?)?;
                *parameter = parameter.subtract(lr.multiply(&update)?)?;
                *state = v;
            }
            false => {
                let update = &v;
                *parameter = parameter.subtract(lr.multiply(update)?)?;
                *state = v;
            }
        }

        Ok(())
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

        let mut optim = Sgd::builder().momentum(0.9).build(1e-2);
        optim.update(&mut model, gradients).unwrap();

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
