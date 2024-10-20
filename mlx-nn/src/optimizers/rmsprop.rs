use std::rc::Rc;

use mlx_internal_macros::GenerateBuilder;
use mlx_rs::{
    array,
    ops::{sqrt, square},
    Array,
};

use crate::{error::RmsPropBuildError, utils::get_mut_or_insert_with};

use super::*;

/// The RMSprop optimizer [1].
///
/// [1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for
///     machine learning
#[derive(Debug, Clone, GenerateBuilder)]
#[generate_builder(generate_build_fn = false)]
pub struct RmsProp {
    /// Learning rate
    pub lr: f32,

    /// The smoothing constant. Default to [`RmsProp::DEFAULT_ALPHA`] if not specified.
    #[optional]
    pub alpha: f32,

    /// The epsilon added to the denominator to improve numerical stability. Default to
    /// [`RmsProp::DEFAULT_EPSILON`] if not specified.
    #[optional]
    pub epsilon: f32,

    /// Inner state
    pub state: OptimizerState,
}

impl RmsPropBuilder {
    /// Builds a new [`RmsProp`].
    ///
    /// # Params
    ///
    /// - `lr`: The learning rate.
    pub fn build(self, lr: f32) -> Result<RmsProp, RmsPropBuildError> {
        let alpha = self.alpha.unwrap_or(RmsProp::DEFAULT_ALPHA);
        let epsilon = self.epsilon.unwrap_or(RmsProp::DEFAULT_EPSILON);

        if alpha < 0.0 {
            return Err(RmsPropBuildError::NegativeAlpha);
        }

        if epsilon < 0.0 {
            return Err(RmsPropBuildError::NegativeEpsilon);
        }

        Ok(RmsProp {
            lr,
            alpha,
            epsilon,
            state: OptimizerState::new(),
        })
    }
}

impl RmsProp {
    /// Default alpha if not specified.
    pub const DEFAULT_ALPHA: f32 = 0.99;

    /// Default epsilon if not specified.
    pub const DEFAULT_EPSILON: f32 = 1e-8;

    /// Creates a new `RmsProp` optimizer with all optional params set to their default values.
    ///
    /// # Params
    ///
    /// - `lr`: The learning rate.
    pub fn new(lr: f32) -> Self {
        Self::builder().build(lr).expect("Default values are valid")
    }
}

impl Optimizer for RmsProp {
    fn update_single(&mut self, key: Rc<str>, gradient: Array, parameter: &mut Array) {
        let state = get_mut_or_insert_with(&mut self.state, &key, || array!(0.0));

        let lr = array!(self.lr);
        let alpha = array!(self.alpha);
        let eps = array!(self.epsilon);

        let v = &alpha * &*state + (array!(1.0) - &alpha) * square(&gradient);
        let new_param = &*parameter - &lr * &gradient / (sqrt(&v) + &eps);

        *parameter = new_param;
        *state = v;
    }
}

#[cfg(test)]
mod tests {

    use mlx_rs::assert_array_eq;
    use mlx_rs::ops::ones;

    use super::optim_test_util::*;
    use super::*;

    // This unit test is adapted from the python unit test `test_rmsprop` in
    // `tests/test_optimizer.py`.
    #[test]
    fn test_rmsprop() {
        const LR: f32 = 1e-2;
        const ALPHA: f32 = 0.99;

        let (mut model, gradients) = create_default_test_model_and_grads();

        let mut optim = RmsProp::builder().alpha(ALPHA).build(LR).unwrap();
        optim.update(&mut model, gradients);

        let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.1;
        let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.1;
        let expected_second = ones::<f32>(&[1]).unwrap() * -0.1;

        assert_array_eq!(model.first.a.as_ref(), expected_first_a, ATOL);
        assert_array_eq!(model.first.b.as_ref(), expected_first_b, ATOL);
        assert_array_eq!(model.second.as_ref(), expected_second, ATOL);

        let expected_state_first_a = ones::<f32>(&[10]).unwrap() * 0.01;
        let expected_state_first_b = ones::<f32>(&[1]).unwrap() * 0.01;
        let expected_state_second = ones::<f32>(&[1]).unwrap() * 0.01;

        assert_array_eq!(
            optim.state.get("first.a").unwrap(),
            expected_state_first_a,
            ATOL
        );
        assert_array_eq!(
            optim.state.get("first.b").unwrap(),
            expected_state_first_b,
            ATOL
        );
        assert_array_eq!(
            optim.state.get("second").unwrap(),
            expected_state_second,
            ATOL
        );
    }
}
