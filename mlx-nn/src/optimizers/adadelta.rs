use std::{collections::HashMap, rc::Rc};

use mlx_internal_macros::generate_builder;
use mlx_rs::{array, ops::sqrt, Array};

use crate::error::AdaDeltaBuildError;

use super::*;

type State = HashMap<Rc<str>, (Array, Array)>;

generate_builder! {
    /// The AdaDelta optimizer with a learning rate
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
    #[derive(Debug)]
    #[generate_builder(generate_build_fn = false)]
    pub struct AdaDelta {
        /// The learning rate
        pub lr: Array,

        /// The coefficient used for computing a running average of squared gradients. Default to
        /// [`AdaDelta::DEFAULT_RHO`].
        #[optional(ty = f32)]
        pub rho: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`AdaDelta::DEFAULT_EPS`].
        #[optional(ty = f32)]
        pub eps: Array,

        /// Inner state
        pub state: State,
    }
}

impl AdaDeltaBuilder {
    /// Builds a new [`AdaDelta`] optimizer
    pub fn build(self, lr: f32) -> Result<AdaDelta, AdaDeltaBuildError> {
        let rho = self.rho.unwrap_or(AdaDelta::DEFAULT_RHO);
        let eps = self.eps.unwrap_or(AdaDelta::DEFAULT_EPS);

        if rho < 0.0 {
            return Err(AdaDeltaBuildError::NegativeRho);
        }

        if eps < 0.0 {
            return Err(AdaDeltaBuildError::NegativeEps);
        }

        Ok(AdaDelta {
            lr: array!(lr),
            rho: array!(rho),
            eps: array!(eps),
            state: State::new(),
        })
    }
}

impl AdaDelta {
    /// Default value for `rho`
    pub const DEFAULT_RHO: f32 = 0.99;

    /// Default value for `eps`
    pub const DEFAULT_EPS: f32 = 1e-6;

    /// Creates a new AdaDelta optimizer with all optional parameters set to their default values
    pub fn new(lr: f32) -> Self {
        // SAFETY: The default values are valid
        Self::builder().build(lr).unwrap()
    }
}

impl Optimizer for AdaDelta {
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), mlx_rs::error::Exception> {
        let (v, u) = self
            .state
            .entry(key.clone())
            .or_insert_with(|| (array!(0.0), array!(0.0)));

        let one_minus_rho = array!(1.0).subtract(&self.rho)?;
        let first_term = self.rho.multiply(&v)?;
        let second_term = one_minus_rho.multiply(gradient.square())?;
        let v_new = first_term.add(&second_term)?;

        let num = sqrt(&u.add(&self.eps)?);
        let den = sqrt(&v_new.add(&self.eps)?);
        let d = num.divide(&den)?.multiply(gradient)?;
        let first_term = self.rho.multiply(&u)?;
        let second_term = one_minus_rho.multiply(d.square())?;
        let u_new = first_term.add(&second_term)?;

        let param_new = parameter.subtract(self.lr.multiply(d)?)?;

        *parameter = param_new;

        *v = v_new;
        *u = u_new;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mlx_macros::ModuleParameters;
    use mlx_rs::assert_array_eq;
    use mlx_rs::module::Param;

    use super::*;

    #[derive(Debug, ModuleParameters)]
    struct Model {
        #[param]
        a: Param<Array>,
    }

    // This unit test is adapted from the swift binding unit test `testAdaDelta` in
    // `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
    #[test]
    fn test_ada_delta() {
        mlx_rs::random::seed(547);
        let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
        assert_eq!(a.shape(), &[4, 3]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);
        assert_array_eq!(
            a.mean(None, None).unwrap(),
            array!(-0.348_337_02),
            0.006966740489006043
        );
        assert_array_eq!(
            a.sum(None, None).unwrap(),
            array!(-4.180_044),
            0.08360088348388672
        );

        let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
        assert_eq!(a_grad.shape(), &[4, 3]);
        assert_eq!(a_grad.dtype(), mlx_rs::Dtype::Float32);
        assert_array_eq!(
            a_grad.mean(None, None).unwrap(),
            array!(0.522_678_4),
            0.010453567504882813
        );
        assert_array_eq!(
            a_grad.sum(None, None).unwrap(),
            array!(6.272_14),
            0.12544280052185058
        );

        let mut a_model = Model {
            a: Param::new(a.clone()),
        };
        let mut a_grad_params = FlattenedModuleParam::new();
        a_grad_params.insert("a".into(), a_grad.clone());

        let mut optimizer = AdaDelta::new(0.1);

        optimizer.apply(&mut a_model, a_grad_params).unwrap();
        assert_eq!(a_model.a.shape(), &[4, 3]);
        assert_eq!(a_model.a.dtype(), mlx_rs::Dtype::Float32);
        assert_array_eq!(
            a_model.a.mean(None, None).unwrap(),
            array!(-0.348_442_4),
            0.348442405462265
        );
        assert_array_eq!(
            a_model.a.sum(None, None).unwrap(),
            array!(-4.181_308_7),
            0.08362617492675782
        );
    }
}
