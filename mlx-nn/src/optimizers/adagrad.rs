use std::rc::Rc;

use mlx_internal_macros::generate_builder;
use mlx_rs::{array, error::Exception, ops::square, Array};

use crate::utils::get_mut_or_insert_with;

use super::*;

generate_builder! {
    /// The Adagrad optimizer.
    /// 
    /// Please refer to the original paper for more details:
    /// 
    /// [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online
    ///     learning and stochastic optimization. JMLR 2011.
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct AdaGrad {
        /// Learning rate
        pub lr: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`AdaGrad::DEFAULT_EPS`].
        #[optional(ty = f32)]
        pub eps: Array,

        /// Inner state
        pub state: OptimizerState,
    }
}

impl AdaGradBuilder {
    /// Builds a new [`AdaGrad`].
    pub fn build(self, lr: f32) -> AdaGrad {
        let eps = array!(self.eps.unwrap_or(AdaGrad::DEFAULT_EPS));

        AdaGrad {
            lr: array!(lr),
            eps,
            state: OptimizerState::new(),
        }
    }
}

impl AdaGrad {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-8;

    /// Creates a new AdaGrad optimizer with all optional parameters set to their default values.
    pub fn new(lr: f32) -> AdaGrad {
        Self::builder().build(lr)
    }
}

impl Optimizer for AdaGrad {
    fn apply_single(&mut self, key: &Rc<str>, gradient: &Array, parameter: &mut Array) -> Result<(), Exception> {
        let state = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let v = state.add(square(gradient))?;

        let num = self.lr.multiply(gradient)?;
        let den = v.sqrt().add(&self.eps)?;
        let new_param = parameter.subtract(num.divide(&den)?)?;

        *state = v;
        *parameter = new_param;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mlx_macros::ModuleParameters;
    use mlx_rs::module::Param;
    use mlx_rs::{assert_array_eq, Dtype};

    use super::optim_test_util::*;
    use super::*;

    #[derive(Debug, ModuleParameters)]
    struct Model {
        #[param]
        a: Param<Array>,
    }

    // This unit test is adapted from the swift binding unit test `testAdaGrad` in
    #[test]
    fn test_adagrad() {
        mlx_rs::random::seed(958);
        let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
        assert_eq!(a.shape(), &[4, 3]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_array_eq!(a.mean(None, None).unwrap(), array!(-0.04584333300590515), ATOL);
        assert_array_eq!(a.sum(None, None).unwrap(), array!(-0.5501199960708618), ATOL);

        let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
        assert_eq!(a_grad.shape(), &[4, 3]);
        assert_eq!(a_grad.dtype(), Dtype::Float32);
        assert_array_eq!(a_grad.mean(None, None).unwrap(), array!(0.23250393569469452), ATOL);
        assert_array_eq!(a_grad.sum(None, None).unwrap(), array!(2.7900471687316895), ATOL);

        let mut a_model = Model { a: Param::new(a.clone()) };
        let mut a_grad_params = FlattenedModuleParam::new();
        a_grad_params.insert("a".into(), a_grad.clone());

        let mut optimizer = AdaGrad::new(0.1);

        optimizer.apply(&mut a_model, a_grad_params).unwrap();
        assert_eq!(a_model.a.shape(), &[4, 3]);
        assert_eq!(a_model.a.dtype(), Dtype::Float32);
        assert_array_eq!(a_model.a.mean(None, None).unwrap(), array!(-0.06250998377799988), ATOL);
        assert_array_eq!(a_model.a.sum(None, None).unwrap(), array!(-0.7501198053359985), ATOL);
    }
}