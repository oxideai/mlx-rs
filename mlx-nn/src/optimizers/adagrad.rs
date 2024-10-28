use std::rc::Rc;

use mlx_internal_macros::generate_builder;
use mlx_rs::{array, error::Exception, ops::square, Array};

use crate::utils::get_mut_or_insert_with;

use super::{Optimizer, OptimizerState};

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

        /// The epsilon added to the denominator to improve numerical stability. 
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
    fn update_single(&mut self, key: Rc<str>, gradient: Array, parameter: &mut Array) -> Result<(), Exception> {
        let state = get_mut_or_insert_with(&mut self.state, &key, || array!(0.0));

        let v = state.add(square(&gradient))?;

        let num = self.lr.multiply(&gradient)?;
        let den = v.sqrt().add(&self.eps)?;
        let new_param = parameter.subtract(num.divide(&den)?)?;

        *state = v;
        *parameter = new_param;

        Ok(())
    }
}