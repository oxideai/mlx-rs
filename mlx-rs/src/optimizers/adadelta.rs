use std::{collections::HashMap, rc::Rc};

use mlx_internal_macros::generate_builder;
use crate::{array, ops::sqrt, Array};

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
    ) -> Result<(), Exception> {
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
