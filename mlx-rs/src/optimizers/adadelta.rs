use std::rc::Rc;

use crate::{array, ops::sqrt, utils::{get_mut_or_insert_with, Updatable}, Array};
use mlx_internal_macros::{generate_builder, Buildable};

use crate::error::AdaDeltaBuildError;

use super::*;

generate_builder! {
    /// The AdaDelta optimizer with a learning rate
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_adadelta,
        err = AdaDeltaBuildError,
        root = crate
    )]
    pub struct AdaDelta {
        /// The learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The coefficient used for computing a running average of squared gradients. Default to
        /// [`AdaDelta::DEFAULT_RHO`].
        #[builder(optional, ty_override = f32, default = AdaDelta::DEFAULT_RHO)]
        pub rho: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`AdaDelta::DEFAULT_EPS`].
        #[builder(optional, ty_override = f32, default = AdaDelta::DEFAULT_EPS)]
        pub eps: Array,

        /// Inner state
        #[builder(ignore)]
        pub state: OptimizerState<(Array, Array)>,
    }
}

/// Builds a new [`AdaDelta`] optimizer
fn build_adadelta(builder: AdaDeltaBuilder) -> Result<AdaDelta, AdaDeltaBuildError> {
    let rho = builder.rho;
    let eps = builder.eps;

    if rho < 0.0 {
        return Err(AdaDeltaBuildError::NegativeRho);
    }

    if eps < 0.0 {
        return Err(AdaDeltaBuildError::NegativeEps);
    }

    Ok(AdaDelta {
        lr: array!(builder.lr),
        rho: array!(rho),
        eps: array!(eps),
        state: OptimizerState::new(),
    })
}

impl AdaDelta {
    /// Default value for `rho`
    pub const DEFAULT_RHO: f32 = 0.99;

    /// Default value for `eps`
    pub const DEFAULT_EPS: f32 = 1e-6;
}

impl Optimizer for AdaDelta {
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> crate::error::Result<()> {
        let (v, u) = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let one_minus_rho = array!(1.0).subtract(&self.rho)?;
        let first_term = self.rho.multiply(&v)?;
        let second_term = one_minus_rho.multiply(gradient.square()?)?;
        let v_new = first_term.add(&second_term)?;

        let num = sqrt(&u.add(&self.eps)?)?;
        let den = sqrt(&v_new.add(&self.eps)?)?;
        let d = num.divide(&den)?.multiply(gradient)?;
        let first_term = self.rho.multiply(&u)?;
        let second_term = one_minus_rho.multiply(d.square()?)?;
        let u_new = first_term.add(&second_term)?;

        let param_new = parameter.subtract(self.lr.multiply(d)?)?;

        *parameter = param_new;

        *v = v_new;
        *u = u_new;

        Ok(())
    }
}

impl Updatable for AdaDelta {
    fn updatable_parameters(&self) -> Vec<&Array> {
        self.state.values().map(|(v, u)| [v, u]).flatten().collect()
    }
    
    fn updatable_parameters_mut(&mut self) -> Vec<&mut Array> {
        self.state.values_mut().map(|(v, u)| vec![v, u]).flatten().collect()
    }
}