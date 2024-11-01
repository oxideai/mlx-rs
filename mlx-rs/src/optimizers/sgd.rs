use std::{borrow::Cow, rc::Rc};

use crate::{array, utils::get_mut_or_insert_with, Array};
use mlx_internal_macros::generate_builder;

use super::*;

generate_builder! {
    /// Stochastic gradient descent optimizer.
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Sgd {
        /// Learning rate
        pub lr: Array,

        /// Momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
        #[optional(ty = f32)]
        pub momentum: Array,

        /// Weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
        #[optional(ty = f32)]
        pub weight_decay: Array,

        /// Dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
        #[optional(ty = f32)]
        pub dampening: Array,

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
        let momentum = array!(self.momentum.unwrap_or(Sgd::DEFAULT_MOMENTUM));
        let weight_decay = array!(self.weight_decay.unwrap_or(Sgd::DEFAULT_WEIGHT_DECAY));
        let dampening = array!(self.dampening.unwrap_or(Sgd::DEFAULT_DAMPENING));
        let nesterov = self.nesterov.unwrap_or(Sgd::DEFAULT_NESTEROV);

        Sgd {
            lr: array!(lr),
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
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        // Using these ops explicitly to avoid potential trait resolving conflict when PartialOrd
        // is implemented for Array.
        use crate::ops::{gt, le, ne};

        let state = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let zero = array!(0.0);

        let mut gradient = Cow::Borrowed(gradient);

        // Apply weight decay
        if ne(&self.weight_decay, &zero)?.item::<bool>() {
            gradient = Cow::Owned(self.weight_decay.multiply(&*parameter)?.add(&*gradient)?);
        }

        // Apply momentum
        if le(&self.momentum, &zero)?.item::<bool>() {
            *parameter = parameter.subtract(self.lr.multiply(gradient)?)?;
            return Ok(());
        }

        let mut v = state.multiply(&self.momentum)?;
        if gt(&self.dampening, &zero)?.item::<bool>() {
            let one_minus_dampening = array!(1.0).subtract(&self.dampening)?;
            v = v.add(&one_minus_dampening.multiply(&gradient)?)?;
        } else {
            v = v.add(&gradient)?;
        }

        match self.nesterov {
            true => {
                let update = gradient.add(&self.momentum.multiply(&v)?)?;
                *parameter = parameter.subtract(self.lr.multiply(&update)?)?;
                *state = v;
            }
            false => {
                let update = &v;
                *parameter = parameter.subtract(self.lr.multiply(update)?)?;
                *state = v;
            }
        }

        Ok(())
    }
}
