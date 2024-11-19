use std::{borrow::Cow, rc::Rc};

use crate::{array, utils::get_mut_or_insert_with, Array};
use mlx_internal_macros::{generate_builder, Buildable};

use super::*;

generate_builder! {
    /// Stochastic gradient descent optimizer.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_sgd,
        root = crate
    )]
    pub struct Sgd {
        /// Learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// Momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
        #[builder(optional, ty_override = f32, default = Sgd::DEFAULT_MOMENTUM)]
        pub momentum: Array,

        /// Weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
        #[builder(optional, ty_override = f32, default = Sgd::DEFAULT_WEIGHT_DECAY)]
        pub weight_decay: Array,

        /// Dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
        #[builder(optional, ty_override = f32, default = Sgd::DEFAULT_DAMPENING)]
        pub dampening: Array,

        /// Enables nesterov momentum. Default to [`Sgd::DEFAULT_NESTEROV`] if not specified.
        #[builder(optional, ty_override = bool, default = Sgd::DEFAULT_NESTEROV)]
        pub nesterov: bool,

        /// Inner state
        #[builder(ignore)]
        pub state: OptimizerState,
    }
}

fn build_sgd(builder: SgdBuilder) -> Result<Sgd, std::convert::Infallible> {
    let momentum = array!(builder.momentum);
    let weight_decay = array!(builder.weight_decay);
    let dampening = array!(builder.dampening);
    let nesterov = builder.nesterov;

    Ok(Sgd {
        lr: array!(builder.lr),
        momentum,
        weight_decay,
        dampening,
        nesterov,
        state: OptimizerState::new(),
    })
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
}

impl Optimizer for Sgd {
    /// Apply SGD to a single parameter. Returns the updated parameter and the updated state.
    #[inline]
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> crate::error::Result<()> {
        // Using these ops explicitly to avoid potential trait resolving conflict when PartialOrd
        // is implemented for Array.
        use crate::ops::{gt, le, ne};

        let state = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let zero = array!(0.0);

        let mut gradient = OwnedOrRef::Ref(gradient);

        // Apply weight decay
        if ne(&self.weight_decay, &zero)?.item::<bool>() {
            gradient = OwnedOrRef::Owned(self.weight_decay.multiply(&*parameter)?.add(&*gradient)?);
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
