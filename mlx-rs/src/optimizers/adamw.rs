use mlx_internal_macros::generate_builder;

use crate::{array, utils::get_mut_or_insert_with, Array};

use super::{Optimizer, OptimizerState};

generate_builder! {
    /// The AdamW optimizer [1].
    ///
    /// Following the above convention, in contrast with [1], we do not use bias
    /// correction in the first and second moments for AdamW. We update the weights
    /// with a `weightDecay` lambda value:
    ///
    /// [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay regularization. ICLR 2019.
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct AdamW {
        /// The learning rate.
        pub lr: Array,

        /// The coefficients used for computing running averages of the gradient and its square.
        ///
        /// Default to [`AdamW::DEFAULT_BETAS`].
        #[optional(ty = super::Betas)]
        pub betas: (Array, Array),

        /// The epsilon added to the denominator to improve numerical stability.
        ///
        /// Default to [`AdamW::DEFAULT_EPS`].
        #[optional(ty = f32)]
        pub eps: Array,

        /// The weight decay
        ///
        /// Default to [`AdamW::DEFAULT_WEIGHT_DECAY`].
        #[optional(ty = f32)]
        pub weight_decay: Array,

        /// Inner state.
        pub state: OptimizerState<(Array, Array)>,
    }
}

impl AdamWBuilder {
    /// Builds a new [`AdamW`] optimizer.
    pub fn build(self, lr: f32) -> AdamW {
        let betas = self.betas.unwrap_or(AdamW::DEFAULT_BETAS);
        let eps = self.eps.unwrap_or(AdamW::DEFAULT_EPS);
        let weight_decay = self.weight_decay.unwrap_or(AdamW::DEFAULT_WEIGHT_DECAY);

        AdamW {
            lr: array!(lr),
            betas: (array!(betas.0), array!(betas.1)),
            eps: array!(eps),
            weight_decay: array!(weight_decay),
            state: OptimizerState::new(),
        }
    }
}

impl AdamW {
    /// Default value for `betas`.
    pub const DEFAULT_BETAS: (f32, f32) = super::Adam::DEFAULT_BETAS;

    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = super::Adam::DEFAULT_EPS;

    /// Default value for `weight_decay`.
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.01;

    /// Creates a new [`AdamW`] optimizer with all optional parameters set to their default values.
    pub fn new(lr: f32) -> AdamW {
        Self::builder().build(lr)
    }
}

impl Optimizer for AdamW {
    fn apply_single(
        &mut self,
        key: &std::rc::Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), crate::error::Exception> {
        let betas = &self.betas;
        let state = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        // SAFETY: These are all single-element arrays and won't panic.
        let one_minus_lr_wd = array!(1.0) - (&self.lr * &self.weight_decay);
        let decayed_parameter = &*parameter * &one_minus_lr_wd;

        let (new_parameter, new_states) = super::adam_apply_single(
            &self.lr,
            betas,
            &self.eps,
            gradient,
            &decayed_parameter,
            state,
        )?;

        *state = new_states;
        *parameter = new_parameter;

        Ok(())
    }
}
