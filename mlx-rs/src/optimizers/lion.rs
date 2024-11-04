use mlx_internal_macros::generate_builder;

use crate::{array, utils::get_mut_or_insert_with, Array};

use super::{Betas, Optimizer, OptimizerState};

generate_builder! {
    /// The Lion optimizer [1].
    ///
    /// Since updates are computed through the sign operation, they tend to have larger norm than
    /// for other optimizers such as SGD and Adam. We recommend a learning rate that is 3-10x
    /// smaller than AdamW and a weight decay 3-10x larger than AdamW to maintain the strength `(lr
    /// * wd)`. Our Lion implementation follows the original paper. In detail,
    ///
    /// [1]: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv preprint
    ///     arXiv:2302.06675.
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Lion {
        /// The learning rate.
        pub lr: Array,

        /// The coefficients used for computing running averages of the gradient and its square.
        /// Default to [`Lion::DEFAULT_BETAS`].
        #[optional(ty = Betas)]
        pub betas: (Array, Array),

        /// The weight decay. Default to [`Lion::DEFAULT_WEIGHT_DECAY`].
        #[optional(ty = f32)]
        pub weight_decay: Array,

        /// Inner state.
        pub state: OptimizerState,
    }
}

impl LionBuilder {
    /// Builds a new [`Lion`] optimizer.
    pub fn build(self, lr: f32) -> Lion {
        let betas = self.betas.unwrap_or(Lion::DEFAULT_BETAS);
        let weight_decay = self.weight_decay.unwrap_or(Lion::DEFAULT_WEIGHT_DECAY);

        Lion {
            lr: array!(lr),
            betas: (array!(betas.0), array!(betas.1)),
            weight_decay: array!(weight_decay),
            state: OptimizerState::new(),
        }
    }
}

impl Lion {
    /// Default values for `betas`
    pub const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);

    /// Default value for `weight_decay`
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;

    /// Creates a new [`Lion`] optimizer with all optional parameters set to their default values.
    pub fn new(lr: f32) -> Lion {
        Self::builder().build(lr)
    }
}

impl Optimizer for Lion {
    fn apply_single(
        &mut self,
        key: &std::rc::Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), crate::error::Exception> {
        use crate::ops::sign;

        let (b1, b2) = &self.betas;
        let m = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let one_minus_b1 = array!(1.0).subtract(b1)?;
        let one_minus_b2 = array!(1.0).subtract(b2)?;

        let c = b1.multiply(&m)?.add(&one_minus_b1.multiply(gradient)?)?;
        *m = b2.multiply(&m)?.add(&one_minus_b2.multiply(gradient)?)?;

        if self.weight_decay.gt(array!(0.0))?.item() {
            // SAFETY: These coeffs are all single-element arrays and won't panic.
            *parameter = (array!(1.0) - &self.lr * &self.weight_decay) * &*parameter;
        }

        *parameter = parameter.subtract(self.lr.multiply(sign(&c))?)?;

        Ok(())
    }
}
