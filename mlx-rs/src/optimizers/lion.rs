use mlx_internal_macros::{generate_builder, Buildable};

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
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_lion,
        root = crate
    )]
    pub struct Lion {
        /// The learning rate.
        pub lr: f32,

        /// The coefficients used for computing running averages of the gradient and its square.
        /// Default to [`Lion::DEFAULT_BETAS`].
        #[builder(optional, ty_override = Betas, default = Lion::DEFAULT_BETAS)]
        pub betas: (Array, Array),

        /// The weight decay. Default to [`Lion::DEFAULT_WEIGHT_DECAY`].
        #[builder(optional, default = Lion::DEFAULT_WEIGHT_DECAY)]
        pub weight_decay: f32,

        /// Inner state.
        #[builder(ignore)]
        pub state: OptimizerState,
    }
}

fn build_lion(builder: LionBuilder) -> Result<Lion, std::convert::Infallible> {
    let lr = builder.lr;
    let betas = builder.betas;
    let weight_decay = builder.weight_decay;

    Ok(Lion {
        lr,
        betas: (array!(betas.0), array!(betas.1)),
        weight_decay,
        state: OptimizerState::new(),
    })
}

impl Lion {
    /// Default values for `betas`
    pub const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);

    /// Default value for `weight_decay`
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;
}

impl Optimizer for Lion {
    fn apply_single(
        &mut self,
        key: &std::rc::Rc<str>,
        gradient: &Array,
        mut parameter: impl std::ops::DerefMut<Target = Array>,
    ) -> Result<(), crate::error::Exception> {
        use crate::ops::sign;

        let (b1, b2) = &self.betas;
        let m = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let one_minus_b1 = array!(1.0).subtract(b1)?;
        let one_minus_b2 = array!(1.0).subtract(b2)?;

        let c = b1.multiply(&m)?.add(&one_minus_b1.multiply(gradient)?)?;
        *m = b2.multiply(&m)?.add(&one_minus_b2.multiply(gradient)?)?;

        if self.weight_decay > 0.0 {
            // SAFETY: These coeffs are all single-element arrays and won't panic.
            *parameter = array!(1.0 - self.lr * self.weight_decay) * &*parameter;
        }

        let lr = array!(self.lr);
        *parameter = parameter.subtract(lr.multiply(sign(&c)?)?)?;

        Ok(())
    }
}
