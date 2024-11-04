use mlx_internal_macros::generate_builder;

use crate::{array, utils::get_mut_or_insert_with};

use super::*;

/// Type alias for betas in the Adam/AdamW/Adamax optimizer builders.
pub type Betas = (f32, f32); // The macro right now can't handle raw tuple types

generate_builder! {
    /// The Adam optimizer.
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
    #[derive(Debug)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Adam {
        /// The learning rate
        pub lr: Array,

        /// The coefficients used for computing running averages of the gradient and its square
        ///
        /// Default to [`Adam::DEFAULT_BETAS`]
        #[optional(ty = Betas)]
        pub betas: (Array, Array),

        /// The epsilon added to the denominator to improve numerical stability
        ///
        /// Default to [`Adam::DEFAULT_EPS`]
        #[optional(ty = f32)]
        pub eps: Array,

        /// Inner state
        pub state: OptimizerState<(Array, Array)>,
    }
}

impl AdamBuilder {
    /// Builds a new [`Adam`].
    pub fn build(self, lr: f32) -> Adam {
        let betas = self.betas.unwrap_or(Adam::DEFAULT_BETAS);
        let eps = array!(self.eps.unwrap_or(Adam::DEFAULT_EPS));

        Adam {
            lr: array!(lr),
            betas: (array!(betas.0), array!(betas.1)),
            eps,
            state: OptimizerState::new(),
        }
    }
}

impl Adam {
    /// Default values for `betas`
    pub const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);

    /// Default value for `eps`
    pub const DEFAULT_EPS: f32 = 1e-8;

    /// Creates a new Adam optimizer with all optional parameters set to their default values.
    pub fn new(lr: f32) -> Adam {
        Self::builder().build(lr)
    }
}

impl Optimizer for Adam {
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let betas = &self.betas;
        let state = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let (new_parameter, new_state) =
            adam_apply_single(&self.lr, betas, &self.eps, gradient, parameter, state)?;

        *state = new_state;
        *parameter = new_parameter;

        Ok(())
    }
}

// Returns (new_parameter, (new_m, new_v))
pub(super) fn adam_apply_single(
    lr: &Array,
    betas: &(Array, Array),
    eps: &Array,
    gradient: &Array,
    parameter: &Array,
    state: &(Array, Array),
) -> Result<(Array, (Array, Array)), Exception> {
    let (b1, b2) = betas;
    let (m, v) = state;

    let one_minus_b1 = array!(1.0).subtract(b1)?;
    let one_minus_b2 = array!(1.0).subtract(b2)?;

    let new_m = b1.multiply(m)?.add(&one_minus_b1.multiply(gradient)?)?;
    let new_v = b2
        .multiply(v)?
        .add(&one_minus_b2.multiply(gradient.square())?)?;

    let new_parameter =
        parameter.subtract(&lr.multiply(&new_m.divide(&new_v.sqrt().add(eps)?)?)?)?;

    Ok((new_parameter, (new_m, new_v)))
}
