use mlx_internal_macros::generate_builder;

use crate::{array, utils::get_mut_or_insert_with};

use super::*;

type AdamBetas = (f32, f32);

generate_builder! {
    /// The Adam optimizer.
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
    #[derive(Debug)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Adam {
        pub lr: Array,

        #[optional(ty = AdamBetas)]
        pub betas: (Array, Array),

        #[optional(ty = f32)]
        pub eps: Array,

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
        let (b1, b2) = &self.betas;
        let (m, v) = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let one_minus_b1 = array!(1.0).subtract(b1)?;
        let one_minus_b2 = array!(1.0).subtract(b2)?;

        *m = b1.multiply(&*m)?.add(&one_minus_b1.multiply(gradient)?)?;
        *v = b2
            .multiply(&*v)?
            .add(&one_minus_b2.multiply(gradient.square())?)?;

        *parameter =
            parameter.subtract(&self.lr.multiply(&m.divide(&v.sqrt().add(&self.eps)?)?)?)?;

        Ok(())
    }
}
