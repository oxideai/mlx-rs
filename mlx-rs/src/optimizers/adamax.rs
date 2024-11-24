use std::rc::Rc;

use mlx_internal_macros::{generate_builder, Buildable};

use crate::{
    array, builder::Builder, error::Exception, ops::{abs, maximum}, utils::get_mut_or_insert_with, Array
};

use super::{Betas, Optimizer, OptimizerState};

generate_builder! {
    /// The Adamax optimizer, a variant of Adam based on the infinity norm [1].
    ///
    /// Our Adam implementation follows the original paper and omits the bias
    /// correction in the first and second moment estimates. In detail,
    ///
    /// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(manual_impl, root = crate)]
    pub struct Adamax {
        /// The learning rate.
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The beta coefficients
        #[builder(optional, ty_override = Betas, default = Adamax::DEFAULT_BETAS)]
        pub betas: (Array, Array),

        /// The epsilon added to the denominator to improve numerical stability.
        #[builder(optional, ty_override = f32, default = Adamax::DEFAULT_EPS)]
        pub eps: Array,

        /// Inner state.
        #[builder(ignore)]
        pub state: OptimizerState<(Array, Array)>,
    }
}

impl Builder<Adamax> for AdamaxBuilder {
    type Error = std::convert::Infallible;

    fn build(self) -> Result<Adamax, Self::Error> {
        let lr = self.lr;
        let betas = self.betas;
        let eps = self.eps;

        Ok(Adamax {
            lr: array!(lr),
            betas: (array!(betas.0), array!(betas.1)),
            eps: array!(eps),
            state: OptimizerState::new(),
        })
    }
}

impl Adamax {
    /// Default value for `betas`.
    pub const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);

    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-8;
}

impl Optimizer for Adamax {
    fn apply_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let (b1, b2) = &self.betas;
        let (m, v) = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let one_minus_b1 = array!(1.0).subtract(b1)?;
        let new_m = b1.multiply(&*m)?.add(&one_minus_b1.multiply(gradient)?)?;
        let new_v = maximum(b2.multiply(&*v)?, abs(gradient))?;

        let new_parameter =
            parameter.subtract(self.lr.multiply(&new_m)?.divide(&new_v.add(&self.eps)?)?)?;

        *m = new_m;
        *v = new_v;
        *parameter = new_parameter;

        Ok(())
    }
}
