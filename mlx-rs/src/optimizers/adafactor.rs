use std::{borrow::Cow, collections::HashMap, rc::Rc};

use mlx_internal_macros::generate_builder;

use crate::{
    array,
    error::{AdafactorBuildError, Exception},
    ops::{matmul, maximum, mean, minimum, rsqrt, sqrt, square, zeros_dtype, zeros_like},
    Array,
};

use super::{Optimizer, OptimizerState};

fn rms(inputs: &Array) -> Result<Array, Exception> {
    Ok(sqrt(&mean(&square(inputs), None, None)?))
}

fn approvate_exp_moving_avg(
    exp_avg_sq_row: &Array,
    exp_avg_sq_col: &Array,
) -> Result<Array, Exception> {
    let rfactor = rsqrt(&exp_avg_sq_row.divide(&mean(exp_avg_sq_row, &[-1], true)?)?);
    let cfactor = rsqrt(exp_avg_sq_col);
    matmul(&rfactor.expand_dims(&[-1])?, &cfactor.expand_dims(&[0])?)
}

/// Type alias for the epsilon values used in Adafactor builder
pub type AdafactorEps = (f32, f32);

/// State of the Adafactor optimizer.
#[derive(Debug, Clone)]
pub struct AdafactorState {
    pub(crate) step: Array,
    pub(crate) exp_avg_sq_row: Option<Array>,
    pub(crate) exp_avg_sq_col: Option<Array>,
    pub(crate) exp_avg_sq: Option<Array>,
    pub(crate) exp_avg: Option<Array>,
}

impl AdafactorState {
    /// Default value for `step`
    pub const DEFAULT_STEP: i32 = 0;

    fn new(parameter: &Array, beta1_is_some: bool) -> Result<Self, Exception> {
        let step = array!(Self::DEFAULT_STEP);
        let mut exp_avg_sq_row = None;
        let mut exp_avg_sq_col = None;
        let mut exp_avg_sq = None;
        let mut exp_avg = None;

        if parameter.ndim() >= 2 {
            let shape = parameter.shape();
            let dtype = parameter.dtype();

            let row_shape = &shape[..shape.len() - 1];
            exp_avg_sq_row = Some(zeros_dtype(row_shape, dtype)?);

            let mut col_shape = shape[..shape.len() - 2].to_vec();
            col_shape.push(*shape.last().unwrap());
            exp_avg_sq_col = Some(zeros_dtype(&col_shape, dtype)?);
        } else {
            exp_avg_sq = Some(zeros_like(parameter)?);
        };

        if beta1_is_some {
            exp_avg = Some(zeros_like(parameter)?);
        }

        Ok(Self {
            step,
            exp_avg_sq_row,
            exp_avg_sq_col,
            exp_avg_sq,
            exp_avg,
        })
    }
}

/// `Option<Array>`. Type alias for the learning rate used in Adafactor builder due to limitation in
/// the `generate_builder` macro
pub type AdafactorBuilderLr = Option<f32>;

/// A thin wrapper around `Option<Array>`. This wrapper is added to prevent accidental change of
/// the learning rate after the optimizer is built while keeping all fields public.
#[derive(Debug, Clone)]
pub struct AdafactorLr(Option<Array>);

impl AdafactorLr {
    /// Returns the inner value.
    pub fn value(&self) -> &Option<Array> {
        &self.0
    }
}

impl From<Option<Array>> for AdafactorLr {
    fn from(lr: Option<Array>) -> Self {
        Self(lr)
    }
}

/// `Option<f32>` Type alias for the beta1 used in Adafactor builder due to limitation in the
/// `generate_builder` macro
pub type AdafactorBuilderBeta1 = Option<f32>;

/// A thin wrapper around `Option<f32>`. This wrapper is added to prevent accidental change of
/// the beta1 after the optimizer is built while keeping all fields public.
#[derive(Debug, Clone)]
pub struct AdafactorBeta1(Option<Array>);

impl AdafactorBeta1 {
    /// Returns the inner value.
    pub fn value(&self) -> &Option<Array> {
        &self.0
    }
}

impl From<Option<Array>> for AdafactorBeta1 {
    fn from(beta1: Option<Array>) -> Self {
        Self(beta1)
    }
}

/// A thin wrapper around `bool`. This wrapper is added to prevent accidental change of
/// the beta1 after the optimizer is built while keeping all fields public.
#[derive(Debug, Clone)]
pub struct AdafactorRelativeStep(bool);

impl AdafactorRelativeStep {
    /// Returns the inner value.
    pub fn value(&self) -> bool {
        self.0
    }
}

impl From<bool> for AdafactorRelativeStep {
    fn from(relative_step: bool) -> Self {
        Self(relative_step)
    }
}

generate_builder! {
    /// The Adafactor optimizer.
    ///
    /// Our Adafactor implementation follows the original paper: `Adafactor:
    /// Adaptive Learning Rates with Sublinear Memory Cost
    /// <https://arxiv.org/abs/1804.04235>
    #[derive(Debug, Clone)]
    #[generate_builder(generate_build_fn = false)]
    pub struct Adafactor {
        /// The learning rate.
        #[optional(skip = true, ty = AdafactorBuilderLr)]
        pub lr: AdafactorLr,

        /// The first term is added to the square of the gradients to improve numerical stability.
        /// Default to [`Adafactor::DEFAULT_EPS`].
        #[optional(ty = AdafactorEps)]
        pub eps: (Array, Array),

        /// Clips the unscaled update. Default to [`Adafactor::DEFAULT_CLIP_THRESHOLD`].
        #[optional(ty = f32)]
        pub clip_threshold: Array,

        /// Coefficient for the running average of the squared gradient. Default to
        /// [`Adafactor::DEFAULT_DECAY_RATE`].
        #[optional(ty = f32)]
        pub decay_rate: Array,

        /// If set then the first moment will be used.
        #[optional(skip = true, ty = AdafactorBuilderBeta1)]
        pub beta1: AdafactorBeta1,

        /// The weight decay. Default to [`Adafactor::DEFAULT_WEIGHT_DECAY`].
        #[optional(ty = f32)]
        pub weight_decay: Array,

        /// If `true` the `learningRate` will be scaled by `max(eps.0, RMS(parameter))`. Default to
        /// [`Adafactor::DEFAULT_SCALE_PARAMETER`].
        #[optional]
        pub scale_parameter: bool,

        /// If `true` the `learningRate` will be ignored and the relative step size will be
        /// computed. Default to [`Adafactor::DEFAULT_RELATIVE_STEP`].
        #[optional(ty = bool)]
        pub relative_step: AdafactorRelativeStep,

        /// If `true` the relative step size will be calculated by the current step. Default to
        /// [`Adafactor::DEFAULT_WARMUP_INIT`].
        #[optional]
        pub warmup_init: bool,

        /// Inner state.
        pub state: OptimizerState<AdafactorState>,
    }
}

impl AdafactorBuilder {
    /// Sets the `lr` field.
    pub fn lr(mut self, lr: impl Into<Option<f32>>) -> Self {
        self.lr = lr.into();
        self
    }

    /// Sets the `beta1` field.
    pub fn beta1(mut self, beta1: impl Into<Option<f32>>) -> Self {
        self.beta1 = beta1.into();
        self
    }

    /// Builds a new [`Adafactor`] optimizer.
    pub fn build(self) -> Result<Adafactor, AdafactorBuildError> {
        let eps = self.eps.unwrap_or(Adafactor::DEFAULT_EPS);
        let clip_threshold = self
            .clip_threshold
            .unwrap_or(Adafactor::DEFAULT_CLIP_THRESHOLD);
        let decay_rate = self.decay_rate.unwrap_or(Adafactor::DEFAULT_DECAY_RATE);
        let weight_decay = self.weight_decay.unwrap_or(Adafactor::DEFAULT_WEIGHT_DECAY);
        let scale_parameter = self
            .scale_parameter
            .unwrap_or(Adafactor::DEFAULT_SCALE_PARAMETER);
        let relative_step: AdafactorRelativeStep = self
            .relative_step
            .unwrap_or(Adafactor::DEFAULT_RELATIVE_STEP)
            .into();
        let warmup_init = self.warmup_init.unwrap_or(Adafactor::DEFAULT_WARMUP_INIT);

        if self.lr.is_none() && !relative_step.value() {
            return Err(AdafactorBuildError::LrIsNoneAndRelativeStepIsFalse);
        }

        Ok(Adafactor {
            lr: self.lr.map(Array::from).into(),
            eps: (array!(eps.0), array!(eps.1)),
            clip_threshold: array!(clip_threshold),
            decay_rate: array!(decay_rate),
            beta1: self.beta1.map(Array::from).into(),
            weight_decay: array!(weight_decay),
            scale_parameter,
            relative_step,
            warmup_init,
            state: OptimizerState::new(),
        })
    }
}

impl Adafactor {
    /// Default values for `eps`
    pub const DEFAULT_EPS: (f32, f32) = (1e-30, 1e-3);

    /// Default value for `clip_threshold`
    pub const DEFAULT_CLIP_THRESHOLD: f32 = 1.0;

    /// Default value for `decay_rate`
    pub const DEFAULT_DECAY_RATE: f32 = -0.8;

    /// Default value for `weight_decay`
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;

    /// Default value for `scale_parameter`
    pub const DEFAULT_SCALE_PARAMETER: bool = true;

    /// Default value for `relative_step`
    pub const DEFAULT_RELATIVE_STEP: bool = true;

    /// Default value for `warmup_init`
    pub const DEFAULT_WARMUP_INIT: bool = false;

    /// Creates a new [`Adafactor`] optimizer with the default values.
    pub fn new() -> Adafactor {
        Self::builder().build().unwrap()
    }
}

impl Default for Adafactor {
    fn default() -> Self {
        Self::new()
    }
}

fn get_mut_or_insert_with<'a, T, E>(
    map: &'a mut HashMap<Rc<str>, T>,
    key: &Rc<str>,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<&'a mut T, E> {
    if !map.contains_key(key) {
        map.insert(key.clone(), f()?);
    }

    Ok(map.get_mut(key).unwrap())
}

fn compute_lr(
    relative_step: bool,
    warmup_init: bool,
    lr: &Option<Array>,
    scale_parameter: bool,
    eps: &(Array, Array),
    step: &Array,
    parameter_rms: &Array,
) -> Result<Array, Exception> {
    let relative_step_size = if relative_step {
        let min_step = if warmup_init {
            // SAFETY: `step` is a single-element array and won't panic.
            array!(1e-6) * step
        } else {
            array!(1e-2)
        };
        // SAFETY: `step` is a single-element array and won't panic.
        Cow::Owned(minimum(min_step, array!(1.0) / sqrt(step))?)
    } else {
        // SAFETY: This is already checked in the `build` stage.
        Cow::Borrowed(
            lr.as_ref()
                .expect("The learning rate should be set if the relative step is not enabled"),
        )
    };

    let mut parameter_scale = array!(1.0);
    if scale_parameter {
        parameter_scale = maximum(&eps.1, parameter_rms)?;
    }

    parameter_scale.multiply(&*relative_step_size)
}

impl Optimizer for Adafactor {
    fn apply_single(
        &mut self,
        key: &std::rc::Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), Exception> {
        let beta1_is_some = self.beta1.value().is_some();
        let state = get_mut_or_insert_with(&mut self.state, key, || {
            AdafactorState::new(parameter, beta1_is_some)
        })?;

        state.step = state.step.add(array!(1))?;

        let gradient_shape = gradient.shape();
        let factored = gradient_shape.len() >= 2;
        let step = &state.step;

        let parameter_rms = rms(parameter)?;
        let lr = compute_lr(
            self.relative_step.value(),
            self.warmup_init,
            self.lr.value(),
            self.scale_parameter,
            &self.eps,
            step,
            &parameter_rms,
        )?;
        let beta2 = array!(1.0).subtract(&step.power(&self.decay_rate)?)?;

        let mut update = gradient.square().add(&self.eps.0)?;

        let one_minus_beta2 = array!(1.0).subtract(&beta2)?;
        if factored {
            // SAFETY: These fields are created in the `new` when ndim >= 2 and won't panic.
            let exp_avg_sq_row = state.exp_avg_sq_row.as_mut().unwrap();
            let exp_avg_sq_col = state.exp_avg_sq_col.as_mut().unwrap();

            *exp_avg_sq_row = beta2
                .multiply(&*exp_avg_sq_row)?
                .add(&one_minus_beta2.multiply(&update.mean(&[-1], None)?)?)?;
            *exp_avg_sq_col = beta2
                .multiply(&*exp_avg_sq_col)?
                .add(&one_minus_beta2.multiply(&update.mean(&[-2], None)?)?)?;

            update = approvate_exp_moving_avg(&*exp_avg_sq_row, &*exp_avg_sq_col)?;
            update = update.multiply(gradient)?;
        } else {
            // SAFETY: This field is created in the `new` when ndim < 2 and won't panic.
            let exp_avg_sq = state.exp_avg_sq.as_mut().unwrap();

            *exp_avg_sq = beta2
                .multiply(&*exp_avg_sq)?
                .add(&one_minus_beta2.multiply(&update)?)?;
            update = rsqrt(&*exp_avg_sq).multiply(gradient)?;
        }

        let update_rms = rms(&update)?;
        let max = maximum(array!(1.0), update_rms.divide(&self.clip_threshold)?)?;
        update = update.divide(max)?;
        update = lr.multiply(update)?;

        if let Some(beta1) = self.beta1.value() {
            // SAFETY: This field is created in the `new` when beta1 is set and won't panic.
            let exp_avg = state.exp_avg.as_mut().unwrap();
            let one_minus_beta1 = array!(1.0).subtract(beta1)?;
            *exp_avg = beta1
                .multiply(&*exp_avg)?
                .add(&one_minus_beta1.multiply(&update)?)?;
            update = exp_avg.clone();
        }

        if self.weight_decay.ne(&array!(0.0))?.item() {
            let rhs = parameter.multiply(self.weight_decay.negative()?.multiply(&lr)?)?;
            *parameter = parameter.add(rhs)?;
        }

        *parameter = parameter.subtract(&update)?;

        Ok(())
    }
}
