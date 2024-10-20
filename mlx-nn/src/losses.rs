//! Loss functions

use mlx_macros::GenerateBuilder;
use mlx_rs::{
    array,
    error::Exception,
    ops::{
        abs, clip, exp, indexing::take_along_axis, log, log_add_exp, log_sum_exp, maximum, minimum,
        multiply, power, r#where, sqrt, square, sum,
    },
    Array,
};

use crate::error::CrossEntropyBuildError;

#[inline]
fn check_shape(
    left: &Array,
    right: &Array,
    left_ident: &str,
    right_ident: &str,
) -> Result<(), Exception> {
    if left.shape() != right.shape() {
        return Err(Exception::custom(format!(
            "The shape of the {} ({:?}) does not match the shape of the {} ({:?})",
            left_ident,
            left.shape(),
            right_ident,
            right.shape()
        )));
    }
    Ok(())
}

/// Different types of loss reductions
#[derive(Debug, Clone, Copy)]
pub enum LossReduction {
    /// No reduction is applied.
    None,
    /// The sum of the output will be computed.
    Sum,
    /// The mean of the output will be computed.
    Mean,
}

impl LossReduction {
    /// Reduces the loss according to the reduction type.
    pub fn reduce(&self, loss: Array) -> Result<Array, Exception> {
        match self {
            LossReduction::None => Ok(loss),
            LossReduction::Sum => Ok(loss.sum(None, None)?),
            LossReduction::Mean => Ok(loss.mean(None, None)?),
        }
    }
}

/// Cross entropy loss function.
#[derive(Debug, Clone, GenerateBuilder)]
#[generate_builder(generate_build_fn = false)]
pub struct CrossEntropy<'a> {
    /// Weights for each target
    #[optional(skip = true)]
    pub weights: Option<&'a Array>,

    /// The axis over which to compute softmax. Default to [`CrossEntropy::DEFAULT_AXIS`]
    #[optional(default_value = CrossEntropy::DEFAULT_AXIS)]
    pub axis: i32,

    /// The label smoothing factor, range [0, 1). Default to
    /// [`CrossEntropy::DEFAULT_LABEL_SMOOTHING`]
    #[optional(default_value = CrossEntropy::DEFAULT_LABEL_SMOOTHING)]
    pub label_smoothing: f32,

    /// Reduction type. Default to [`CrossEntropy::DEFAULT_REDUCTION`]
    #[optional(default_value = CrossEntropy::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl<'a> CrossEntropyBuilder<'a> {
    /// Sets the `weight` field.
    pub fn weights(mut self, weights: impl Into<Option<&'a Array>>) -> Self {
        self.weights = weights.into();
        self
    }

    /// Build a [`CrossEntropy`] loss function.
    pub fn build(self) -> Result<CrossEntropy<'a>, CrossEntropyBuildError> {
        let axis = self.axis.unwrap_or(CrossEntropy::DEFAULT_AXIS);
        let label_smoothing = self
            .label_smoothing
            .unwrap_or(CrossEntropy::DEFAULT_LABEL_SMOOTHING);
        let reduction = self.reduction.unwrap_or(CrossEntropy::DEFAULT_REDUCTION);

        if !(0.0..1.0).contains(&label_smoothing) {
            return Err(CrossEntropyBuildError::InvalidLabelSmoothingFactor);
        }

        Ok(CrossEntropy {
            weights: self.weights,
            axis,
            label_smoothing,
            reduction,
        })
    }
}

impl<'a> Default for CrossEntropy<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CrossEntropy<'a> {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `label_smoothing` parameter.
    pub const DEFAULT_LABEL_SMOOTHING: f32 = 0.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Create a new [`CrossEntropy`] with all optional parameters set to their default values.
    pub fn new() -> Self {
        Self::builder().build().expect("Default values are valid")
    }

    /// Apply the cross entropy loss function on the given logits and targets.
    ///
    /// # Params
    ///
    /// - `logits`: unnormalized predicted logits
    /// - `targets`: target values, as class indices
    pub fn apply(
        &self,
        logits: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let logits = logits.as_ref();
        let targets = targets.as_ref();

        let target_as_probs = targets.ndim() == logits.ndim();

        let score = if target_as_probs {
            sum(&logits.multiply(targets)?, &[self.axis], None)?
        } else {
            take_along_axis(logits, &targets.expand_dims(&[-1])?, self.axis)?.squeeze(&[-1])?
        };
        let log_sum_exp_logits = log_sum_exp(logits, &[self.axis], None)?;

        let mut loss = if self.label_smoothing > 0.0 {
            // adjust the true class score with label smoothing
            let adjusted_score = multiply(array!(1.0 - self.label_smoothing), score)?;

            // calculate the mean logit across the classes for smoothed loss
            let mean_logits = logits.mean(&[self.axis], None)?;
            let smoothed_loss = -multiply(mean_logits, array!(self.label_smoothing))?;

            // combine the adjusted score and smoothed loss with the logsumexp logits
            log_sum_exp_logits
                .subtract(adjusted_score)?
                .add(smoothed_loss)?
        } else {
            log_sum_exp_logits.subtract(score)?
        };

        if let Some(weights) = self.weights {
            check_shape(weights, &loss, "weights", "loss")?;
            loss = multiply(loss, weights)?;
        }

        self.reduction.reduce(loss)
    }
}

/// Binary cross entropy loss.
///
/// By default, this function takes the pre-sigmoid logits, which results in a faster
/// and more precise loss. For improved numerical stability when `inputs_are_logits` is true,
/// the loss calculation clips the input probabilities (in log-space) to a minimum value
/// of `-100`.
#[derive(Debug, Clone, GenerateBuilder)]
#[generate_builder(generate_build_fn = false)]
pub struct BinaryCrossEntropy<'a> {
    /// Optional weights for each target
    #[optional(skip = true)]
    pub weights: Option<&'a Array>,

    /// Whether the inputs are logits. Default to
    /// [`BinaryCrossEntropy::DEFAULT_INPUTS_ARE_LOGITS`]
    #[optional]
    pub inputs_are_logits: bool,

    /// Reduction type. Default to [`BinaryCrossEntropy::DEFAULT_REDUCTION`]
    #[optional]
    pub reduction: LossReduction,
}

impl<'a> BinaryCrossEntropyBuilder<'a> {
    /// Sets the `weights` field.
    pub fn weights(mut self, weights: impl Into<Option<&'a Array>>) -> Self {
        self.weights = weights.into();
        self
    }

    /// Build a [`BinaryCrossEntropy`] loss function.
    pub fn build(self) -> BinaryCrossEntropy<'a> {
        BinaryCrossEntropy {
            weights: self.weights,
            inputs_are_logits: self
                .inputs_are_logits
                .unwrap_or(BinaryCrossEntropy::DEFAULT_INPUTS_ARE_LOGITS),
            reduction: self
                .reduction
                .unwrap_or(BinaryCrossEntropy::DEFAULT_REDUCTION),
        }
    }
}

impl<'a> Default for BinaryCrossEntropy<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> BinaryCrossEntropy<'a> {
    /// Default value for the `with_logits` parameter.
    pub const DEFAULT_INPUTS_ARE_LOGITS: bool = true;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Create a new [`BinaryCrossEntropy`] with all optional parameters set to their default values.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Apply the binary cross entropy loss function on the given logits and targets.
    ///
    /// # Params
    ///
    /// - `logits`: unnormalized predicted logits
    /// - `targets`: binary target values in {0, 1}
    pub fn apply(
        &self,
        logits: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let logits = logits.as_ref();
        let targets = targets.as_ref();
        let weights = self.weights;
        let inputs_are_logits = self.inputs_are_logits;
        let reduction = self.reduction;

        let mut loss = if inputs_are_logits {
            log_add_exp(array!(0.0), logits)?.subtract(targets.multiply(logits)?)?
        } else {
            let log_inputs_clip = clip(&log(logits), (-100.0, ()))?;
            let log_inputs_inverse_clip = clip(&log(&array!(1.0).subtract(logits)?), (-100.0, ()))?;
            -(targets.multiply(log_inputs_clip)?.add(
                array!(1.0)
                    .subtract(targets)?
                    .multiply(log_inputs_inverse_clip)?,
            )?)
        };

        if let Some(weights) = weights {
            check_shape(weights, &loss, "weights", "loss")?;
            loss = multiply(loss, weights)?;
        }

        reduction.reduce(loss)
    }
}

/// Computes the L1 loss
#[derive(Debug, Clone, GenerateBuilder)]
pub struct L1Loss {
    /// Reduction type. Default to [`L1loss::DEFAULT_REDUCTION`]
    #[optional(default_value = L1Loss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl L1Loss {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;

    /// Compute the L1 loss.
    ///
    /// # Params
    ///
    /// - `predictions`: predicted values
    /// - `targets`: target values
    pub fn apply(
        &self,
        predictions: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let predictions = predictions.as_ref();
        let targets = targets.as_ref();
        let reduction = self.reduction;

        check_shape(predictions, targets, "predictions", "targets")?;
        let loss = predictions.subtract(targets)?.abs();
        reduction.reduce(loss)
    }
}

/// Computes the mean squared error loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct MseLoss {
    /// Reduction type. Default to [`MseLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = MseLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MseLoss {
    /// Default value for the reduction parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;

    /// Compute the mean squared error loss.
    ///
    /// # Params
    ///
    /// - `predictions`: predicted values
    /// - `targets`: target values
    pub fn apply(
        &self,
        predictions: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let predictions = predictions.as_ref();
        let targets = targets.as_ref();
        let reduction = self.reduction;

        check_shape(predictions, targets, "predictions", "targets")?;
        let loss = predictions.subtract(targets)?.square();
        reduction.reduce(loss)
    }
}

/// Computes the negative log likelihood loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct NllLoss {
    /// distribution axis. Default to [`NllLoss::DEFAULT_AXIS`]
    #[optional(default_value = NllLoss::DEFAULT_AXIS)]
    pub axis: i32,

    /// Reduction type. Default to [`NllLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = NllLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for NllLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl NllLoss {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Compute the negative log likelihood loss.
    ///
    /// # Params
    ///
    /// - `inputs`: predicted distribution in log space
    /// - `targets`: target values
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let axis = self.axis;
        let reduction = self.reduction;

        let loss = -take_along_axis(inputs, &targets.expand_dims(&[-1])?, axis)?.squeeze(&[-1])?;
        reduction.reduce(loss)
    }
}

/// Compute the negative log likelihood loss for a Gaussian distribution.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct GaussianNllLoss {
    /// Whether to include the constant term in the loss calculation. Default to
    /// [`GaussianNllLoss::DEFAULT_FULL`]
    #[optional(default_value = GaussianNllLoss::DEFAULT_FULL)]
    pub full: bool,

    /// Small positive constant for numerical stability. Default to
    /// [`GaussianNllLoss::DEFAULT_EPS`]
    #[optional(default_value = GaussianNllLoss::DEFAULT_EPS)]
    pub eps: f32,

    /// Reduction type. Default to [`GaussianNllLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = GaussianNllLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for GaussianNllLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianNllLoss {
    /// Default value for the `full` parameter.
    pub const DEFAULT_FULL: bool = false;

    /// Default value for the `eps` parameter.
    pub const DEFAULT_EPS: f32 = 1e-6;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Compute the negative log likelihood loss for a Gaussian distribution.
    ///
    /// # Params
    ///
    /// - `inputs`: The predicted expectation of the Gaussian distribution.
    /// - `targets`: The target values (samples from the Gaussian distribution).
    /// - `vars`: The predicted variance of the Gaussian distribution.
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
        vars: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let vars = vars.as_ref();
        let full = self.full;
        let eps = self.eps;
        let reduction = self.reduction;

        check_shape(inputs, targets, "inputs", "targets")?;
        check_shape(inputs, vars, "inputs", "vars")?;

        let vars = maximum(vars, array!(eps))?;
        let mut loss =
            array!(0.5) * (log(&vars).add(square(&targets.subtract(inputs)?).divide(&vars)?)?);

        if full {
            let pi = array!(std::f32::consts::PI);
            loss = loss.add(array!(0.5).multiply(log(&array!(2.0).multiply(pi)?))?)?;
        }

        reduction.reduce(loss)
    }
}

/// Compute the Kullback-Leibler divergence loss.
///
/// Computes the following when the `reduction` is `LossReduction::None`:
///
/// ```rust, ignore
/// sum(exp(targets) * (targets - inputs), axis, None)
/// ```
#[derive(Debug, Clone, GenerateBuilder)]
pub struct KlDivLoss {
    /// The distribution axis. Default to [`KlDivLoss::DEFAULT_AXIS`]
    #[optional(default_value = KlDivLoss::DEFAULT_AXIS)]
    pub axis: i32,

    /// Reduction type. Default to [`KlDivLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = KlDivLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for KlDivLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl KlDivLoss {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Compute the Kullback-Leibler divergence loss.
    ///
    /// # Params
    ///
    /// - `inputs`: Log probabilities for the predicted distribution.
    /// - `targets`: Log probabilities for the target distribution.
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let axis = self.axis;
        let reduction = self.reduction;

        let loss = sum(
            &exp(targets).multiply(targets.subtract(inputs)?)?,
            &[axis],
            None,
        )?;
        reduction.reduce(loss)
    }
}

/// Computes the smooth L1 loss.
///
/// The smooth L1 loss is a variant of the L1 loss which replaces the absolute
/// difference with a squared difference when the absolute difference is less
/// than `beta`.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct SmoothL1Loss {
    /// The threshold after which the loss changes from the squared to the absolute difference.
    /// Default to [`SmoothL1Loss::DEFAULT_BETA`]
    #[optional(default_value = SmoothL1Loss::DEFAULT_BETA)]
    pub beta: f32,

    /// Reduction type. Default to [`SmoothL1Loss::DEFAULT_REDUCTION`]
    #[optional(default_value = SmoothL1Loss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl SmoothL1Loss {
    /// Default value for the `beta` parameter.
    pub const DEFAULT_BETA: f32 = 1.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;

    /// Compute the smooth L1 loss.
    ///
    /// # Params
    ///
    /// - `predictions`: predicted values
    /// - `targets`: target values
    pub fn apply(
        &self,
        predictions: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let predictions = predictions.as_ref();
        let targets = targets.as_ref();
        let beta = self.beta;
        let reduction = self.reduction;

        check_shape(predictions, targets, "predictions", "targets")?;
        let diff = predictions.subtract(targets)?;
        let loss = r#where(
            &diff.lt(array!(beta))?,
            array!(0.5).multiply(square(&diff))?.divide(&array!(beta))?,
            abs(&diff).subtract(array!(0.5).multiply(array!(beta))?)?,
        )?;
        reduction.reduce(loss)
    }
}

/// Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is
/// represented with alpha in the math section.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct TripletLoss {
    /// Distribution axis. Default to [`TripletLoss::DEFAULT_AXIS`]
    #[optional(default_value = TripletLoss::DEFAULT_AXIS)]
    pub axis: i32,

    /// The norm degree for pairwise distance. Default to [`TripletLoss::DEFAULT_P`]
    #[optional(default_value = TripletLoss::DEFAULT_P)]
    pub p: f32,

    /// Margin for the triplet loss. Default to [`TripletLoss::DEFAULT_MARGIN`]
    #[optional(default_value = TripletLoss::DEFAULT_MARGIN)]
    pub margin: f32,

    /// Small positive constant for numerical stability. Default to [`TripletLoss::DEFAULT_EPS`]
    #[optional(default_value = TripletLoss::DEFAULT_EPS)]
    pub eps: f32,

    /// Reduction type. Default to [`TripletLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = TripletLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for TripletLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TripletLoss {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `p` parameter.
    pub const DEFAULT_P: f32 = 2.0;

    /// Default value for the `margin` parameter.
    pub const DEFAULT_MARGIN: f32 = 1.0;

    /// Default value for the `eps` parameter.
    pub const DEFAULT_EPS: f32 = 1e-6;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is
    /// represented with alpha in the math section.
    ///
    /// # Params
    ///
    /// - `anchors`: The anchor samples
    /// - `positives`: The positive samples
    /// - `neonatives`: The negative samples
    pub fn apply(
        &self,
        anchors: impl AsRef<Array>,
        positives: impl AsRef<Array>,
        negatives: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let anchors = anchors.as_ref();
        let positives = positives.as_ref();
        let negatives = negatives.as_ref();
        let axis = self.axis;
        let p = self.p;
        let margin = self.margin;
        let eps = self.eps;
        let reduction = self.reduction;

        let eps = array!(eps);
        let p = array!(p);
        let margin = array!(margin);

        let pos = sqrt(
            &power(&anchors.subtract(positives)?, &p)?
                .sum(&[axis], None)?
                .add(&eps)?,
        );
        let neg = sqrt(
            &power(&anchors.subtract(negatives)?, &p)?
                .sum(&[axis], None)?
                .add(&eps)?,
        );
        let loss = maximum(pos.subtract(neg)?.add(margin)?, array!(0.0))?;
        reduction.reduce(loss)
    }
}

/// Compute the hinge loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct HingeLoss {
    /// Reduction type. Default to [`HingeLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = HingeLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl HingeLoss {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Compute the hinge loss.
    ///
    /// # Params
    ///
    /// - `inputs`: predicted values
    /// - `targets`: target values, -1 or 1
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let reduction = self.reduction;

        let a = array!(1.0).subtract(inputs.multiply(targets)?)?;
        let b = array!(0.0);
        let loss = maximum(a, b)?;
        reduction.reduce(loss)
    }
}

/// Compute the Huber loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct HuberLoss {
    /// The threshold at which to change between L1 and L2 loss. Default to
    /// [`HuberLoss::DEFAULT_DELTA`]
    #[optional(default_value = HuberLoss::DEFAULT_DELTA)]
    pub delta: f32,

    /// Reduction type. Default to [`HuberLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = HuberLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl HuberLoss {
    /// Default value for the `delta` parameter.
    pub const DEFAULT_DELTA: f32 = 1.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Compute the Huber loss.
    ///
    /// # Params
    ///
    /// - `inputs`: predicted values
    /// - `targets`: target values
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let delta = self.delta;
        let reduction = self.reduction;

        let errors = inputs.subtract(targets)?;
        let abs_errors = errors.abs();
        let quadratic = minimum(&abs_errors, array!(delta))?;
        let linear = abs_errors.subtract(&quadratic)?;
        let loss = array!(0.5)
            .multiply(square(&quadratic))?
            .add(array!(delta).multiply(linear)?)?;
        reduction.reduce(loss)
    }
}

/// Computes the log cosh loss between inputs and targets.
///
/// Logcosh acts like L2 loss for small errors, ensuring stable gradients,
/// and like the L1 loss for large errors, reducing sensitivity to outliers. This
/// dual behavior offers a balanced, robust approach for regression tasks.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct LogCoshLoss {
    /// Reduction type. Default to [`LogCoshLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = LogCoshLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for LogCoshLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LogCoshLoss {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Computes the log cosh loss between inputs and targets.
    ///
    /// # Params
    ///
    /// - `inputs`: predicted values
    /// - `targets`: target values
    pub fn apply(
        &self,
        inputs: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs = inputs.as_ref();
        let targets = targets.as_ref();
        let reduction = self.reduction;

        let errors = inputs.subtract(targets)?;
        let neg_errors = errors.negative()?;
        let loss = log_add_exp(errors, neg_errors)?.subtract(log(&array!(2.0)))?;
        reduction.reduce(loss)
    }
}

/// Computes the cosine similarity loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct CosineSimilarityLoss {
    /// Embedding axis. Default to [`CosineSimilarityLoss::DEFAULT_AXIS`]
    #[optional(default_value = CosineSimilarityLoss::DEFAULT_AXIS)]
    pub axis: i32,

    /// minimum value of the denominator used for numerical stability. Default to
    /// [`CosineSimilarityLoss::DEFAULT_EPS`]
    #[optional(default_value = CosineSimilarityLoss::DEFAULT_EPS)]
    pub eps: f32,

    /// Reduction type. Default to [`CosineSimilarityLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = CosineSimilarityLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for CosineSimilarityLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CosineSimilarityLoss {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `eps` parameter.
    pub const DEFAULT_EPS: f32 = 1e-8;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Computes the cosine similarity loss.
    ///
    /// # Params
    ///
    /// - `x1`: first array
    /// - `x2`: second array
    pub fn apply(&self, x1: impl AsRef<Array>, x2: impl AsRef<Array>) -> Result<Array, Exception> {
        let x1 = x1.as_ref();
        let x2 = x2.as_ref();
        let axis = self.axis;
        let eps = self.eps;
        let reduction = self.reduction;

        fn l2_loss(a: &Array, axis: i32) -> Result<Array, Exception> {
            if a.dtype().is_complex() {
                Ok(sqrt(&sum(&abs(a).square(), &[axis], None)?))
            } else {
                Ok(sqrt(&sum(&a.square(), &[axis], None)?))
            }
        }

        let x1_norm = l2_loss(x1, axis)?;
        let x2_norm = l2_loss(x2, axis)?;

        let num = sum(&x1.multiply(x2)?, &[axis], None)?;
        let den = maximum(x1_norm.multiply(x2_norm)?, array!(eps))?;
        let loss = num.divide(&den)?;

        reduction.reduce(loss)
    }
}

/// Computes the margin ranking loss.
#[derive(Debug, Clone, GenerateBuilder)]
pub struct MarginRankingLoss {
    /// The margin by which the scores should be separated. Default to
    /// [`MarginRankingLoss::DEFAULT_MARGIN`]
    #[optional(default_value = MarginRankingLoss::DEFAULT_MARGIN)]
    pub margin: f32,

    /// Reduction type. Default to [`MarginRankingLoss::DEFAULT_REDUCTION`]
    #[optional(default_value = MarginRankingLoss::DEFAULT_REDUCTION)]
    pub reduction: LossReduction,
}

impl Default for MarginRankingLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MarginRankingLoss {
    /// Default value for the `margin` parameter.
    pub const DEFAULT_MARGIN: f32 = 0.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;

    /// Computes the margin ranking loss.
    ///
    /// # Params
    ///
    /// - `inputs1`: Scores for the first input.
    /// - `inputs2`: Scores for the second input.
    /// - `targets`: Labels indicating whether samples in `inputs1` should be ranked higher than samples
    ///   in `inputs2`. Values should be 1 or -1.
    pub fn apply(
        &self,
        inputs1: impl AsRef<Array>,
        inputs2: impl AsRef<Array>,
        targets: impl AsRef<Array>,
    ) -> Result<Array, Exception> {
        let inputs1 = inputs1.as_ref();
        let inputs2 = inputs2.as_ref();
        let targets = targets.as_ref();
        let margin = self.margin;
        let reduction = self.reduction;

        check_shape(inputs1, inputs2, "inputs1", "inputs2")?;
        check_shape(inputs1, targets, "inputs1", "targets")?;

        let margin = array!(margin);
        let diff = inputs1.subtract(inputs2)?;
        let loss = maximum(array!(0.0), -targets.multiply(diff)?.add(margin)?)?;
        reduction.reduce(loss)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::{array, assert_array_eq, ops::is_nan};

    use super::*;

    // The following unit tests are adapted from the python API at: mlx/python/tests/test_losses.py

    #[test]
    fn test_cross_entropy() {
        // No weights, no label smoothing
        let logits = array!([[0.0, f32::NEG_INFINITY], [f32::NEG_INFINITY, 0.0]]);
        let indices = array!([0, 1]);
        let expected = array!([0.0, 0.0]);
        let loss = CrossEntropy::default().apply(&logits, indices).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let cross_entropy = CrossEntropy::builder()
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(logits, probs).unwrap();
        assert!(is_nan(&loss).all(None, None).unwrap().item::<bool>());

        // With weights, no label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let weights = array!([1.0, 2.0]);
        let expected = array!([0.04858735, 0.0971747]);
        let cross_entropy = CrossEntropy::builder()
            .weights(&weights)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(&logits, indices).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let cross_entropy = CrossEntropy::builder()
            .weights(&weights)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(logits, probs).unwrap();
        assert_array_eq!(loss, expected);

        // No weights, with label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let expected = array!([0.498587, 0.498587]);
        let cross_entropy = CrossEntropy::builder()
            .label_smoothing(0.3)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(&logits, indices).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let cross_entropy = CrossEntropy::builder()
            .label_smoothing(0.3)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(logits, probs).unwrap();
        assert_array_eq!(loss, expected);

        // With weights and label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let weights = array!([1.0, 2.0]);
        let expected = array!([0.49858734, 0.9971747]);
        let cross_entropy = CrossEntropy::builder()
            .weights(&weights)
            .label_smoothing(0.3)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(&logits, indices).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let cross_entropy = CrossEntropy::builder()
            .weights(&weights)
            .label_smoothing(0.3)
            .reduction(LossReduction::None)
            .build()
            .unwrap();
        let loss = cross_entropy.apply(logits, probs).unwrap();
        assert_array_eq!(loss, expected);
    }

    #[test]
    fn test_binary_cross_entropy_with_logits_as_inputs() {
        let logits = array!([0.105361, 0.223144, 1.20397, 0.916291]);
        let targets = array!([0.0, 0.0, 1.0, 1.0]);

        // Test with reduction 'none'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .reduction(LossReduction::None)
            .build();
        let loss_none = binary_cross_entropy.apply(&logits, &targets).unwrap();
        let expected_none = array!([0.747215, 0.810930, 0.262365, 0.336472]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = binary_cross_entropy.apply(&logits, &targets).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .reduction(LossReduction::Sum)
            .build();
        let loss = binary_cross_entropy.apply(&logits, &targets).unwrap();
        let expected = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss, expected);

        // With weights, no label smoothing
        let weights = array!([1.0, 2.0, 1.0, 2.0]);
        let expected = array!([0.747215, 1.62186, 0.262365, 0.672944]);
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .weights(&weights)
            .reduction(LossReduction::None)
            .build();
        let loss = binary_cross_entropy.apply(&logits, &targets).unwrap();
        assert_array_eq!(loss, expected);
    }

    #[test]
    fn test_binary_cross_entropy_with_probs_as_inputs() {
        let probs = array!([0.5, 0.6, 0.7, 0.8]);
        let targets = array!([0.0, 0.0, 1.0, 1.0]);

        // Test with reduction 'none'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::None)
            .build();
        let loss_none = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected_none = array!([0.693147, 0.916291, 0.356675, 0.223144]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::Sum)
            .build();
        let loss = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss, expected);
    }

    #[test]
    fn test_binary_cross_entropy_with_tiny_probs_as_inputs() {
        let tiny_prob = 1e-59;
        let probs = array!([0.0, tiny_prob, 1.0 - tiny_prob, 1.0]);
        let targets = array!([0.0, 0.0, 1.0, 1.0]);

        // Test with reduction 'none'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::None)
            .build();
        let loss_none = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected_none = array!([0.0, tiny_prob, tiny_prob, 0.0]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let binary_cross_entropy = BinaryCrossEntropy::builder()
            .inputs_are_logits(false)
            .reduction(LossReduction::Sum)
            .build();
        let loss = binary_cross_entropy.apply(&probs, &targets).unwrap();
        let expected = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss, expected);
    }

    #[test]
    fn test_l1_loss() {
        let predictions = array!([0.5, 0.2, 0.9, 0.0]);
        let targets = array!([0.5, 0.2, 0.9, 0.0]);

        let expected_none = array!([0.0, 0.0, 0.0, 0.0]);
        let expected_sum = expected_none.sum(None, None).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();

        let l1_loss = L1Loss::builder().reduction(LossReduction::None).build();
        let loss_none = l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_none, expected_none);

        let l1_loss = L1Loss::builder().reduction(LossReduction::Sum).build();
        let loss_sum = l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_sum, expected_sum);

        let l1_loss = L1Loss::builder().reduction(LossReduction::Mean).build();
        let loss_mean = l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_mean, expected_mean);
    }

    #[test]
    fn test_mse_loss() {
        let predictions = array!([0.5, 0.2, 0.9, 0.0]);
        let targets = array!([0.7, 0.1, 0.8, 0.2]);

        let expected_none = array!([0.04, 0.01, 0.01, 0.04]);
        let expected_mean = expected_none.mean(None, None).unwrap();
        let expected_sum = expected_none.sum(None, None).unwrap();

        let mse_loss = MseLoss::builder().reduction(LossReduction::None).build();
        let loss_none = mse_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_none, expected_none);

        let mse_loss = MseLoss::builder().reduction(LossReduction::Mean).build();
        let loss_mean = mse_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        let mse_loss = MseLoss::builder().reduction(LossReduction::Sum).build();
        let loss_sum = mse_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_sum, expected_sum);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let predictions = array!([1.5, 2.5, 0.5, 3.5]);
        let targets = array!([1.0, 2.0, 0.5, 2.5]);
        let beta = 1.0;

        let expected_none = array!([0.125, 0.125, 0.0, 0.5]);
        let expected_sum = expected_none.sum(None, None).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();

        let smooth_l1_loss = SmoothL1Loss::builder()
            .beta(beta)
            .reduction(LossReduction::None)
            .build();
        let loss_none = smooth_l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_none, expected_none);

        let smooth_l1_loss = SmoothL1Loss::builder()
            .beta(beta)
            .reduction(LossReduction::Sum)
            .build();
        let loss_sum = smooth_l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_sum, expected_sum);

        let smooth_l1_loss = SmoothL1Loss::builder()
            .beta(beta)
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = smooth_l1_loss.apply(&predictions, &targets).unwrap();
        assert_array_eq!(loss_mean, expected_mean);
    }

    #[test]
    fn test_nll_loss() {
        let logits = array!([[0.0, f32::NEG_INFINITY], [f32::NEG_INFINITY, 0.0]]);
        let targets = array!([0, 1]);

        let expected_none = array!([0.0, 0.0]);
        let expected_sum = expected_none.sum(None, None).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();

        let nll_loss = NllLoss::builder().reduction(LossReduction::None).build();
        let loss_none = nll_loss.apply(&logits, &targets).unwrap();
        assert_array_eq!(loss_none, expected_none);

        let nll_loss = NllLoss::builder().reduction(LossReduction::Mean).build();
        let loss_mean = nll_loss.apply(&logits, &targets).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        let nll_loss = NllLoss::builder().reduction(LossReduction::Sum).build();
        let loss_sum = nll_loss.apply(&logits, &targets).unwrap();
        assert_array_eq!(loss_sum, expected_sum);
    }

    #[test]
    fn test_gaussian_nll_loss() {
        let inputs = array!([[0.1, 0.2], [0.3, 0.4]]);
        let targets = array!([[0.2, 0.1], [0.1, 0.2]]);
        let vars = array!([[0.1, 0.2], [0.3, 0.4]]);

        // Test with reduction 'none', full=False
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(false)
            .reduction(LossReduction::None)
            .build();
        let loss_none = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_none = array!([[-1.101293, -0.779719], [-0.535320, -0.408145]]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean', full=False
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(false)
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum', full=False
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(false)
            .reduction(LossReduction::Sum)
            .build();
        let loss_sum = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_sum = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss_sum, expected_sum);

        // Test with reduction='none', full=True
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(true)
            .reduction(LossReduction::None)
            .build();
        let loss_none_full = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_none_full = array!([[-0.182354, 0.139220], [0.383619, 0.510793]]);
        assert_array_eq!(loss_none_full, expected_none_full);

        // Test with reduction='mean', full=True
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(true)
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean_full = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_mean_full = expected_none_full.mean(None, None).unwrap();
        assert_array_eq!(loss_mean_full, expected_mean_full);

        // Test with reduction='sum', full=True
        let gaussian_nll_loss = GaussianNllLoss::builder()
            .full(true)
            .reduction(LossReduction::Sum)
            .build();
        let loss_sum_full = gaussian_nll_loss.apply(&inputs, &targets, &vars).unwrap();
        let expected_sum_full = expected_none_full.sum(None, None).unwrap();
        assert_array_eq!(loss_sum_full, expected_sum_full);
    }

    #[test]
    fn test_kl_div_loss() {
        let p_logits = array!([[0.5, 0.5], [0.8, 0.2]]).log();
        let q_logits = array!([[0.5, 0.5], [0.2, 0.8]]).log();

        // Test with reduction 'none'
        let kl_div_loss = KlDivLoss::builder().reduction(LossReduction::None).build();
        let loss_none = kl_div_loss.apply(&p_logits, &q_logits).unwrap();
        let expected_none = array!([0.0, 0.831777]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let kl_div_loss = KlDivLoss::builder().reduction(LossReduction::Mean).build();
        let loss_mean = kl_div_loss.apply(&p_logits, &q_logits).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let kl_div_loss = KlDivLoss::builder().reduction(LossReduction::Sum).build();
        let loss_sum = kl_div_loss.apply(&p_logits, &q_logits).unwrap();
        let expected_sum = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss_sum, expected_sum);
    }

    #[test]
    fn test_triplet_loss() {
        let anchors = array!([[1, 2, 3], [1, 2, 3]]);
        let positives = array!([[4, 5, 6], [0, -1, 2]]);
        let negatives = array!([[7, 8, 9], [3, 2, 3]]);

        // Test with reduction 'none'
        let triplet_loss = TripletLoss::builder()
            .reduction(LossReduction::None)
            .build();
        let loss_none = triplet_loss
            .apply(&anchors, &positives, &negatives)
            .unwrap();
        let expected_none = array!([0.0, 2.31662]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let triplet_loss = TripletLoss::builder()
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = triplet_loss
            .apply(&anchors, &positives, &negatives)
            .unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let triplet_loss = TripletLoss::builder().reduction(LossReduction::Sum).build();
        let loss_sum = triplet_loss
            .apply(&anchors, &positives, &negatives)
            .unwrap();
        let expected_sum = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss_sum, expected_sum);
    }

    #[test]
    fn test_hinge_loss() {
        let inputs = array!([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]);
        let targets = array!([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        let hinge_loss = HingeLoss::builder().reduction(LossReduction::Mean).build();
        let loss = hinge_loss.apply(&inputs, &targets).unwrap();
        assert_eq!(loss.item::<f32>(), 1.0);
    }

    #[test]
    fn test_huber_loss() {
        let inputs = array!([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]);
        let targets = array!([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        let huber_loss = HuberLoss::builder().reduction(LossReduction::Mean).build();
        let loss = huber_loss.apply(&inputs, &targets).unwrap();
        assert_eq!(loss.item::<f32>(), 0.5);
    }

    #[test]
    fn test_log_cosh_loss() {
        let inputs = array!([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]);
        let targets = array!([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        let log_cosh_loss = LogCoshLoss::builder()
            .reduction(LossReduction::Mean)
            .build();
        let loss = log_cosh_loss.apply(&inputs, &targets).unwrap();
        assert_float_eq!(loss.item::<f32>(), 0.433781, abs <= 1e-6);
    }

    #[test]
    fn test_cosine_similarity_loss() {
        let embeddings1 = array!([[0.5, 0.5, 0.2, 0.9], [0.1, 0.3, 0.5, 0.5]]);
        let embeddings2 = array!([[0.6, 0.4, 0.3, 0.8], [0.2, 0.5, 0.6, 0.4]]);

        // Test with reduction 'none'
        let cosine_similarity_loss = CosineSimilarityLoss::builder()
            .reduction(LossReduction::None)
            .build();
        let loss_none = cosine_similarity_loss
            .apply(&embeddings1, &embeddings2)
            .unwrap();
        let expected_none = array!([0.985344, 0.961074]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let cosine_similarity_loss = CosineSimilarityLoss::builder()
            .reduction(LossReduction::Mean)
            .build();
        let loss_mean = cosine_similarity_loss
            .apply(&embeddings1, &embeddings2)
            .unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let cosine_similarity_loss = CosineSimilarityLoss::builder()
            .reduction(LossReduction::Sum)
            .build();
        let loss_sum = cosine_similarity_loss
            .apply(&embeddings1, &embeddings2)
            .unwrap();
        let expected_sum = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss_sum, expected_sum);
    }

    #[test]
    fn test_margin_ranking_loss() {
        let inputs1 = array!([-0.573409, -0.765166, -0.0638]);
        let inputs2 = array!([0.75596, 0.225763, 0.256995]);
        let targets = array!([1, 1, -1]);

        // Test with no margin
        let margin_ranking_loss = MarginRankingLoss::builder()
            .reduction(LossReduction::None)
            .build();
        let loss = margin_ranking_loss
            .apply(&inputs1, &inputs2, &targets)
            .unwrap();
        let expected = array!([1.329369, 0.990929, 0.0]);
        assert_array_eq!(loss, expected);

        // Test with margin
        let margin_ranking_loss = MarginRankingLoss::builder()
            .margin(0.5)
            .reduction(LossReduction::None)
            .build();
        let loss = margin_ranking_loss
            .apply(&inputs1, &inputs2, &targets)
            .unwrap();
        let expected = array!([1.829369, 1.490929, 0.179205]);
        assert_array_eq!(loss, expected);
    }
}
