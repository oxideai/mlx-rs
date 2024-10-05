use mlx_rs::{
    array,
    error::Exception,
    ops::{
        abs, clip, exp, indexing::take_along_axis, log, log_add_exp, log_sum_exp, maximum, minimum,
        multiply, power, r#where, sqrt, square, sum,
    },
    Array,
};

#[inline]
fn check_shape(
    left: &Array,
    right: &Array,
    left_ident: &str,
    right_ident: &str,
) -> Result<(), Exception> {
    if left.shape() != right.shape() {
        return Err(Exception::from(format!(
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

/// Optional parameters for the `cross_entropy` function.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyOptions<'a> {
    /// Weights for each target
    pub weights: Option<&'a Array>,

    /// The axis over which to compute softmax. Default to [`CrossEntropyOptions::DEFAULT_AXIS`] if
    /// `None`
    pub axis: Option<i32>,

    /// The label smoothing factor, range [0, 1). Default to
    /// [`CrossEntropyOptions::DEFAULT_LABEL_SMOOTHING`] if `None`
    pub label_smoothing: Option<f32>,

    /// Reduction type. Default to [`CrossEntropyOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl<'a> CrossEntropyOptions<'a> {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `label_smoothing` parameter.
    pub const DEFAULT_LABEL_SMOOTHING: f32 = 0.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the cross entropy loss.
///
/// # Params
///
/// - `logits`: unnormalized predicted logits
/// - `targets`: target values, as class indices
/// - `options`: optional parameters. See [`CrossEntropyOptions`] for more details
///
/// # Panics
///
/// - Panics if `options.label_smoothing` is not in the range [0, 1)
pub fn cross_entropy(
    logits: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: CrossEntropyOptions<'_>,
) -> Result<Array, Exception> {
    let logits = logits.as_ref();
    let targets = targets.as_ref();
    let weight = options.weights;
    let axis = options.axis.unwrap_or(CrossEntropyOptions::DEFAULT_AXIS);
    let label_smoothing = options
        .label_smoothing
        .unwrap_or(CrossEntropyOptions::DEFAULT_LABEL_SMOOTHING);
    let reduction = options
        .reduction
        .unwrap_or(CrossEntropyOptions::DEFAULT_REDUCTION);

    assert!(
        (0.0..1.0).contains(&label_smoothing),
        "Label smoothing factor must be in the range [0, 1)"
    );

    let target_as_probs = targets.ndim() == logits.ndim();

    let score = if target_as_probs {
        sum(&logits.multiply(targets)?, &[axis], None)?
    } else {
        take_along_axis(logits, &targets.expand_dims(&[-1])?, axis)?.squeeze(&[-1])?
    };
    let log_sum_exp_logits = log_sum_exp(logits, &[axis], None)?;

    let mut loss = if label_smoothing > 0.0 {
        // adjust the true class score with label smoothing
        let adjusted_score = multiply(array!(1.0 - label_smoothing), score)?;

        // calculate the mean logit across the classes for smoothed loss
        let mean_logits = logits.mean(&[axis], None)?;
        let smoothed_loss = -multiply(mean_logits, array!(label_smoothing))?;

        // combine the adjusted score and smoothed loss with the logsumexp logits
        log_sum_exp_logits
            .subtract(adjusted_score)?
            .add(smoothed_loss)?
    } else {
        log_sum_exp_logits.subtract(score)?
    };

    if let Some(weights) = weight {
        check_shape(weights, &loss, "weights", "loss")?;
        loss = multiply(loss, weights)?;
    }

    reduction.reduce(loss)
}

/// Optional parameters for the `binary_cross_entropy` function.
#[derive(Debug, Clone, Default)]
pub struct BinaryCrossEntropyOptions<'a> {
    /// Optional weights for each target
    pub weights: Option<&'a Array>,

    /// Whether the inputs are logits
    pub with_logits: Option<bool>,

    /// Reduction type. Default to [`BinaryCrossEntropyOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl<'a> BinaryCrossEntropyOptions<'a> {
    /// Default value for the `with_logits` parameter.
    pub const DEFAULT_WITH_LOGITS: bool = true;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the binary cross entropy loss.
///
/// # Params
///
/// - `logits`: unnormalized predicted logits
/// - `targets`: binary target values in {0, 1}
/// - `options`: optional parameters. See [`BinaryCrossEntropyOptions`] for more details
pub fn binary_cross_entropy(
    logits: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: BinaryCrossEntropyOptions<'_>,
) -> Result<Array, Exception> {
    let logits = logits.as_ref();
    let targets = targets.as_ref();
    let weights = options.weights;
    let with_logits = options
        .with_logits
        .unwrap_or(BinaryCrossEntropyOptions::DEFAULT_WITH_LOGITS);
    let reduction = options
        .reduction
        .unwrap_or(BinaryCrossEntropyOptions::DEFAULT_REDUCTION);

    let mut loss = if with_logits {
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

/// Optional parameters for the `l1_loss` function.
#[derive(Debug, Clone, Default)]
pub struct L1LossOptions {
    /// Reduction type. Default to [`L1lossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl L1LossOptions {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;
}

/// Computes the L1 loss.
///
/// # Params
///
/// - `predictions`: predicted values
/// - `targets`: target values
/// - `options`: optional parameters. See [`L1LossOptions`] for more details
pub fn l1_loss(
    predictions: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: L1LossOptions,
) -> Result<Array, Exception> {
    let predictions = predictions.as_ref();
    let targets = targets.as_ref();
    let reduction = options
        .reduction
        .unwrap_or(L1LossOptions::DEFAULT_REDUCTION);

    check_shape(predictions, targets, "predictions", "targets")?;
    let loss = predictions.subtract(targets)?.abs();
    reduction.reduce(loss)
}

/// Optional parameters for the `mse_loss` function.
#[derive(Debug, Clone, Default)]
pub struct MseLossOptions {
    /// Reduction type. Default to [`MseLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl MseLossOptions {
    /// Default value for the reduction parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;
}

/// Computes the mean squared error loss.
///
/// # Params
///
/// - `predictions`: predicted values
/// - `targets`: target values
/// - `options`: optional parameters. See [`MseLossOptions`] for more details
pub fn mse_loss(
    predictions: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: MseLossOptions,
) -> Result<Array, Exception> {
    let predictions = predictions.as_ref();
    let targets = targets.as_ref();
    let reduction = options
        .reduction
        .unwrap_or(MseLossOptions::DEFAULT_REDUCTION);

    check_shape(predictions, targets, "predictions", "targets")?;
    let loss = predictions.subtract(targets)?.square();
    reduction.reduce(loss)
}

/// Optional parameters for the `nll_loss` function.
#[derive(Debug, Clone, Default)]
pub struct NllLossOptions {
    /// distribution axis. Default to [`NllLossOptions::DEFAULT_AXIS`] if `None`
    pub axis: Option<i32>,

    /// Reduction type. Default to [`NllLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl NllLossOptions {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the negative log likelihood loss.
///
/// # Params
///
/// - `inputs`: predicted distribution in log space
/// - `targets`: target values
/// - `options`: optional parameters. See [`NllLossOptions`] for more details
pub fn nll_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: NllLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let axis = options.axis.unwrap_or(NllLossOptions::DEFAULT_AXIS);
    let reduction = options
        .reduction
        .unwrap_or(NllLossOptions::DEFAULT_REDUCTION);

    let loss = -take_along_axis(inputs, &targets.expand_dims(&[-1])?, axis)?.squeeze(&[-1])?;
    reduction.reduce(loss)
}

/// Optional parameters for the `gaussian_nll_loss` function.
#[derive(Debug, Clone, Default)]
pub struct GaussianNllLossOptions {
    /// Whether to include the constant term in the loss calculation. Default to
    /// [`GaussianNllLossOptions::DEFAULT_FULL`] if `None`
    pub full: Option<bool>,

    /// Small positive constant for numerical stability. Default to
    /// [`GaussianNllLossOptions::DEFAULT_EPS`] if `None`
    pub eps: Option<f32>,

    /// Reduction type. Default to [`GaussianNllLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl GaussianNllLossOptions {
    /// Default value for the `full` parameter.
    pub const DEFAULT_FULL: bool = false;

    /// Default value for the `eps` parameter.
    pub const DEFAULT_EPS: f32 = 1e-6;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the negative log likelihood loss for a Gaussian distribution.
///
/// # Params
///
/// - `inputs`: The predicted expectation of the Gaussian distribution.
/// - `targets`: The target values (samples from the Gaussian distribution).
/// - `vars`: The predicted variance of the Gaussian distribution.
/// - `options`: optional parameters. See [`GaussianNllLossOptions`] for more details
pub fn gaussian_nll_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    vars: impl AsRef<Array>,
    options: GaussianNllLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let vars = vars.as_ref();
    let full = options.full.unwrap_or(GaussianNllLossOptions::DEFAULT_FULL);
    let eps = options.eps.unwrap_or(GaussianNllLossOptions::DEFAULT_EPS);
    let reduction = options
        .reduction
        .unwrap_or(GaussianNllLossOptions::DEFAULT_REDUCTION);

    check_shape(inputs, targets, "inputs", "targets")?;
    check_shape(inputs, vars, "inputs", "vars")?;

    // For numerical stability
    let vars = maximum(vars, array!(eps))?;
    let mut loss =
        array!(0.5) * (log(&vars).add(square(&targets.subtract(inputs)?).divide(&vars)?)?);

    if full {
        let pi = array!(std::f32::consts::PI);
        loss = loss.add(array!(0.5).multiply(log(&array!(2.0).multiply(pi)?))?)?;
    }

    reduction.reduce(loss)
}

/// Optional parameters for the `kl_div_loss` function.
#[derive(Debug, Clone, Default)]
pub struct KlDivLossOptions {
    /// The distribution axis. Default to [`KlDivLossOptions::DEFAULT_AXIS`] if `None`
    pub axis: Option<i32>,

    /// Reduction type. Default to [`KlDivLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl KlDivLossOptions {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the Kullback-Leibler divergence loss.
///
/// # Params
///
/// - `inputs`: Log probabilities for the predicted distribution.
/// - `targets`: Log probabilities for the target distribution.
/// - `options`: optional parameters. See [`KlDivLossOptions`] for more details
pub fn kl_div_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: KlDivLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let axis = options.axis.unwrap_or(KlDivLossOptions::DEFAULT_AXIS);
    let reduction = options
        .reduction
        .unwrap_or(KlDivLossOptions::DEFAULT_REDUCTION);

    let loss = sum(
        &exp(targets).multiply(targets.subtract(inputs)?)?,
        &[axis],
        None,
    )?;
    reduction.reduce(loss)
}

/// Optional parameters for the `smooth_l1_loss` function.
#[derive(Debug, Clone, Default)]
pub struct SmoothL1LossOptions {
    /// The threshold after which the loss changes from the squared to the absolute difference.
    /// Default to [`SmoothL1LossOptions::DEFAULT_BETA`] if `None`
    pub beta: Option<f32>,

    /// Reduction type. Default to [`SmoothL1LossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl SmoothL1LossOptions {
    /// Default value for the `beta` parameter.
    pub const DEFAULT_BETA: f32 = 1.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::Mean;
}

/// Computes the smooth L1 loss.
///
/// The smooth L1 loss is a variant of the L1 loss which replaces the absolute
/// difference with a squared difference when the absolute difference is less
/// than `beta`.
///
/// # Params
///
/// - `predictions`: predicted values
/// - `targets`: target values
/// - `options`: optional parameters. See [`SmoothL1LossOptions`] for more details
pub fn smooth_l1_loss(
    predictions: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: SmoothL1LossOptions,
) -> Result<Array, Exception> {
    let predictions = predictions.as_ref();
    let targets = targets.as_ref();
    let beta = options.beta.unwrap_or(SmoothL1LossOptions::DEFAULT_BETA);
    let reduction = options
        .reduction
        .unwrap_or(SmoothL1LossOptions::DEFAULT_REDUCTION);

    check_shape(predictions, targets, "predictions", "targets")?;
    let diff = predictions.subtract(targets)?;
    let loss = r#where(
        &diff.lt(array!(beta))?,
        array!(0.5).multiply(square(&diff))?.divide(&array!(beta))?,
        abs(&diff).subtract(array!(0.5).multiply(array!(beta))?)?,
    )?;
    reduction.reduce(loss)
}

/// Optional parameters for the `triplet_loss` function.
#[derive(Debug, Clone, Default)]
pub struct TripletLossOptions {
    /// Distribution axis. Default to [`TripletLossOptions::DEFAULT_AXIS`] if `None`
    pub axis: Option<i32>,

    /// The norm degree for pairwise distance. Default to [`TripletLossOptions::DEFAULT_P`] if `None`
    pub p: Option<f32>,

    /// Margin for the triplet loss. Default to [`TripletLossOptions::DEFAULT_MARGIN`] if `None`
    pub margin: Option<f32>,

    /// Small positive constant for numerical stability. Default to [`TripletLossOptions::DEFAULT_EPS`] if `None`
    pub eps: Option<f32>,

    /// Reduction type. Default to [`TripletLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl TripletLossOptions {
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
}

/// Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is represented with alpha in the math section.
///
/// # Params
///
/// - `anchors`: The anchor samples
/// - `positives`: The positive samples
/// - `neonatives`: The negative samples
/// - `options`: optional parameters. See [`TripletLossOptions`] for more details
pub fn triplet_loss(
    anchors: impl AsRef<Array>,
    positives: impl AsRef<Array>,
    negatives: impl AsRef<Array>,
    options: TripletLossOptions,
) -> Result<Array, Exception> {
    let anchors = anchors.as_ref();
    let positives = positives.as_ref();
    let negatives = negatives.as_ref();
    let axis = options.axis.unwrap_or(TripletLossOptions::DEFAULT_AXIS);
    let p = options.p.unwrap_or(TripletLossOptions::DEFAULT_P);
    let margin = options.margin.unwrap_or(TripletLossOptions::DEFAULT_MARGIN);
    let eps = options.eps.unwrap_or(TripletLossOptions::DEFAULT_EPS);
    let reduction = options
        .reduction
        .unwrap_or(TripletLossOptions::DEFAULT_REDUCTION);

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

/// Optional parameters for the `hinge_loss` function.
#[derive(Debug, Clone, Default)]
pub struct HingeLossOptions {
    /// Reduction type. Default to [`HingeLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl HingeLossOptions {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the hinge loss.
///
/// # Params
///
/// - `inputs`: predicted values
/// - `targets`: target values, -1 or 1
/// - `options`: optional parameters. See [`HingeLossOptions`] for more details
pub fn hinge_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: HingeLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let reduction = options
        .reduction
        .unwrap_or(HingeLossOptions::DEFAULT_REDUCTION);

    let a = array!(1.0).subtract(inputs.multiply(targets)?)?;
    let b = array!(0.0);
    let loss = maximum(a, b)?;
    reduction.reduce(loss)
}

/// Optional parameters for the `huber_loss` function.
#[derive(Debug, Clone, Default)]
pub struct HuberLossOptions {
    /// The threshold at which to change between L1 and L2 loss. Default to
    /// [`HuberLossOptions::DEFAULT_DELTA`] if `None`
    pub delta: Option<f32>,

    /// Reduction type. Default to [`HuberLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl HuberLossOptions {
    /// Default value for the `delta` parameter.
    pub const DEFAULT_DELTA: f32 = 1.0;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the Huber loss.
///
/// # Params
///
/// - `inputs`: predicted values
/// - `targets`: target values
/// - `options`: optional parameters. See [`HuberLossOptions`] for more details
pub fn huber_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: HuberLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let delta = options.delta.unwrap_or(HuberLossOptions::DEFAULT_DELTA);
    let reduction = options
        .reduction
        .unwrap_or(HuberLossOptions::DEFAULT_REDUCTION);

    let errors = inputs.subtract(targets)?;
    let abs_errors = errors.abs();
    let quadratic = minimum(&abs_errors, array!(delta))?;
    let linear = abs_errors.subtract(&quadratic)?;
    let loss = array!(0.5)
        .multiply(square(&quadratic))?
        .add(array!(delta).multiply(linear)?)?;
    reduction.reduce(loss)
}

/// Optional parameters for the `log_cosh_loss` function.
#[derive(Debug, Clone, Default)]
pub struct LogCoshLossOptions {
    /// Reduction type. Default to [`LogCoshLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl LogCoshLossOptions {
    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the log cosh loss between inputs and targets.
///
/// Logcosh acts like L2 loss for small errors, ensuring stable gradients,
/// and like the L1 loss for large errors, reducing sensitivity to outliers. This
/// dual behavior offers a balanced, robust approach for regression tasks.
///
/// # Params
///
/// - `inputs`: predicted values
/// - `targets`: target values
/// - `options`: optional parameters. See [`LogCoshLossOptions`] for more details
pub fn log_cosh_loss(
    inputs: impl AsRef<Array>,
    targets: impl AsRef<Array>,
    options: LogCoshLossOptions,
) -> Result<Array, Exception> {
    let inputs = inputs.as_ref();
    let targets = targets.as_ref();
    let reduction = options
        .reduction
        .unwrap_or(LogCoshLossOptions::DEFAULT_REDUCTION);

    let errors = inputs.subtract(targets)?;
    let neg_errors = errors.negative()?;
    let loss = log_add_exp(errors, neg_errors)?.subtract(log(&array!(2.0)))?;
    reduction.reduce(loss)
}

/// Optional parameters for the `cosine_similarity_loss` function.
#[derive(Debug, Clone, Default)]
pub struct CosineSimilarityLossOptions {
    /// Embedding axis. Default to [`CosineSimilarityLossOptions::DEFAULT_AXIS`] if `None`
    pub axis: Option<i32>,

    /// minimum value of the denominator used for numerical stability. Default to
    /// [`CosineSimilarityLossOptions::DEFAULT_EPS`] if `None`
    pub eps: Option<f32>,

    /// Reduction type. Default to [`CosineSimilarityLossOptions::DEFAULT_REDUCTION`] if `None`
    pub reduction: Option<LossReduction>,
}

impl CosineSimilarityLossOptions {
    /// Default value for the `axis` parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the `eps` parameter.
    pub const DEFAULT_EPS: f32 = 1e-8;

    /// Default value for the `reduction` parameter.
    pub const DEFAULT_REDUCTION: LossReduction = LossReduction::None;
}

/// Computes the cosine similarity loss.
///
/// # Params
///
/// - `x1`: first array
/// - `x2`: second array
/// - `options`: optional parameters. See [`CosineSimilarityLossOptions`] for more details
pub fn cosime_similarity_loss(
    x1: impl AsRef<Array>,
    x2: impl AsRef<Array>,
    options: CosineSimilarityLossOptions,
) -> Result<Array, Exception> {
    let x1 = x1.as_ref();
    let x2 = x2.as_ref();
    let axis = options
        .axis
        .unwrap_or(CosineSimilarityLossOptions::DEFAULT_AXIS);
    let eps = options
        .eps
        .unwrap_or(CosineSimilarityLossOptions::DEFAULT_EPS);
    let reduction = options
        .reduction
        .unwrap_or(CosineSimilarityLossOptions::DEFAULT_REDUCTION);

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

#[cfg(test)]
mod tests {
    use mlx_rs::{array, assert_array_eq, ops::is_nan};

    use super::*;

    // The following unit tests are adapted from the python API at: mlx/python/tests/test_losses.py

    #[test]
    fn test_cross_entropy() {
        // No weights, no label smoothing
        let logits = array!([[0.0, f32::NEG_INFINITY], [f32::NEG_INFINITY, 0.0]]);
        let indices = array!([0, 1]);
        let expected = array!([0.0, 0.0]);
        let options = CrossEntropyOptions::default();
        let loss = cross_entropy(&logits, indices, options).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let options = CrossEntropyOptions {
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(logits, probs, options).unwrap();
        assert!(is_nan(&loss).all(None, None).unwrap().item::<bool>());

        // With weights, no label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let weights = array!([1.0, 2.0]);
        let expected = array!([0.04858735, 0.0971747]);
        let options = CrossEntropyOptions {
            weights: Some(&weights),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(&logits, indices, options).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let options = CrossEntropyOptions {
            weights: Some(&weights),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(logits, probs, options).unwrap();
        assert_array_eq!(loss, expected);

        // No weights, with label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let expected = array!([0.498587, 0.498587]);
        let options = CrossEntropyOptions {
            label_smoothing: Some(0.3),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(&logits, indices, options).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let options = CrossEntropyOptions {
            label_smoothing: Some(0.3),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(logits, probs, options).unwrap();
        assert_array_eq!(loss, expected);

        // With weights and label smoothing
        let logits = array!([[2.0, -1.0], [-1.0, 2.0]]);
        let indices = array!([0, 1]);
        let weights = array!([1.0, 2.0]);
        let expected = array!([0.49858734, 0.9971747]);
        let options = CrossEntropyOptions {
            weights: Some(&weights),
            label_smoothing: Some(0.3),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(&logits, indices, options).unwrap();
        assert_array_eq!(loss, expected);

        let probs = array!([[1.0, 0.0], [0.0, 1.0]]);
        let options = CrossEntropyOptions {
            weights: Some(&weights),
            label_smoothing: Some(0.3),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = cross_entropy(logits, probs, options).unwrap();
        assert_array_eq!(loss, expected);
    }

    #[test]
    fn test_binary_cross_entropy_with_logits_as_inputs() {
        let logits = array!([0.105361, 0.223144, 1.20397, 0.916291]);
        let targets = array!([0.0, 0.0, 1.0, 1.0]);

        // Test with reduction 'none'
        let options = BinaryCrossEntropyOptions {
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss_none = binary_cross_entropy(&logits, &targets, options).unwrap();
        let expected_none = array!([0.747215, 0.810930, 0.262365, 0.336472]);
        assert_array_eq!(loss_none, expected_none);

        // Test with reduction 'mean'
        let options = BinaryCrossEntropyOptions {
            reduction: Some(LossReduction::Mean),
            ..Default::default()
        };
        let loss_mean = binary_cross_entropy(&logits, &targets, options).unwrap();
        let expected_mean = expected_none.mean(None, None).unwrap();
        assert_array_eq!(loss_mean, expected_mean);

        // Test with reduction 'sum'
        let options = BinaryCrossEntropyOptions {
            reduction: Some(LossReduction::Sum),
            ..Default::default()
        };
        let loss = binary_cross_entropy(&logits, &targets, options).unwrap();
        let expected = expected_none.sum(None, None).unwrap();
        assert_array_eq!(loss, expected);

        // With weights, no label smoothing
        let weights = array!([1.0, 2.0, 1.0, 2.0]);
        let expected = array!([0.747215, 1.62186, 0.262365, 0.672944]);
        let options = BinaryCrossEntropyOptions {
            weights: Some(&weights),
            reduction: Some(LossReduction::None),
            ..Default::default()
        };
        let loss = binary_cross_entropy(&logits, &targets, options).unwrap();
        assert_array_eq!(loss, expected);
    }
}
