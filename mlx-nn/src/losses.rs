use mlx_rs::{
    array,
    error::Exception,
    ops::{clip, indexing::take_along_axis, log, log_add_exp, log_sum_exp, multiply, sum},
    Array,
};

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
    /// Default value for the axis parameter.
    pub const DEFAULT_AXIS: i32 = -1;

    /// Default value for the label smoothing parameter.
    pub const DEFAULT_LABEL_SMOOTHING: f32 = 0.0;

    /// Default value for the reduction parameter.
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
        // assert_eq!(weights.shape(), loss.shape()); // TODO: is this necessary?
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
    /// Default value for the with_logits parameter.
    pub const DEFAULT_WITH_LOGITS: bool = true;

    /// Default value for the reduction parameter.
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
        // assert_eq!(weights.shape(), loss.shape()); // TODO: is this necessary?
        loss = multiply(loss, weights)?;
    }

    reduction.reduce(loss)
}

// The following unit tests are adapted from the python API at: mlx/python/tests/test_losses.py
#[cfg(test)]
mod tests {
    use mlx_rs::{array, assert_array_eq, ops::is_nan};

    use super::*;

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
