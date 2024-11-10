use std::borrow::Cow;

use mlx_macros::ModuleParameters;
use mlx_rs::{array, error::Exception, module::{Module, Param}, ops::{ones, rsqrt, zeros}, Array};

fn instance_norm(x: &Array, axes: &[i32], eps: &Array) -> Result<Array, Exception> {
    // Compute stats
    let mean = x.mean(axes, true)?;
    let variance = x.variance(axes, true, None)?;

    // Normalize
    let x = x.subtract(&mean)?.multiply(&rsqrt(&variance.add(eps)?))?;
    
    Ok(x)
}

/// Builder for [`InstanceNorm`].
#[derive(Debug, Clone, Default)]
pub struct InstanceNormBuilder {
    /// Value added to the denominator for numerical stability. Default to
    /// [`InstanceNorm::DEFAULT_EPS`].
    pub eps: Option<f32>,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`InstanceNorm::DEFAULT_AFFINE`].
    pub affine: Option<bool>,
}

impl InstanceNormBuilder {
    /// Creates a new [`InstanceNormBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `eps`
    pub fn eps(mut self, eps: impl Into<Option<f32>>) -> Self {
        self.eps = eps.into();
        self
    }

    /// Sets the `affine`
    pub fn affine(mut self, affine: impl Into<Option<bool>>) -> Self {
        self.affine = affine.into();
        self
    }

    /// Builds the [`InstanceNorm`] layer.
    pub fn build(self, dimensions: i32) -> Result<InstanceNorm, Exception> {
        let eps = self.eps.unwrap_or(InstanceNorm::DEFAULT_EPS);
        let affine = self.affine.unwrap_or(InstanceNorm::DEFAULT_AFFINE);

        let (weight, bias) = if affine {
            (
                Some(ones::<f32>(&[dimensions])?),
                Some(zeros::<f32>(&[dimensions])?),
            )
        } else {
            (None, None)
        };

        Ok(InstanceNorm {
            dimensions,
            eps: array!(eps),
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }
}

/// Applies instance normalization [1] on the inputs.
///
/// ### References
/// 
/// 1. [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)
#[derive(Debug, Clone, ModuleParameters)]
pub struct InstanceNorm {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability. 
    pub eps: Array,

    /// An optional trainable weight
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    pub bias:  Param<Option<Array>>,
}

impl InstanceNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Disable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = false;

    /// Creates a new [`InstanceNormBuilder`].
    pub fn builder() -> InstanceNormBuilder {
        InstanceNormBuilder::new()
    }

    /// Creates a new instance normalization layer with the default parameters.
    pub fn new(dimensions: i32) -> Result<Self, Exception> {
        InstanceNormBuilder::new().build(dimensions)
    }
}

impl Module for InstanceNorm {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let reduction_axes = (1..x.ndim() as i32 - 1).collect::<Vec<_>>();

        let x = instance_norm(x, &reduction_axes, &self.eps)?;

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(x)?.add(bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, _mode: bool) { }
}

/// Builder for [`LayerNorm`].
#[derive(Debug, Clone, Default)]
pub struct LayerNormBuilder {
    /// Value added to the denominator for numerical stability. Default to
    /// [`LayerNorm::DEFAULT_EPS`].
    pub eps: Option<f32>,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`LayerNorm::DEFAULT_AFFINE`].
    pub affine: Option<bool>,
}

impl LayerNormBuilder {
    /// Creates a new [`LayerNormBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `eps`
    pub fn eps(mut self, eps: impl Into<Option<f32>>) -> Self {
        self.eps = eps.into();
        self
    }

    /// Sets the `affine`
    pub fn affine(mut self, affine: impl Into<Option<bool>>) -> Self {
        self.affine = affine.into();
        self
    }

    /// Builds the [`LayerNorm`] layer.
    pub fn build(self, dimensions: i32) -> Result<LayerNorm, Exception> {
        let eps = self.eps.unwrap_or(LayerNorm::DEFAULT_EPS);
        let affine = self.affine.unwrap_or(LayerNorm::DEFAULT_AFFINE);

        let (weight, bias) = if affine {
            (
                Some(ones::<f32>(&[dimensions])?),
                Some(zeros::<f32>(&[dimensions])?),
            )
        } else {
            (None, None)
        };

        Ok(LayerNorm {
            dimensions,
            eps,
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }
}

/// Applies layer normalization [1] on the inputs.
///
/// ### References
/// 
/// 1. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
#[derive(Debug, Clone, ModuleParameters)]
pub struct LayerNorm {
    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: f32,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,
}

impl LayerNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;

    /// Creates a new [`LayerNormBuilder`].
    pub fn builder() -> LayerNormBuilder {
        LayerNormBuilder::new()
    }

    /// Creates a new layer normalization layer with the default parameters.
    pub fn new(dimensions: i32) -> Result<Self, Exception> {
        LayerNormBuilder::new().build(dimensions)
    }
}

impl Module for LayerNorm {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let weight = self.weight.as_ref();
        let bias = self.bias.as_ref();
        let eps = self.eps;
        mlx_rs::fast::layer_norm(x, weight, bias, eps)
    }

    fn training_mode(&mut self, _mode: bool) { }
}

/// Builder for [`RmsNorm`].
#[derive(Debug, Clone, Default)]
pub struct RmsNormBuilder {
    /// Value added to the denominator for numerical stability. Default to
    /// [`RmsNorm::DEFAULT_EPS`].
    pub eps: Option<f32>,
}

impl RmsNormBuilder {
    /// Creates a new [`RmsNormBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `eps`
    pub fn eps(mut self, eps: impl Into<Option<f32>>) -> Self {
        self.eps = eps.into();
        self
    }

    /// Builds the [`RmsNorm`] layer.
    pub fn build(self, dimensions: i32) -> Result<RmsNorm, Exception> {
        let weight = ones::<f32>(&[dimensions])?;
        let eps = self.eps.unwrap_or(RmsNorm::DEFAULT_EPS);
        Ok(RmsNorm {
            weight: Param::new(weight),
            eps,
        })
    }
}

/// Applies Root Mean Square normalization [1] to the inputs.
///
/// Concretely:
///
/// ```swift
/// weight * x * MLX.rsqrt(x.square().mean() + eps)
/// ```
///
/// where `weight` is initialized with ones and `eps` is a small float to
/// ensure the numerical stability of inverse square root.
///
/// ### References
/// 
/// 1. [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNorm {
    /// Weight
    #[param]
    pub weight: Param<Array>,

    /// A small float to ensure the numerical stability
    pub eps: f32,
}

impl RmsNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Creates a new [`RmsNormBuilder`].
    pub fn builder() -> RmsNormBuilder {
        RmsNormBuilder::new()
    }

    /// Creates a new RMS normalization layer with the default parameters.
    pub fn new(dimensions: i32) -> Result<Self, Exception> {
        RmsNormBuilder::new().build(dimensions)
    }
}

impl Module for RmsNorm {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let weight = self.weight.as_ref();
        let eps = self.eps;
        mlx_rs::fast::rms_norm(x, weight, eps)
    }

    fn training_mode(&mut self, _mode: bool) { }
}

/// Builder for [`GroupNorm`].
#[derive(Debug, Clone, Default)]
pub struct GroupNormBuilder {
    /// Value added to the denominator for numerical stability. Default to
    /// [`GroupNorm::DEFAULT_EPS`].
    pub eps: Option<f32>,

    /// If `true`, add a trainable `weight` and `bias`. Default to
    /// [`GroupNorm::DEFAULT_AFFINE`].
    pub affine: Option<bool>,

    /// If `true`, perform the group normalization in the same order/grouping as PyTorch.
    /// Default to [`GroupNorm::DEFAULT_PYTORCH_COMPATIBLE`].
    pub pytorch_compatible: Option<bool>,
}

impl GroupNormBuilder {
    /// Creates a new [`GroupNormBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `eps`
    pub fn eps(mut self, eps: impl Into<Option<f32>>) -> Self {
        self.eps = eps.into();
        self
    }

    /// Sets the `affine`
    pub fn affine(mut self, affine: impl Into<Option<bool>>) -> Self {
        self.affine = affine.into();
        self
    }

    /// Sets the `pytorch_compatible`
    pub fn pytorch_compatible(mut self, pytorch_compatible: impl Into<Option<bool>>) -> Self {
        self.pytorch_compatible = pytorch_compatible.into();
        self
    }

    /// Builds the [`GroupNorm`] layer.
    pub fn build(self, group_count: i32, dimensions: i32) -> Result<GroupNorm, Exception> {
        let eps = self.eps.unwrap_or(GroupNorm::DEFAULT_EPS);
        let affine = self.affine.unwrap_or(GroupNorm::DEFAULT_AFFINE);
        let pytorch_compatible = self.pytorch_compatible.unwrap_or(GroupNorm::DEFAULT_PYTORCH_COMPATIBLE);

        let (weight, bias) = if affine {
            (
                Some(ones::<f32>(&[dimensions])?),
                Some(zeros::<f32>(&[dimensions])?),
            )
        } else {
            (None, None)
        };

        Ok(GroupNorm {
            group_count,
            dimensions,
            eps: array!(eps),
            pytorch_compatible,
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }
}

/// Applies Group Normalization [1] on the inputs.
///
/// ### References
/// 
/// 1. [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
#[derive(Debug, Clone, ModuleParameters)]
pub struct GroupNorm {
    /// Number of groups to separate the features into
    pub group_count: i32,

    /// Number of features in the input
    pub dimensions: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: Array,

    /// If `true`, perform the group normalization in the same order/grouping as PyTorch.
    pub pytorch_compatible: bool,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,
}

impl GroupNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;

    /// Default value for `pytorch_compatible`.
    pub const DEFAULT_PYTORCH_COMPATIBLE: bool = false;

    /// Creates a new [`GroupNormBuilder`].
    pub fn builder() -> GroupNormBuilder {
        GroupNormBuilder::new()
    }

    /// Creates a new group normalization layer with the default parameters.
    pub fn new(group_count: i32, dimensions: i32) -> Result<Self, Exception> {
        GroupNormBuilder::new().build(group_count, dimensions)
    }

    fn pytorch_group_norm(&self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let rest = &x.shape()[1..x.ndim() - 1];
        let group_size = dims / self.group_count;

        // Split into groups
        let x = x.reshape(&[batch, -1, self.group_count, group_size])?;
        let x = x.transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, self.group_count, -1])?;
            
        // Normalize
        let x = mlx_rs::fast::layer_norm(
            x,
            None,
            None,
            self.eps.item::<f32>(),
        )?;

        let x = x.reshape(&[batch, self.group_count, -1, group_size])?;

        let new_shape: Vec<_> = [batch].into_iter()
            .chain(rest.into_iter().copied())
            .chain([dims].into_iter())
            .collect();
        x.transpose(&[0, 2, 1, 3])?
            .reshape(&new_shape[..])
    }

    fn group_norm(&self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let rest = &x.shape()[1..x.ndim() - 1];

        // Split into groups
        let x = x.reshape(&[batch, -1, self.group_count])?;

        // Normalize
        let x = instance_norm(&x, &[1], &self.eps)?;

        let new_shape: Vec<_> = [batch].into_iter()
            .chain(rest.into_iter().copied())
            .chain([dims].into_iter())
            .collect();
        x.reshape(&new_shape[..])
    }
}

impl Module for GroupNorm {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let x = if self.pytorch_compatible {
            self.pytorch_group_norm(x)?
        } else {
            self.group_norm(x)?
        };

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(&x)?.add(&bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, _mode: bool) { }
}

/// Builder for [`BatchNorm`].
#[derive(Debug, Clone, Default)]
pub struct BatchNormBuilder {
    /// Value added to the denominator for numerical stability. Default to
    /// [`BatchNorm::DEFAULT_EPS`].
    pub eps: Option<f32>,

    /// Momentum for updating the running mean and variance. Default to
    /// [`BatchNorm::DEFAULT_MOMENTUM`].
    pub momentum: Option<f32>,

    /// If `true`, addes a trainable `weight` and `bias`. Default to
    /// [`BatchNorm::DEFAULT_AFFINE`].
    pub affine: Option<bool>,

    /// If `true`, track the running mean and variance. Default to
    /// [`BatchNorm::DEFAULT_TRACK_RUNNING_STATS`].
    pub track_running_stats: Option<bool>,
}

impl BatchNormBuilder {
    /// Creates a new [`BatchNormBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `eps`
    pub fn eps(mut self, eps: impl Into<Option<f32>>) -> Self {
        self.eps = eps.into();
        self
    }

    /// Sets the `momentum`
    pub fn momentum(mut self, momentum: impl Into<Option<f32>>) -> Self {
        self.momentum = momentum.into();
        self
    }

    /// Sets the `affine`
    pub fn affine(mut self, affine: impl Into<Option<bool>>) -> Self {
        self.affine = affine.into();
        self
    }

    /// Sets the `track_running_stats`
    pub fn track_running_stats(mut self, track_running_stats: impl Into<Option<bool>>) -> Self {
        self.track_running_stats = track_running_stats.into();
        self
    }

    /// Builds the [`BatchNorm`] layer.
    pub fn build(self, feature_count: i32) -> Result<BatchNorm, Exception> {
        let eps = self.eps.unwrap_or(BatchNorm::DEFAULT_EPS);
        let momentum = self.momentum.unwrap_or(BatchNorm::DEFAULT_MOMENTUM);
        let affine = self.affine.unwrap_or(BatchNorm::DEFAULT_AFFINE);
        let track_running_stats = self.track_running_stats.unwrap_or(BatchNorm::DEFAULT_TRACK_RUNNING_STATS);

        let (weight, bias) = if affine {
            (
                Some(ones::<f32>(&[feature_count])?),
                Some(zeros::<f32>(&[feature_count])?),
            )
        } else {
            (None, None)
        };

        let (running_mean, running_var) = if track_running_stats {
            (
                Some(zeros::<f32>(&[feature_count])?),
                Some(ones::<f32>(&[feature_count])?),
            )
        } else {
            (None, None)
        };

        Ok(BatchNorm {
            feature_count,
            eps: array!(eps),
            momentum: array!(momentum),
            weight: Param::new(weight),
            bias: Param::new(bias),
            running_mean: Param::new(running_mean),
            running_var: Param::new(running_var),
            training: BatchNorm::DEFAULT_TRAINING,
        })
    }
}

/// Applies batch normalization [1] on the inputs.
///
/// ### References
/// 
/// 1. [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
#[derive(Debug, Clone, ModuleParameters)]
pub struct BatchNorm {
    /// Number of features in the input
    pub feature_count: i32,

    /// Value added to the denominator for numerical stability.
    pub eps: Array,

    /// Momentum for updating the running mean and variance.
    pub momentum: Array,

    /// An optional trainable weight
    #[param]
    pub weight: Param<Option<Array>>,

    /// An optional trainable bias
    #[param]
    pub bias: Param<Option<Array>>,

    /// Tracked running mean
    #[param]
    pub running_mean: Param<Option<Array>>,

    /// Tracked running variance
    #[param]
    pub running_var: Param<Option<Array>>,

    /// If `true`, the module is in training mode.
    pub training: bool,
}

impl BatchNorm {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// Default value for `momentum`.
    pub const DEFAULT_MOMENTUM: f32 = 0.1;

    /// Enable trainable `weight` and `bias` by default.
    pub const DEFAULT_AFFINE: bool = true;

    /// Enable tracking of running mean and variance by default.
    pub const DEFAULT_TRACK_RUNNING_STATS: bool = true;

    /// Enable training mode by default.
    pub const DEFAULT_TRAINING: bool = true;

    /// Creates a new [`BatchNormBuilder`].
    pub fn builder() -> BatchNormBuilder {
        BatchNormBuilder::new()
    }

    /// Creates a new batch normalization layer with the default parameters.
    pub fn new(feature_count: i32) -> Result<Self, Exception> {
        BatchNormBuilder::new().build(feature_count)
    }

    fn stats(x: &Array) -> Result<(Array, Array), Exception> {
        let reduction_axes = (0..x.ndim() as i32 - 1).collect::<Vec<_>>();

        let mean = x.mean(&reduction_axes, None)?;
        let variance = x.variance(&reduction_axes, None, None)?;

        Ok((mean, variance))
    }
}

impl Module for BatchNorm {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();
        if ndim < 2 || ndim > 4 {
            return Err(Exception::custom("Input tensor must be at least 2 dimensions and at most 4 dimensions"));
        }

        let (mean, variance) = Self::stats(x)?;
        let mut mean = Cow::Owned(mean);
        let mut variance = Cow::Owned(variance);

        if let (Some(running_mean), Some(running_var)) = (self.running_mean.as_mut(), self.running_var.as_mut()) {
            if self.training {
                let mu = &self.momentum;
                // SAFETY: momentum is a single element array
                let one_minus_mu = array!(1.0) - mu;

                *running_mean = one_minus_mu.multiply(&running_mean)?
                    .add(mu.multiply(&mean)?)?;
                *running_var = one_minus_mu.multiply(&running_var)?
                    .add(mu.multiply(&variance)?)?;
            } else {
                mean = Cow::Borrowed(&*running_mean);
                variance = Cow::Borrowed(&*running_var);
            }
        }

        let x = x.subtract(&mean)?.multiply(&rsqrt(&variance.add(&self.eps)?))?;

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            weight.multiply(&x)?.add(&bias)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, mode: bool) { 
        self.training = mode;
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::{prelude::{Ellipsis, IndexOp}, Dtype};
    
    use super::*;

    #[test]
    fn test_instance_norm() {
        mlx_rs::random::seed(435);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(), 
            0.5000646114349365,
            abs <= 0.01000129222869873
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(), 
            128.01654052734375,
            abs <= 2.560330810546875
        );

        let result = InstanceNorm::new(8).unwrap().forward(&a).unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(), 
            0.10645411163568497,
            abs <= 0.0021290822327136995
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(), 
            1.7032657861709595,
            abs <= 0.03406531572341919
        );
    }

    #[test]
    fn test_layer_norm() {
        mlx_rs::random::seed(635);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(), 
            0.4926903247833252,
            abs <= 0.009853806495666504
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(), 
            126.12872314453125,
            abs <= 2.522574462890625
        );

        let result = LayerNorm::new(16).unwrap().forward(&a).unwrap()
            .index((Ellipsis, 0));
        assert_eq!(result.shape(), &[2, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(), 
            0.2909903824329376,
            abs <= 0.005819807648658752
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(), 
            4.655846118927002,
            abs <= 0.09311692237854004
        );
    }

    #[test]
    fn test_rms_norm() {
        mlx_rs::random::seed(103);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(), 
            0.5054763555526733,
            abs <= 0.010109527111053467
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(), 
            129.40194702148438,
            abs <= 2.5880389404296875
        );

        let result = RmsNorm::new(16).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(), 
            0.8729387521743774,
            abs <= 0.01745877504348755
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(), 
            223.47232055664062,
            abs <= 4.469446411132813
        );
    }

    #[test]
    fn test_group_norm() {
        mlx_rs::random::seed(855);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(), 
            0.48666587471961975,
            abs <= 0.009733317494392395
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(), 
            124.58646392822266,
            abs <= 2.491729278564453
        );

        let result = GroupNorm::new(4, 16).unwrap().forward(&a).unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(), 
            -0.054606519639492035,
            abs <= 0.0010921303927898408
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(), 
            -0.8737043142318726,
            abs <= 0.017474086284637452
        );
    }

    #[test]
    fn test_batch_norm() {
        mlx_rs::random::seed(266);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(), 
            0.5058146715164185,
            abs <= 0.010116293430328369
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(), 
            129.48855590820312,
            abs <= 2.5897711181640624
        );

        let result = BatchNorm::new(16).unwrap().forward(&a).unwrap()
            .index((0, 0));
        assert_eq!(result.shape(), &[16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(), 
            0.4397852420806885,
            abs <= 0.00879570484161377
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(), 
            7.036563873291016,
            abs <= 0.14073127746582031
        );
    }
}