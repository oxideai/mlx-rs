use mlx_macros::ModuleParameters;
use mlx_rs::{array, error::Exception, module::{Module, Param}, ops::{ones, rsqrt, zeros}, Array};

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

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let reduction_axes = (0..x.ndim() as i32 - 1).collect::<Vec<_>>();

        // Compute stats
        let mean = x.mean(&reduction_axes, true)?;
        let variance = x.variance(&reduction_axes, true, None)?;

        // Normalize
        let x = x.subtract(mean)?
            .multiply(rsqrt(&variance.add(&self.eps)?))?;

        if let (Some(weight), Some(bias)) = (self.weight.as_ref(), self.bias.as_ref()) {
            Ok(weight.multiply(x)?.add(bias)?)
        } else {
            Ok(x)
        }
    }

    fn training_mode(&mut self, _mode: bool) { }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::{prelude::IndexOp, Dtype};
    
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
}