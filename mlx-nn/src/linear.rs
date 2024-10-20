use std::iter::once;

use mlx_rs::{error::Exception, Array};

use crate::{
    macros::ModuleParameters,
    module::{Module, Param},
};

/// Builder for [`Linear`] module
#[derive(Debug, Clone, Default)]
pub struct LinearBuilder {
    /// Whether to include bias in the linear layer. Default to [`Linear::DEFAULT_WITH_BIAS`].
    pub with_bias: Option<bool>,
}

impl LinearBuilder {
    /// Creates a new [`LinearBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `with_bias` field.
    pub fn with_bias(mut self, with_bias: impl Into<Option<bool>>) -> Self {
        self.with_bias = with_bias.into();
        self
    }

    /// Builds a new [`Linear`] layer.
    pub fn build(self, input_dims: i32, output_dims: i32) -> Result<Linear, Exception> {
        let with_bias = self.with_bias.unwrap_or(Linear::DEFAULT_WITH_BIAS);

        let scale = f32::sqrt(1.0 / (input_dims as f32));
        let weight =
            mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

        let bias = if with_bias {
            Some(
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)?
            )
        } else {
            None
        };

        Ok(Linear {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }
}

/// Applies an affine transformation to the input.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Linear {
    /// The weight of the linear layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the linear layer.
    #[param]
    pub bias: Param<Option<Array>>,
}

impl Linear {
    /// Default value for `with_bias`
    pub const DEFAULT_WITH_BIAS: bool = true;

    /// Creates a new [`LinearBuilder`].
    pub fn builder() -> LinearBuilder {
        LinearBuilder::new()
    }

    /// Creates a new [`Linear`] layer.
    pub fn new(input_dims: i32, output_dims: i32) -> Result<Self, Exception> {
        LinearBuilder::new().build(input_dims, output_dims)
    }

    /// Returns the shape of the linear layer.
    pub fn shape(&self) -> (i32, i32) {
        let weight_shape = self.weight.as_ref().shape();
        (weight_shape[0], weight_shape[1])
    }
}

impl Module for Linear {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        match &self.bias.value {
            Some(bias) => mlx_rs::ops::addmm(bias, x, self.weight.value.t(), None, None),
            None => mlx_rs::ops::matmul(x, &self.weight.value.t()),
        }
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for [`Bilinear`] module
#[derive(Debug, Clone, Default)]
pub struct BilinearBuilder {
    /// Whether to include bias in the bilinear layer. Default to [Bilinear::DEFAULT_WITH_BIAS].
    with_bias: Option<bool>,
}

impl BilinearBuilder {
    /// Creates a new [`BilinearBuilder`].
    pub fn new() -> Self {
        Self { with_bias: None }
    }

    /// Sets the `with_bias` field.
    pub fn with_bias(mut self, with_bias: impl Into<Option<bool>>) -> Self {
        self.with_bias = with_bias.into();
        self
    }

    /// Builds a new [`Bilinear`] layer.
    pub fn build(self, input_dims_1: i32, input_dims_2: i32, output_dims: i32) -> Result<Bilinear, Exception> {
        let with_bias = self.with_bias.unwrap_or(Bilinear::DEFAULT_WITH_BIAS);

        let scale = f32::sqrt(1.0 / (input_dims_1 as f32));
        let weights = mlx_rs::random::uniform::<_, f32>(
            -scale,
            scale,
            &[output_dims, input_dims_2, input_dims_1],
            None,
        )?;

        let bias = if with_bias {
            Some(
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)?
            )
        } else {
            None
        };

        Ok(Bilinear {
            weights: Param::new(weights),
            bias: Param::new(bias),
        })
    }
}

/// Applies a bilinear transformation to the inputs.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Bilinear {
    /// The weight of the bilinear layer.
    #[param]
    pub weights: Param<Array>,

    /// The bias of the bilinear layer.
    #[param]
    pub bias: Param<Option<Array>>,
}

impl Bilinear {
    /// Default value for `with_bias`
    pub const DEFAULT_WITH_BIAS: bool = true;

    /// Creates a new [`BilinearBuilder`].
    pub fn builder() -> BilinearBuilder {
        BilinearBuilder::new()
    }

    /// Creates a new [`Bilinear`] layer.
    pub fn new(input_dims_1: i32, input_dims_2: i32, output_dims: i32) -> Result<Self, Exception> {
        BilinearBuilder::new().build(input_dims_1, input_dims_2, output_dims)
    }
}

impl Module for Bilinear {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let shape = self.weights.shape();
        let (out, in2, in1) = (shape[0], shape[1], shape[2]);
        let x_shape = &x.shape()[..x.shape().len() - 1];
        let x1 = x.reshape(&[-1, in1])?;
        let x2 = x.reshape(&[-1, 1, in2])?;

        // perform the bilinear transform
        let w = self.weights.reshape(&[out * in2, in1])?;
        let mut y = mlx_rs::ops::matmul(&x1, &w.t())?;
        y = y.reshape(&[-1, out, in2])?.swap_axes(-2, -1)?;
        y = mlx_rs::ops::matmul(&x2, &y)?;
        y = y.squeeze(&[1])?;

        // reset the shape
        let new_shape = x_shape.iter().cloned().chain(once(out)).collect::<Vec<_>>();
        y = y.reshape(&new_shape)?;

        if let Some(bias) = &self.bias.value {
            y = mlx_rs::ops::add(&y, bias)?;
        }

        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

// The following tests are ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::{random::uniform, Dtype};

    use super::*;

    #[test]
    fn test_linear() {
        mlx_rs::random::seed(744);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.508_688_57,
            abs <= 0.010_173_771_5
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            130.224_27,
            abs <= 2.604_485_5
        );
        let result = Linear::new(16, 5).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 5]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.104_193_09,
            abs <= 0.002_083_861_7
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            8.335_447,
            abs <= 0.166_708_95
        );
    }
}
