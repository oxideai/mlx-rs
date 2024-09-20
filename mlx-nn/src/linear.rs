use std::iter::once;

use mlx_rs::{error::Exception, Array};

use crate::{
    error::Error,
    macros::ModuleParameters,
    module::{Module, Param},
    utils::WithBias,
};

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
    /// Creates a new [`Linear`] layer.
    pub fn new(input_dims: i32, output_dims: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / (input_dims as f32));
        let weight =
            mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

        let bias = WithBias::default()
            .map_into_option(|| {
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)
            })
            .transpose()?;

        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Creates a new [`Linear`] layer with the given weight and bias.
    pub fn new_with_weight_and_bias(weight: Array, bias: Option<Array>) -> Self {
        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
        }
    }

    /// Sets the bias of the linear layer.
    pub fn with_bias(mut self, with_bias: impl Into<WithBias>) -> Result<Self, Exception> {
        self.bias.value = with_bias
            .into()
            .map_into_option(|| {
                let shape = self.weight.shape();
                let output_dims = shape[0];
                let input_dims = shape[1];
                let scale = f32::sqrt(1.0 / (input_dims as f32));
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)
            })
            .transpose()?;
        Ok(self)
    }

    /// Returns the shape of the linear layer.
    pub fn shape(&self) -> (i32, i32) {
        let weight_shape = self.weight.as_ref().shape();
        (weight_shape[0], weight_shape[1])
    }
}

impl Module for Linear {
    type Error = Error;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        match &self.bias.value {
            Some(bias) => {
                mlx_rs::ops::addmm(bias, x, self.weight.value.t(), None, None).map_err(Into::into)
            }
            None => mlx_rs::ops::matmul(x, &self.weight.value.t()).map_err(Into::into),
        }
    }

    fn training_mode(&mut self, _: bool) {}
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
    /// Creates a new [`Bilinear`] layer.
    pub fn new(input_dims_1: i32, input_dims_2: i32, output_dims: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / (input_dims_1 as f32));
        let weights = mlx_rs::random::uniform::<_, f32>(
            -scale,
            scale,
            &[output_dims, input_dims_2, input_dims_1],
            None,
        )?;

        let bias = WithBias::default()
            .map_into_option(|| {
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)
            })
            .transpose()?;

        Ok(Self {
            weights: Param::new(weights),
            bias: Param::new(bias),
        })
    }

    /// Sets the bias of the bilinear layer.
    pub fn with_bias(mut self, with_bias: impl Into<WithBias>) -> Result<Self, Exception> {
        self.bias.value = with_bias
            .into()
            .map_into_option(|| {
                let shape = self.weights.shape();
                let output_dims = shape[0];
                let input_dims_1 = shape[2];
                let scale = f32::sqrt(1.0 / (input_dims_1 as f32));
                mlx_rs::random::uniform::<_, f32>(-scale, scale, &[output_dims], None)
            })
            .transpose()?;
        Ok(self)
    }
}

impl Module for Bilinear {
    type Error = Error;

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
