use mlx_macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::{
    error::Exception,
    ops::{conv1d, conv2d, zeros},
    random::uniform,
    Array,
};

use crate::utils::{IntOrPair, IntOrTriple};

/// Builder for the `Conv1d` module.
#[derive(Debug, Clone, Default)]
pub struct Conv1dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`Conv1d::DEFAULT_BIAS`] if not
    /// specified.
    pub bias: Option<bool>,

    /// Padding. Default to [`Conv1d::DEFAULT_PADDING`] if not specified.
    pub padding: Option<i32>,

    /// Stride. Default to [`Conv1d::DEFAULT_STRIDE`] if not specified.
    pub stride: Option<i32>,
}

impl Conv1dBuilder {
    /// Creates a new `Conv1dBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `bias` parameter.
    pub fn bias(mut self, bias: impl Into<Option<bool>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Sets the `padding` parameter.
    pub fn padding(mut self, padding: impl Into<Option<i32>>) -> Self {
        self.padding = padding.into();
        self
    }

    /// Sets the `stride` parameter.
    pub fn stride(mut self, stride: impl Into<Option<i32>>) -> Self {
        self.stride = stride.into();
        self
    }

    /// Builds a new `Conv1d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<Conv1d, Exception> {
        let bias = self.bias.unwrap_or(Conv1d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(Conv1d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(Conv1d::DEFAULT_STRIDE);

        let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size) as f32);
        let weight = uniform::<_, f32>(
            -scale,
            scale,
            &[output_channels, kernel_size, input_channels],
            None,
        )?;
        let bias = if bias {
            Some(zeros::<f32>(&[output_channels])?)
        } else {
            None
        };

        Ok(Conv1d {
            weight: Param::new(weight),
            bias: Param::new(bias),
            padding,
            stride,
        })
    }
}

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
///
/// The channels are expected to be last i.e. the input shape should be `NLC` where:
///
/// - `N` is the batch dimension
/// - `L` is the sequence length
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters)]
pub struct Conv1d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to 0 if not specified.
    pub padding: i32,

    /// Stride. Default to 1 if not specified.
    pub stride: i32,
}

impl Conv1d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: i32 = 0;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: i32 = 1;

    /// Creates a new `Conv1dBuilder`.
    pub fn builder() -> Conv1dBuilder {
        Conv1dBuilder::new()
    }

    /// Creates a new Conv1d module with all optional parameters set to their default values.
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        Conv1dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for Conv1d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv1d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            None,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `Conv2d` module.
#[derive(Debug, Clone, Default)]
pub struct Conv2dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`Conv2d::DEFAULT_BIAS`] if not
    /// specified.
    bias: Option<bool>,

    /// Padding. Default to [`Conv2d::DEFAULT_PADDING`] if not specified.
    padding: Option<(i32, i32)>,

    /// Stride. Default to [`Conv2d::DEFAULT_STRIDE`] if not specified.
    stride: Option<(i32, i32)>,
}

impl Conv2dBuilder {
    /// Creates a new `Conv2dBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `bias` parameter.
    pub fn bias(mut self, bias: impl Into<Option<bool>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Sets the `padding` parameter.
    pub fn padding(mut self, padding: Option<impl IntOrPair>) -> Self {
        self.padding = padding.map(IntOrPair::into_pair);
        self
    }

    /// Sets the `stride` parameter.
    pub fn stride(mut self, stride: Option<impl IntOrPair>) -> Self {
        self.stride = stride.map(IntOrPair::into_pair);
        self
    }

    /// Builds a new `Conv2d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: (i32, i32),
    ) -> Result<Conv2d, Exception> {
        let bias = self.bias.unwrap_or(Conv2d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(Conv2d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(Conv2d::DEFAULT_STRIDE);

        let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size.0 * kernel_size.1) as f32);
        let weight = uniform::<_, f32>(
            -scale,
            scale,
            &[
                output_channels,
                kernel_size.0,
                kernel_size.1,
                input_channels,
            ],
            None,
        )?;
        let bias = if bias {
            Some(zeros::<f32>(&[output_channels])?)
        } else {
            None
        };

        Ok(Conv2d {
            weight: Param::new(weight),
            bias: Param::new(bias),
            padding,
            stride,
        })
    }
}

/// Applies a 2-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters)]
pub struct Conv2d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to `(0, 0)` if not specified.
    pub padding: (i32, i32),

    /// Stride. Default to `(1, 1)` if not specified.
    pub stride: (i32, i32),
}

impl Conv2d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: (i32, i32) = (0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: (i32, i32) = (1, 1);

    /// Creates a new `Conv2dBuilder`.
    pub fn builder() -> Conv2dBuilder {
        Conv2dBuilder::new()
    }

    /// Creates a new 2-dimensional convolution layer.
    ///
    /// # Params
    ///
    /// - `input_channels`: number of input channels
    /// - `output_channels`: number of output channels
    /// - `kernel_size`: size of the convolution filters
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: impl IntOrPair,
    ) -> Result<Self, Exception> {
        let kernel_size = kernel_size.into_pair();
        Conv2dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for Conv2d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv2d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            None,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Builder for the `Conv3d` module.
#[derive(Debug, Clone, Default)]
pub struct Conv3dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`Conv3d::DEFAULT_BIAS`] if not
    /// specified.
    bias: Option<bool>,

    /// Padding. Default to [`Conv3d::DEFAULT_PADDING`] if not specified.
    padding: Option<(i32, i32, i32)>,

    /// Stride. Default to [`Conv3d::DEFAULT_STRIDE`] if not specified.
    stride: Option<(i32, i32, i32)>,
}

impl Conv3dBuilder {
    /// Creates a new `Conv3dBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `bias` parameter.
    pub fn bias(mut self, bias: impl Into<Option<bool>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Sets the `padding` parameter.
    pub fn padding(mut self, padding: Option<impl IntOrTriple>) -> Self {
        self.padding = padding.map(IntOrTriple::into_triple);
        self
    }

    /// Sets the `stride` parameter.
    pub fn stride(mut self, stride: Option<impl IntOrTriple>) -> Self {
        self.stride = stride.map(IntOrTriple::into_triple);
        self
    }

    /// Builds a new `Conv3d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: (i32, i32, i32),
    ) -> Result<Conv3d, Exception> {
        let bias = self.bias.unwrap_or(Conv3d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(Conv3d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(Conv3d::DEFAULT_STRIDE);

        let scale = f32::sqrt(
            1.0 / (input_channels * kernel_size.0 * kernel_size.1 * kernel_size.2) as f32,
        );
        let weight = uniform::<_, f32>(
            -scale,
            scale,
            &[
                output_channels,
                kernel_size.0,
                kernel_size.1,
                kernel_size.2,
                input_channels,
            ],
            None,
        )?;
        let bias = if bias {
            Some(zeros::<f32>(&[output_channels])?)
        } else {
            None
        };

        Ok(Conv3d {
            weight: Param::new(weight),
            bias: Param::new(bias),
            padding,
            stride,
        })
    }
}

/// Applies a 3-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters)]
pub struct Conv3d {
    /// The weight of the convolution layer.
    #[param]
    pub weight: Param<Array>,

    /// The bias of the convolution layer.
    #[param]
    pub bias: Param<Option<Array>>,

    /// Padding. Default to `(0, 0, 0)` if not specified.
    pub padding: (i32, i32, i32),

    /// Stride. Default to `(1, 1, 1)` if not specified.
    pub stride: (i32, i32, i32),
}

impl Conv3d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: (i32, i32, i32) = (0, 0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: (i32, i32, i32) = (1, 1, 1);

    /// Creates a new `Conv3dBuilder`.
    pub fn builder() -> Conv3dBuilder {
        Conv3dBuilder::new()
    }

    /// Creates a new 3-dimensional convolution layer.
    ///
    /// # Params
    ///
    /// - `input_channels`: number of input channels
    /// - `output_channels`: number of output channels
    /// - `kernel_size`: size of the convolution filters
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: impl IntOrTriple,
    ) -> Result<Self, Exception> {
        let kernel_size = kernel_size.into_triple();
        Conv3dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for Conv3d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = mlx_rs::ops::conv3d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            None,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            y += bias;
        }
        Ok(y)
    }

    fn training_mode(&mut self, _: bool) {}
}

// The following tests are ported from the swift bindings:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use mlx_rs::module::Module;
    use mlx_rs::{random::uniform, Dtype};

    use crate::Conv1d;

    #[test]
    fn test_conv1d() {
        mlx_rs::random::seed(819);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.512_987_5,
            abs <= 0.010_259_75
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            131.324_8,
            abs <= 2.626_496
        );
        let result = Conv1d::new(16, 2, 8).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 1, 2]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.264_865_2,
            abs <= 0.005_297_303_7
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            1.059_460_8,
            abs <= 0.021_189_215
        );
    }

    #[test]
    fn test_conv2d() {
        mlx_rs::random::seed(62);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 8, 4], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            0.522_504_27,
            abs <= 0.010_450_086
        );
        assert_float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            267.522_2,
            abs <= 5.350_444
        );
        let result = crate::Conv2d::new(4, 2, (8, 8))
            .unwrap()
            .forward(&a)
            .unwrap();
        assert_eq!(result.shape(), &[2, 1, 1, 2]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            -0.279_321_5,
            abs <= 0.005_586_43
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            -1.117_286,
            abs <= 0.022_345_72
        );
    }
}
