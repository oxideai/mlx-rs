use mlx_macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::{
    error::Exception,
    ops::{conv_transpose1d, conv_transpose2d, conv_transpose3d, zeros},
    random::uniform,
    Array,
};

use crate::utils::{IntOrPair, IntOrTriple};

/// Builder for the `ConvTranspose1d` module.
#[derive(Debug, Clone, Default)]
pub struct ConvTranspose1dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose1d::DEFAULT_BIAS`] if not
    /// specified.
    pub bias: Option<bool>,

    /// Padding. Default to [`ConvTranspose1d::DEFAULT_PADDING`] if not specified.
    pub padding: Option<i32>,

    /// Stride. Default to [`ConvTranspose1d::DEFAULT_STRIDE`] if not specified.
    pub stride: Option<i32>,
}

impl ConvTranspose1dBuilder {
    /// Creates a new `ConvTranspose1dBuilder`.
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

    /// Builds a new `ConvTranspose1d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<ConvTranspose1d, Exception> {
        let bias = self.bias.unwrap_or(ConvTranspose1d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(ConvTranspose1d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(ConvTranspose1d::DEFAULT_STRIDE);

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

        Ok(ConvTranspose1d {
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
pub struct ConvTranspose1d {
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

impl ConvTranspose1d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: i32 = 0;

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: i32 = 1;

    /// Creates a new `ConvTranspose1dBuilder`.
    pub fn builder() -> ConvTranspose1dBuilder {
        ConvTranspose1dBuilder::new()
    }

    /// Creates a new ConvTranspose1d module with all optional parameters set to their default values.
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        ConvTranspose1dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for ConvTranspose1d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose1d(
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

/// Builder for the `ConvTranspose2d` module.
#[derive(Debug, Clone, Default)]
pub struct ConvTranspose2dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose2d::DEFAULT_BIAS`] if not
    /// specified.
    bias: Option<bool>,

    /// Padding. Default to [`ConvTranspose2d::DEFAULT_PADDING`] if not specified.
    padding: Option<(i32, i32)>,

    /// Stride. Default to [`ConvTranspose2d::DEFAULT_STRIDE`] if not specified.
    stride: Option<(i32, i32)>,
}

impl ConvTranspose2dBuilder {
    /// Creates a new `ConvTranspose2dBuilder`.
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

    /// Builds a new `ConvTranspose2d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: (i32, i32),
    ) -> Result<ConvTranspose2d, Exception> {
        let bias = self.bias.unwrap_or(ConvTranspose2d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(ConvTranspose2d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(ConvTranspose2d::DEFAULT_STRIDE);

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

        Ok(ConvTranspose2d {
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
pub struct ConvTranspose2d {
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

impl ConvTranspose2d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: (i32, i32) = (0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: (i32, i32) = (1, 1);

    /// Creates a new `ConvTranspose2dBuilder`.
    pub fn builder() -> ConvTranspose2dBuilder {
        ConvTranspose2dBuilder::new()
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
        ConvTranspose2dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for ConvTranspose2d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose2d(
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

/// Builder for the `ConvTranspose3d` module.
#[derive(Debug, Clone, Default)]
pub struct ConvTranspose3dBuilder {
    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose3d::DEFAULT_BIAS`] if not
    /// specified.
    bias: Option<bool>,

    /// Padding. Default to [`ConvTranspose3d::DEFAULT_PADDING`] if not specified.
    padding: Option<(i32, i32, i32)>,

    /// Stride. Default to [`ConvTranspose3d::DEFAULT_STRIDE`] if not specified.
    stride: Option<(i32, i32, i32)>,
}

impl ConvTranspose3dBuilder {
    /// Creates a new `ConvTranspose3dBuilder`.
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

    /// Builds a new `ConvTranspose3d` module.
    pub fn build(
        self,
        input_channels: i32,
        output_channels: i32,
        kernel_size: (i32, i32, i32),
    ) -> Result<ConvTranspose3d, Exception> {
        let bias = self.bias.unwrap_or(ConvTranspose3d::DEFAULT_BIAS);
        let padding = self.padding.unwrap_or(ConvTranspose3d::DEFAULT_PADDING);
        let stride = self.stride.unwrap_or(ConvTranspose3d::DEFAULT_STRIDE);

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

        Ok(ConvTranspose3d {
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
pub struct ConvTranspose3d {
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

impl ConvTranspose3d {
    /// Default value for `bias` if not specified.
    pub const DEFAULT_BIAS: bool = true;

    /// Default value for `padding` if not specified.
    pub const DEFAULT_PADDING: (i32, i32, i32) = (0, 0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: (i32, i32, i32) = (1, 1, 1);

    /// Creates a new `ConvTranspose3dBuilder`.
    pub fn builder() -> ConvTranspose3dBuilder {
        ConvTranspose3dBuilder::new()
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
        ConvTranspose3dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for ConvTranspose3d {
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transpose3d(
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
