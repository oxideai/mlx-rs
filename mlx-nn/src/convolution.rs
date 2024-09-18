use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{
    error::Exception,
    ops::{conv1d, conv2d, zeros},
    random::uniform,
    Array,
};

use crate::utils::{IntOrPair, IntOrTriple, WithBias};

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
    /// Creates a new 1-dimensional convolution layer.
    ///
    /// # Params
    ///
    /// - `input_channels`: number of input channels
    /// - `output_channels`: number of output channels
    /// - `kernel_size`: size of the convolution filters
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0f32 / (input_channels * kernel_size) as f32);
        let weight = uniform::<_, f32>(
            -scale,
            scale,
            &[output_channels, kernel_size, input_channels],
            None,
        )?;
        let default_bias = WithBias::default()
            .map_into_option(|| zeros::<f32>(&[output_channels]))
            .transpose()?;
        let default_padding = 0;
        let default_stride = 1;

        let conv1d = Self {
            weight: Param::new(weight),
            bias: Param::new(default_bias),
            padding: default_padding,
            stride: default_stride,
        };
        conv1d.with_bias(WithBias::default())
    }

    /// If `true`, add a learnable bias to the output
    pub fn with_bias(mut self, with_bias: impl Into<WithBias>) -> Result<Self, Exception> {
        let bias = with_bias
            .into()
            .map_into_option(|| zeros::<f32>(&[self.weight.shape()[0]]))
            .transpose()?;
        self.bias.value = bias;
        Ok(self)
    }

    /// Sets the stride when applying the filter
    pub fn with_stride(mut self, stride: i32) -> Self {
        self.stride = stride;
        self
    }

    /// Sets the padding when applying the filter
    pub fn with_padding(mut self, padding: i32) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
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
        let scale = f32::sqrt(1.0 / (input_channels * kernel_size.0 * kernel_size.1) as f32);

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
        let default_bias = WithBias::default()
            .map_into_option(|| zeros::<f32>(&[output_channels]))
            .transpose()?;
        let default_padding = (0, 0);
        let default_stride = (1, 1);

        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(default_bias),
            padding: default_padding,
            stride: default_stride,
        })
    }

    /// If `true`, add a learnable bias to the output
    pub fn with_bias(mut self, with_bias: impl Into<WithBias>) -> Result<Self, Exception> {
        let bias = with_bias
            .into()
            .map_into_option(|| {
                let output_channels = self.weight.shape()[0];
                zeros::<f32>(&[output_channels])
            })
            .transpose()?;
        self.bias.value = bias;
        Ok(self)
    }

    /// Sets the stride when applying the filter
    pub fn with_stride(mut self, stride: impl IntOrPair) -> Self {
        self.stride = stride.into_pair();
        self
    }

    /// Sets the padding when applying the filter
    pub fn with_padding(mut self, padding: impl IntOrPair) -> Self {
        self.padding = padding.into_pair();
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
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
        let default_bias = WithBias::default()
            .map_into_option(|| zeros::<f32>(&[output_channels]))
            .transpose()?;
        let default_padding = (0, 0, 0);
        let default_stride = (1, 1, 1);

        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(default_bias),
            padding: default_padding,
            stride: default_stride,
        })
    }

    /// If `true`, add a learnable bias to the output
    pub fn with_bias(mut self, with_bias: impl Into<WithBias>) -> Result<Self, Exception> {
        let bias = with_bias
            .into()
            .map_into_option(|| {
                let output_channels = self.weight.shape()[0];
                zeros::<f32>(&[output_channels])
            })
            .transpose()?;
        self.bias.value = bias;
        Ok(self)
    }

    /// Sets the stride when applying the filter
    pub fn with_stride(mut self, stride: impl IntOrTriple) -> Self {
        self.stride = stride.into_triple();
        self
    }

    /// Sets the padding when applying the filter
    pub fn with_padding(mut self, padding: impl IntOrTriple) -> Self {
        self.padding = padding.into_triple();
        self
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
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
