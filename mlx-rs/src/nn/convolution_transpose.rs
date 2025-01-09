use crate::module::{Module, Param};
use crate::{
    error::Exception,
    ops::{conv_transpose1d, conv_transpose2d, conv_transpose3d, zeros},
    random::uniform,
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::utils::{SingleOrPair, SingleOrTriple};

/// Builder for the `ConvTranspose1d` module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_1d,
    err = Exception,
)]
pub struct ConvTranspose1dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: i32,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose1d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Padding. Default to [`ConvTranspose1d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_PADDING)]
    pub padding: i32,

    /// Stride. Default to [`ConvTranspose1d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose1d::DEFAULT_STRIDE)]
    pub stride: i32,
}

fn build_conv_transpose_1d(builder: ConvTranspose1dBuilder) -> Result<ConvTranspose1d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size = builder.kernel_size;

    let bias = builder.bias;
    let padding = builder.padding;
    let stride = builder.stride;

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

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
///
/// The channels are expected to be last i.e. the input shape should be `NLC` where:
///
/// - `N` is the batch dimension
/// - `L` is the sequence length
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
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
}

impl<'a> Module<'a> for ConvTranspose1d {
    type Input = &'a Array;
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &'a Array) -> Result<Array, Self::Error> {
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
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_2d,
    err = Exception,
)]
pub struct ConvTranspose2dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: SingleOrPair<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose2d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_BIAS)]
    bias: bool,

    /// Padding. Default to [`ConvTranspose2d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_PADDING)]
    padding: SingleOrPair<i32>,

    /// Stride. Default to [`ConvTranspose2d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose2d::DEFAULT_STRIDE)]
    stride: SingleOrPair<i32>,
}

fn build_conv_transpose_2d(builder: ConvTranspose2dBuilder) -> Result<ConvTranspose2d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32) = builder.kernel_size.into();

    let bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();

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

/// Applies a 2-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
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
    pub const DEFAULT_PADDING: SingleOrPair<i32> = SingleOrPair::Pair(0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrPair<i32> = SingleOrPair::Pair(1, 1);
}

impl<'a> Module<'a> for ConvTranspose2d {
    type Input = &'a Array;
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &'a Array) -> Result<Array, Self::Error> {
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
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_conv_transpose_3d,
    err = Exception,
)]
pub struct ConvTranspose3dBuilder {
    /// The number of input channels.
    pub input_channels: i32,

    /// The number of output channels.
    pub output_channels: i32,

    /// The size of the convolution filters.
    pub kernel_size: SingleOrTriple<i32>,

    /// If `true`, add a learnable bias to the output. Default to [`ConvTranspose3d::DEFAULT_BIAS`] if not
    /// specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_BIAS)]
    pub bias: bool,

    /// Padding. Default to [`ConvTranspose3d::DEFAULT_PADDING`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_PADDING)]
    pub padding: SingleOrTriple<i32>,

    /// Stride. Default to [`ConvTranspose3d::DEFAULT_STRIDE`] if not specified.
    #[builder(optional, default = ConvTranspose3d::DEFAULT_STRIDE)]
    pub stride: SingleOrTriple<i32>,
}

fn build_conv_transpose_3d(builder: ConvTranspose3dBuilder) -> Result<ConvTranspose3d, Exception> {
    let input_channels = builder.input_channels;
    let output_channels = builder.output_channels;
    let kernel_size: (i32, i32, i32) = builder.kernel_size.into();

    let bias = builder.bias;
    let padding = builder.padding.into();
    let stride = builder.stride.into();

    let scale =
        f32::sqrt(1.0 / (input_channels * kernel_size.0 * kernel_size.1 * kernel_size.2) as f32);
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

/// Applies a 3-dimensional convolution over the multi-channel input image.
///
/// The channels are expected to be last i.e. the input shape should be `NHWC` where:
///
/// - `N` is the batch dimension
/// - `H` is the input image height
/// - `W` is the input image width
/// - `C` is the number of input channels
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
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
    pub const DEFAULT_PADDING: SingleOrTriple<i32> = SingleOrTriple::Triple(0, 0, 0);

    /// Default value for `stride` if not specified.
    pub const DEFAULT_STRIDE: SingleOrTriple<i32> = SingleOrTriple::Triple(1, 1, 1);
}

impl<'a> Module<'a> for ConvTranspose3d {
    type Input = &'a Array;
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &'a Array) -> Result<Array, Self::Error> {
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
