use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{
    error::Exception,
    ops::{conv1d, zeros},
    random::uniform,
    Array,
};

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
#[derive(ModuleParameters)]
pub struct Conv1d {
    output_channels: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub bias: Param<Option<Array>>,
    pub padding: Option<i32>,
    pub stride: Option<i32>,
}

impl Conv1d {
    /// Creates a new 1-dimensional convolution layer.
    ///
    /// # Arguments
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

        Ok(Self {
            output_channels,
            weight: Param::new(weight),
            bias: Param::new(None),
            padding: None,
            stride: None,
        })
    }

    /// Sets the stride when applying the filter
    ///
    /// Default to 1 if not specified (see [`conv1d`]).
    pub fn stride(mut self, stride: impl Into<Option<i32>>) -> Self {
        self.stride = stride.into();
        self
    }

    /// Sets the padding when applying the filter
    ///
    /// Default to 0 if not specified (see [`conv1d`]).
    pub fn padding(mut self, padding: impl Into<Option<i32>>) -> Self {
        self.padding = padding.into();
        self
    }

    /// If `true`, add  a learnable bias to the output
    pub fn bias(mut self, bias: bool) -> Result<Self, Exception> {
        if bias {
            self.bias.value = Some(zeros::<f32>(&[self.output_channels])?);
        } else {
            self.bias.value = None;
        }
        Ok(self)
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let y = conv1d(
            x,
            self.weight.as_ref(),
            self.stride,
            self.padding,
            None,
            None,
        )?;
        if let Some(bias) = &self.bias.value {
            Ok(y + bias)
        } else {
            Ok(y)
        }
    }
}
