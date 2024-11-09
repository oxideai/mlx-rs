use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::{
    error::Exception,
    module::{Module, Param},
    ops::{conv_transposed1d, zeros},
    random::uniform,
    Array,
};

generate_builder! {
    /// Applies a 1-dimensional transposed convolution over the multi-channel input sequence.
    #[derive(Debug, Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = false)]
    pub struct ConvTranspose1d {
        /// The learnable weight.
        #[param]
        pub weight: Param<Array>,

        /// The learnable bias that can be added to the output. Enabled by default.
        #[param]
        #[optional(ty = bool)]
        pub bias: Param<Option<Array>>,

        /// Padding. Default to [`ConvTranspose1d::DEFAULT_PADDING`].
        #[optional]
        pub padding: i32,

        /// Stride. Default to [`ConvTranspose1d::DEFAULT_STRIDE`].
        #[optional]
        pub stride: i32,
    }
}

impl ConvTranspose1dBuilder {
    /// Build a new `ConvTranspose1d` module.
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

impl ConvTranspose1d {
    /// Default to enable bias.
    pub const DEFAULT_BIAS: bool = true;

    /// Default padding.
    pub const DEFAULT_PADDING: i32 = 0;

    /// Default stride.
    pub const DEFAULT_STRIDE: i32 = 1;

    /// Create a new `ConvTranspose1d` module.
    pub fn new(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
    ) -> Result<ConvTranspose1d, Exception> {
        ConvTranspose1dBuilder::new().build(input_channels, output_channels, kernel_size)
    }
}

impl Module for ConvTranspose1d {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let mut y = conv_transposed1d(
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

    fn training_mode(&mut self, _mode: bool) {}
}
