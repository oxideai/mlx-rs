use mlx_internal_macros::{generate_builder, Buildable};
use mlx_macros::ModuleParameters;
use mlx_rs::{error::Exception, module::Module, Array};

/// Type alias for [`RotaryPositionalEncoding`].
pub type Rope = RotaryPositionalEncoding;

/// Type alias for [`RotaryPositionalEncodingBuilder`].
pub type RopeBuilder = RotaryPositionalEncodingBuilder;

generate_builder! {
    /// Implements the rotary positional encoding.
    ///
    /// The traditional implementation rotates consecutive pairs of elements in the
    /// feature dimension while the default implementation rotates pairs with
    /// stride half the feature dimensions for efficiency.
    ///
    /// For more details see _RoFormer: Enhanced Transformer with Rotary Position
    /// Embedding_ ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864))
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    pub struct RotaryPositionalEncoding {
        /// The feature dimensions to be rotated. If the input feature is larger
        /// than dims then the rest is left unchanged
        pub dimensions: i32,

        /// If `true` choose the traditional implementation which is slightly
        /// less efficient
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_TRADITIONAL)]
        pub traditional: bool,
        
        /// The base used to compute angular frequency for each dimension in the
        /// positional encodings
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_BASE)]
        pub base: f32,

        /// scale used to scale the positions
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_SCALE)]
        pub scale: f32,
    }
}

impl RotaryPositionalEncoding {
    /// Default value for `traditional` field.
    pub const DEFAULT_TRADITIONAL: bool = false;

    /// Default value for `base` field.
    pub const DEFAULT_BASE: f32 = 10_000.0;

    /// Default value for `scale` field.
    pub const DEFAULT_SCALE: f32 = 1.0;
}

generate_builder! {
    /// Input for the [`RotaryPositionalEncoding`] module.
    #[derive(Debug, Buildable, Clone)]
    pub struct RopeInput<'a> {
        /// The input tensor.
        pub x: &'a Array,
    
        /// Offset
        #[builder(optional, default = RopeInput::DEFAULT_OFFSET)]
        pub offset: i32,
    }
}

impl RopeInput<'_> {
    /// Default value for `offset` field.
    pub const DEFAULT_OFFSET: i32 = 0;
}

impl<'a> From<&'a Array> for RopeInput<'a> {
    fn from(x: &'a Array) -> Self {
        RopeInput {
            x,
            offset: Self::DEFAULT_OFFSET,
        }
    }
}

impl<'a> From<(&'a Array,)> for RopeInput<'a> {
    fn from((x,): (&'a Array,)) -> Self {
        RopeInput { x, offset: Self::DEFAULT_OFFSET }
    }
}

impl<'a> From<(&'a Array, i32)> for RopeInput<'a> {
    fn from((x, offset): (&'a Array, i32)) -> Self {
        RopeInput { x, offset }
    }
}

impl<'a, Input> Module<Input> for RotaryPositionalEncoding 
where
    Input: Into<RopeInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)])?;
        let x = mlx_rs::fast::rope(x, self.dimensions, self.traditional, self.base, self.scale, offset, None)?;
        x.reshape(shape)
    }

    fn training_mode(&mut self, _mode: bool) { }
}