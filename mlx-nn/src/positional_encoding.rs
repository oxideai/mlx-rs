use mlx_internal_macros::{generate_builder, Buildable, Builder};
use mlx_macros::ModuleParameters;
use mlx_rs::{array, error::Exception, module::{Module, Param}, ops::{concatenate, exp, log}, Array};

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

/// Type alias for [`SinusoidalPositionalEncoding`].
pub type Sinpe = SinusoidalPositionalEncoding;

/// Type alias for [`SinusoidalPositionalEncodingBuilder`].
pub type SinpeBuilder = SinusoidalPositionalEncodingBuilder;

/// Implements sinusoidal positional encoding.
///
/// For more details see the paper "Attention Is All You Need"
/// <https://arxiv.org/abs/1706.03762>.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
pub struct SinusoidalPositionalEncoding {
    #[param]
    sigmas: Param<Array>,

    /// multiplicative scale for the embeddings.  Default is `sqrt(2/dimensions)`
    pub scale: f32,

    /// if `true` embed using `[cos(x), sin(x)]` instead of the reverse
    pub cosine_first: bool,
}

impl Sinpe {
    /// Default value for `cosine_first` field.
    pub const DEFAULT_COSINE_FIRST: bool = false;

    /// Default value for min frequency.
    pub const DEFAULT_MIN_FREQUENCY: f32 = 0.0001;

    /// Default value for max frequency.
    pub const DEFAULT_MAX_FREQUENCY: f32 = 1.0;

    /// Default value for full turns.
    pub const DEFAULT_FULL_TURNS: bool = false;
}

/// Builder for [`SinusoidalPositionalEncoding`].
#[derive(Debug, Clone, Builder)]
#[builder(
    build_with = build_sinpe,
    err = Exception,
)]
pub struct SinusoidalPositionalEncodingBuilder {
    dimensions: i32,

    #[builder(optional, default = Sinpe::DEFAULT_MIN_FREQUENCY)]
    min_frequency: f32,

    #[builder(optional, default = Sinpe::DEFAULT_MAX_FREQUENCY)]
    max_frequency: f32,

    #[builder(optional, default = None)]
    scale: Option<f32>,

    #[builder(optional, default = Sinpe::DEFAULT_COSINE_FIRST)]
    cosine_first: bool,

    #[builder(optional, default = Sinpe::DEFAULT_FULL_TURNS)]
    full_turns: bool,
}

fn build_sinpe(builder: SinpeBuilder) -> Result<SinusoidalPositionalEncoding, Exception> {
    let SinpeBuilder {
        dimensions,
        min_frequency,
        max_frequency,
        scale,
        cosine_first,
        full_turns,
    } = builder;

    let half_dim = dimensions / 2;
    let one_zero = array!(1.0)
        .subtract(
            Array::from_iter(0..half_dim, &[half_dim])
                .divide(array!(half_dim - 1))?
        )?;
    let min_frequency = log(array!(min_frequency))?;
    let max_frequency = log(array!(max_frequency))?;

    // SAFETY: max_frequency and min_frequency are scalars and operations with scalars won't throw
    let mut sigmas = exp(
        &one_zero * (&max_frequency - &min_frequency) + &min_frequency
    )?;
    if full_turns {
        // SAFETY: scalar array operation won't throw
        sigmas = sigmas * array!(2.0 * std::f32::consts::PI);
    }

    let scale = scale.unwrap_or_else(|| (2.0 / dimensions as f32).sqrt());

    Ok(SinusoidalPositionalEncoding {
        sigmas: Param::new(sigmas),
        scale,
        cosine_first,
    })
}

impl Module<&Array> for Sinpe {
    type Error = Exception;

    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let mut y = x.expand_dims(&[-1])
            .and_then(|x| x.multiply(&self.sigmas))?;

        let cosy = y.cos()?;
        let siny = y.sin()?;

        if self.cosine_first {
            y = concatenate(&[cosy, siny], -1)?;
        } else {
            y = concatenate(&[siny, cosy], -1)?;
        }

        if self.scale != 1.0 {
            // SAFETY: multiplication with scalar won't throw
            y = y * self.scale;
        }

        Ok(y)
    }

    fn training_mode(&mut self, _mode: bool) { }
}
