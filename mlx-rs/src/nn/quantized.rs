use std::iter::once;

use crate::{
    array,
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{dequantize, quantize, quantized_matmul, zeros},
    prelude::IndexOp,
    random::uniform,
    Array,
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::nn::{Embedding, Linear};

/// Builder for [`QuantizedEmbedding`]
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_quantized_embedding,
    err = Exception,
)]
pub struct QuantizedEmbeddingBuilder {
    /// How many possible discrete tokens can we embed. Usually called the vocabulary size.
    pub embedding_count: i32,

    /// The dimensionality of the embeddings.
    pub dimensions: i32,

    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    #[builder(optional, default = QuantizedEmbedding::DEFAULT_GROUP_SIZE)]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    #[builder(optional, default = QuantizedEmbedding::DEFAULT_BITS)]
    pub bits: i32,
}

/// The same as ``Embedding`` but with a quantized weight matrix.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct QuantizedEmbedding {
    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub bits: i32,

    /// Scales
    pub scales: Param<Array>,

    /// Biases
    pub biases: Param<Array>,

    /// Inner embedding
    pub inner: Embedding,
}

impl QuantizedEmbeddingBuilder {
    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing [`Embedding`]
    pub fn build_with_embedding(
        self,
        embedding: Embedding,
    ) -> Result<QuantizedEmbedding, Exception> {
        let weight = embedding.weight.value;
        self.build_with_weight(weight)
    }

    /// Convenience method to build a new [`QuantizedEmbedding`] with an existing weight matrix
    pub fn build_with_weight(self, weight: Array) -> Result<QuantizedEmbedding, Exception> {
        let group_size = self.group_size;
        let bits = self.bits;

        let (quantized_weight, scales, biases) = quantize(&weight, group_size, bits)?;

        let inner = Embedding {
            weight: Param::new(quantized_weight),
        };

        Ok(QuantizedEmbedding {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        })
    }
}

fn build_quantized_embedding(
    builder: QuantizedEmbeddingBuilder,
) -> Result<QuantizedEmbedding, Exception> {
    let embedding_count = builder.embedding_count;
    let dims = builder.dimensions;

    let scale = array!(f32::sqrt(1.0 / (dims as f32)));
    // SAFETY: This is safe because the array scale is a single element array
    let weight = crate::random::normal::<f32>(&[embedding_count, dims], None, None, None)? * &scale;

    builder.build_with_weight(weight)
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(&self, x: impl AsRef<Array>) -> Result<Array, Exception> {
        quantized_matmul(
            x.as_ref(),
            &self.inner.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

impl Module<&Array> for QuantizedEmbedding {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let s = x.shape();
        let x = x.flatten(None, None)?;
        let w = self.inner.weight.index(&x);
        let scales = self.scales.index(&x);

        let out = dequantize(&w, &scales, &self.biases, self.group_size, self.bits)?;

        let ret_shape = s.iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

/// Builder for [`QuantizedLinear`]
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_quantized_linear,
    err = Exception,
)]
pub struct QuantizedLinearBuilder {
    /// The dimensionality of the input features.
    pub input_dims: i32,

    /// The dimensionality of the output features.
    pub output_dims: i32,

    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    #[builder(optional, default = QuantizedLinear::DEFAULT_GROUP_SIZE)]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    #[builder(optional, default = QuantizedLinear::DEFAULT_BITS)]
    pub bits: i32,

    /// Whether the linear layer has a bias. Default to [`Linear::DEFAULT_BIAS`]
    #[builder(optional, default = Linear::DEFAULT_BIAS)]
    pub bias: bool,
}

impl QuantizedLinearBuilder {
    /// Convenience method to build a new [`QuantizedLinear`] with an existing [`Linear`]
    pub fn build_with_linear(self, other: Linear) -> Result<QuantizedLinear, Exception> {
        self.build_with_weight_and_bias(other.weight.value, other.bias.value)
    }

    fn build_with_weight_and_bias(
        self,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<QuantizedLinear, Exception> {
        let (quantized_weight, scales, biases) = quantize(&weight, self.group_size, self.bits)?;

        let inner = Linear {
            weight: Param::new(quantized_weight),
            bias: Param::new(bias),
        };

        let mut linear = QuantizedLinear {
            group_size: self.group_size,
            bits: self.bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        };

        // Freeze all parameters
        linear.freeze_parameters(true);

        Ok(linear)
    }
}

/// Builds a new [`QuantizedLinear`]
pub fn build_quantized_linear(
    builder: QuantizedLinearBuilder,
) -> Result<QuantizedLinear, Exception> {
    let input_dims = builder.input_dims;
    let output_dims = builder.output_dims;
    let scale = f32::sqrt(1.0 / (input_dims as f32));
    let weight = uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

    let bias = if builder.bias {
        Some(zeros::<f32>(&[output_dims])?)
    } else {
        None
    };

    builder.build_with_weight_and_bias(weight, bias)
}

/// Applies an affine transformation to the input using a quantized weight matrix.
///
/// It is the quantized equivalent of [`Linear`].  For now its
/// parameters are frozen and will not be included in any gradient computation
/// but this will probably change in the future.
///
/// QuantizedLinear also provides several useful static to convert linear
/// layers to QuantizedLinear layers.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct QuantizedLinear {
    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub bits: i32,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Array>,

    /// Inner linear layer
    #[param]
    pub inner: Linear,
}

impl QuantizedLinear {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;
}

impl Module<&Array> for QuantizedLinear {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = quantized_matmul(
            x,
            &self.inner.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        if let Some(bias) = &self.inner.bias.value {
            x = x.add(bias)?;
        }
        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}
