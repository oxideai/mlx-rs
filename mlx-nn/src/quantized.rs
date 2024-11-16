use std::iter::once;

use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::{
    array,
    error::Exception,
    module::{Module, Param, Parameter},
    ops::{dequantize, quantize, quantized_matmul, zeros},
    prelude::IndexOp,
    random::uniform,
    Array,
};

use crate::{Embedding, Linear};

generate_builder! {
    /// The same as ``Embedding`` but with a quantized weight matrix.
    #[derive(Debug, Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = false)]
    pub struct QuantizedEmbedding {
        /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
        #[optional]
        pub group_size: i32,

        /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
        #[optional]
        pub bits: i32,

        /// Scales
        #[param]
        pub scales: Param<Array>,

        /// Biases
        #[param]
        pub biases: Param<Array>,

        /// Inner embedding
        #[param]
        pub inner: Param<Embedding>,
    }
}

impl QuantizedEmbeddingBuilder {
    /// Builds a new [`QuantizedEmbedding`]
    pub fn build(self, embedding_count: i32, dims: i32) -> Result<QuantizedEmbedding, Exception> {
        let scale = array!(f32::sqrt(1.0 / (dims as f32)));
        // SAFETY: This is safe because the array scale is a single element array
        let weight =
            mlx_rs::random::normal::<f32>(&[embedding_count, dims], None, None, None)? * &scale;

        self.build_with_weight(weight)
    }

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
        let group_size = self
            .group_size
            .unwrap_or(QuantizedEmbedding::DEFAULT_GROUP_SIZE);
        let bits = self.bits.unwrap_or(QuantizedEmbedding::DEFAULT_BITS);

        let (quantized_weight, scales, biases) = quantize(&weight, group_size, bits)?;

        let inner = Embedding {
            weight: Param::new(quantized_weight),
        };

        Ok(QuantizedEmbedding {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: Param::new(inner),
        })
    }
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Creates a new [`QuantizedEmbedding`]
    pub fn new(embedding_count: i32, dims: i32) -> Result<Self, Exception> {
        QuantizedEmbeddingBuilder::default().build(embedding_count, dims)
    }

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

impl Module for QuantizedEmbedding {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let s = x.shape();
        let x = x.flatten(None, None)?;
        let w = self.inner.weight.index(&x);
        let scales = self.scales.index(&x);

        let out = dequantize(&w, &scales, &self.biases, self.group_size, self.bits)?;

        let ret_shape = s.into_iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

/// Builder for [`QuantizedLinear`]
#[derive(Debug, Clone, Default)]
pub struct QuantizedLinearBuilder {
    /// Quantization group size. Default to [`QuantizedLinear::DEFAULT_GROUP_SIZE`]
    pub group_size: Option<i32>,

    /// Bits per parameter. Default to [`QuantizedLinear::DEFAULT_BITS`]
    pub bits: Option<i32>,

    /// Whether the linear layer has a bias. Default to [`Linear::DEFAULT_BIAS`]
    pub bias: Option<bool>,
}

impl QuantizedLinearBuilder {
    /// Creates a new [`QuantizedLinearBuilder`]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the group size
    pub fn group_size(mut self, group_size: impl Into<Option<i32>>) -> Self {
        self.group_size = group_size.into();
        self
    }

    /// Sets the bits
    pub fn bits(mut self, bits: impl Into<Option<i32>>) -> Self {
        self.bits = bits.into();
        self
    }

    /// Sets bias
    pub fn bias(mut self, bias: impl Into<Option<bool>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Builds a new [`QuantizedLinear`]
    pub fn build(self, input_dims: i32, output_dims: i32) -> Result<QuantizedLinear, Exception> {
        let scale = f32::sqrt(1.0 / (input_dims as f32));
        let weight = uniform::<_, f32>(-scale, scale, &[output_dims, input_dims], None)?;

        let bias = if self.bias.unwrap_or(Linear::DEFAULT_BIAS) {
            Some(zeros::<f32>(&[output_dims])?)
        } else {
            None
        };

        Self::build_with_weight_and_bias(self.group_size, self.bits, weight, bias)
    }

    /// Convenience method to build a new [`QuantizedLinear`] with an existing [`Linear`]
    pub fn build_with_linear(self, other: Linear) -> Result<QuantizedLinear, Exception> {
        Self::build_with_weight_and_bias(
            self.group_size,
            self.bits,
            other.weight.value,
            other.bias.value,
        )
    }

    fn build_with_weight_and_bias(
        group_size: Option<i32>,
        bits: Option<i32>,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<QuantizedLinear, Exception> {
        let group_size = group_size.unwrap_or(QuantizedLinear::DEFAULT_GROUP_SIZE);
        let bits = bits.unwrap_or(QuantizedLinear::DEFAULT_BITS);

        let (quantized_weight, scales, biases) = quantize(&weight, group_size, bits)?;

        let inner = Linear {
            weight: Param::new(quantized_weight),
            bias: Param::new(bias),
        };

        let mut linear = QuantizedLinear {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: Param::new(inner),
        };

        // Freeze the parameters
        // TODO: we need a way to recursively freeze/unfreeze parameters
        linear.scales.freeze();
        linear.biases.freeze();
        linear.inner.freeze();

        Ok(linear)
    }
}

/// Applies an affine transformation to the input using a quantized weight matrix.
///
/// It is the quantized equivalent of [`Linear`].  For now its
/// parameters are frozen and will not be included in any gradient computation
/// but this will probably change in the future.
///
/// QuantizedLinear also provides several useful static to convert linear
/// layers to QuantizedLinear layers.
#[derive(Debug, Clone, ModuleParameters)]
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
    pub inner: Param<Linear>,
}

impl QuantizedLinear {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Creates a new builder for [`QuantizedLinear`]
    pub fn builder() -> QuantizedLinearBuilder {
        QuantizedLinearBuilder::new()
    }

    /// Creates a new [`QuantizedLinear`]
    pub fn new(input_dims: i32, output_dims: i32) -> Result<Self, Exception> {
        QuantizedLinearBuilder::default().build(input_dims, output_dims)
    }
}

impl Module for QuantizedLinear {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
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
