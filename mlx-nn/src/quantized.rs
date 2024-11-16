use std::iter::once;

use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::{
    array,
    error::Exception,
    module::{Module, Param},
    ops::{dequantize, quantize, quantized_matmul, sqrt},
    prelude::IndexOp,
    Array,
};

use crate::{embedding, Embedding};

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
    pub fn build(
        self,
        embedding_count: i32,
        dimensions: i32,
    ) -> Result<QuantizedEmbedding, Exception> {
        let scale = array!(f32::sqrt(1.0 / (dimensions as f32)));
        // SAFETY: This is safe because the array scale is a single element array
        let weight =
            mlx_rs::random::normal::<f32>(&[embedding_count, dimensions], None, None, None)?
                * &scale;

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
    pub fn new(embedding_count: i32, dimensions: i32) -> Result<Self, Exception> {
        QuantizedEmbeddingBuilder::default().build(embedding_count, dimensions)
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
