//! Embedding layer.

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::module::Param;
use mlx_rs::prelude::IndexOp;
use mlx_rs::Array;

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Embedding {
    /// The weight of the
    #[param]
    pub weight: Param<Array>,
}

impl Embedding {
    /// Creates a new [`Embedding`] layer.
    ///
    /// # Params
    ///
    /// - `embedding_count`: How many possible discrete tokens can we embed.  Usually called the vocabulary size.
    /// - `dimensions`: The dimensionality of the embeddings.
    pub fn new(embedding_count: i32, dimensions: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / (dimensions as f32));
        let weight =
            mlx_rs::random::normal::<f32>(&[embedding_count, dimensions], None, None, None)?
                * scale;

        Ok(Self {
            weight: Param::new(weight),
        })
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::ops::matmul(x, &self.weight.value.t())
    }
}

impl Module for Embedding {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        Ok(self.weight.index(x.clone()))
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::array;

    #[test]
    fn test_embedding() {
        let embedding = Embedding::new(10, 3).unwrap();
        let input = array!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let output = embedding.forward(&input).unwrap();

        assert_eq!(output.shape(), [10, 3]);
    }
}
