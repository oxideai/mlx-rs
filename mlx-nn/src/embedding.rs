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
        Ok(self.weight.index(x))
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_embedding() {
        mlx_rs::random::seed(557);
        let a = mlx_rs::random::randint::<_, i32>(0, 10, &[2, 8, 8, 4], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Int32);
        float_eq!(
            a.mean(None, None).unwrap().item::<f32>(),
            4.60546875,
            abs <= 0.09210937500000001
        );
        float_eq!(
            a.sum(None, None).unwrap().item::<f32>(),
            2358.0,
            abs <= 47.160000000000004
        );

        let result = Embedding::new(10, 8).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8, 4, 8]);
        assert_eq!(result.dtype(), mlx_rs::Dtype::Float32);
        float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            -0.0011973462533205748,
            abs <= 2.3946925066411497e-05
        );
        float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            -4.904330253601074,
            abs <= 0.0980866050720215
        );
    }
}
