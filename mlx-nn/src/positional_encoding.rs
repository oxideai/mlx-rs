use mlx_internal_macros::generate_builder;
use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::Array;

generate_builder! {
    /// Implements the rotary positional encoding.
    ///
    /// The traditional implementation rotates consecutive pairs of elements in the
    /// feature dimension while the default implementation rotates pairs with
    /// stride half the feature dimensions for efficiency.
    ///
    /// For more details see _RoFormer: Enhanced Transformer with Rotary Position
    /// Embedding_ ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864))
    #[derive(Debug, Clone, ModuleParameters)]
    #[generate_builder(generate_build_fn = true)]
    struct RoPE {
        /// The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged
        dimensions: i32,
        /// Whether to use the traditional implementation which is slightly less efficient
        #[optional(default_value = RoPE::DEFAULT_TRAIDITIONAL)]
        traditional: bool,
        /// The base used to compute angular frequency for each dimension in the positional encodings
        #[optional(default_value = RoPE::DEFAULT_BASE)]
        base: f32,
        /// The scale used to scale the positions
        #[optional(default_value = RoPE::DEFAULT_SCALE)]
        scale: f32,
    }
}

impl RoPE {
    /// Default value for the `traditional` field.
    pub const DEFAULT_TRAIDITIONAL: bool = false;
    /// Default value for the `base` field.
    pub const DEFAULT_BASE: f32 = 10000.0;
    /// Default value for the `scale` field.
    pub const DEFAULT_SCALE: f32 = 1.0;

    fn forward_with_offset(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)])?;
        let x = mlx_rs::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            self.base,
            self.scale,
            offset,
            None,
        )?;

        x.reshape(shape)
    }
}

impl Module for RoPE {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        self.forward_with_offset(x, 0)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    use super::RoPE;
    use float_eq::assert_float_eq;
    use mlx_rs::module::Module;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        mlx_rs::random::seed(71);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);

        let result = RoPE::new(8).forward(&a).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), mlx_rs::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }
}
