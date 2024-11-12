use mlx_macros::ModuleParameters;
use mlx_rs::{error::Exception, module::Module};

use crate::Linear;

/// Enum for quantized modules.
/// 
/// This enum is used to represent a module that may or may not be quantized.
#[derive(Debug)]
pub enum MaybeQuantized<Q, N> {
    /// The module is quantized.
    Quantized(Q),

    /// The module is not quantized.
    NotQuantized(N),
}

/// Trait for modules that may be quantized.
pub trait MaybeQuantizable: Sized {
    /// The default group size if not provided.
    const DEFAULT_GROUP_SIZE: i32 = 64;
    
    /// The default number of bits if not provided.
    const DEFAULT_BITS: i32 = 4;

    /// The quantized version of the module.
    type Quantized;

    /// Convert the module to its quantized version if possible. Default to not quantized unless
    /// manually implemented.
    fn to_maybe_quantized(
        self,
        _group_size: impl Into<Option<i32>>,
        _bits: impl Into<Option<i32>>,
    ) -> MaybeQuantized<Self::Quantized, Self> {
        MaybeQuantized::NotQuantized(self)
    }
}

///
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedEmbedding {

}

///
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedLinear {

}

impl Module for QuantizedLinear {
    type Error = Exception;

    fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array, Self::Error> {
        todo!()
    }

    fn training_mode(&mut self, mode: bool) {
        todo!()
    }
}