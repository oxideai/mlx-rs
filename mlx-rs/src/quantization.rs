//! Traits for quantization

use crate::module::Module;

/// Marker trait for a quantizable module.z
pub trait QuantizableModule {
    /// The arguments to the module.
    type Args;

    /// The quantized version of the module.
    type Quantized: Module<Self::Args>;

    /// Convert the module into a quantized version.
    fn into_quantized(self) -> Self::Quantized;
}

/// A wrapper for a quantizable module.
#[derive(Debug, Clone)]
pub enum Quantizable<M> 
where 
    M: QuantizableModule,
{
    /// The original module.
    Original(M),

    /// The quantized version of the module.
    Quantized(M::Quantized),
}
