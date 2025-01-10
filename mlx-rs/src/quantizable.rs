//! Traits for quantization

use crate::module::{Module, ModuleParameters};

/// Trait for quantization.
pub trait Quantizable {
    /// The default group size for quantization.
    const DEFAULT_GROUP_SIZE: i32 = 64;

    /// The default number of bits for quantization.
    const DEFAULT_BITS: i32 = 4;

    /// The quantized type.
    type Quantized;

    /// The error type for quantization.
    type QuantizationError;

    /// Quantize the module with the specified group size and number of bits.
    fn quantize_with_group_size_and_bits(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError>;

    /// Quantize the module with the [`QuantizableModule::DEFAULT_GROUP_SIZE`]
    /// and [`QuantizableModule::DEFAULT_BITS`].
    fn quantize(self) -> Result<Self::Quantized, Self::QuantizationError>
    where
        Self: Sized,
    {
        Self::quantize_with_group_size_and_bits(self, Self::DEFAULT_GROUP_SIZE, Self::DEFAULT_BITS)
    }
}

/// A wrapper for a quantizable module.
#[derive(Debug, Clone)]
pub enum MaybeQuantized<M>
where
    M: Quantizable,
{
    /// The original module.
    Original(M),

    /// The quantized version of the module.
    Quantized(M::Quantized),
}

impl<M> Quantizable for MaybeQuantized<M>
where
    M: Quantizable,
{
    type Quantized = Self;
    type QuantizationError = <M as Quantizable>::QuantizationError;

    fn quantize_with_group_size_and_bits(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Self::QuantizationError> {
        match self {
            MaybeQuantized::Original(m) => {
                let quantized = m.quantize_with_group_size_and_bits(group_size, bits)?;
                Ok(MaybeQuantized::Quantized(quantized))
            }
            MaybeQuantized::Quantized(q) => Ok(MaybeQuantized::Quantized(q)),
        }
    }
}

impl<M> MaybeQuantized<M>
where
    M: Quantizable,
{
    /// Create a new [`MaybeQuantized`] from the original module.
    pub fn new(module: M) -> Self {
        MaybeQuantized::Original(module)
    }

    /// Quantize the module with a custom quantization function.
    ///
    /// This is useful if one would like to quantize with a custom group size or bit width.
    pub fn quantize_with(
        self,
        op: impl FnOnce(M) -> Result<M::Quantized, M::QuantizationError>,
    ) -> Result<Self, M::QuantizationError> {
        match self {
            MaybeQuantized::Original(m) => op(m).map(MaybeQuantized::Quantized),
            MaybeQuantized::Quantized(q) => Ok(MaybeQuantized::Quantized(q)),
        }
    }

    /// Check if the module is quantized.
    pub fn is_quantized(&self) -> bool {
        match self {
            MaybeQuantized::Original(_) => false,
            MaybeQuantized::Quantized(_) => true,
        }
    }
}

impl<M> ModuleParameters for MaybeQuantized<M>
where
    M: Quantizable + ModuleParameters,
    M::Quantized: ModuleParameters,
{
    fn parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            MaybeQuantized::Original(m) => m.parameters(),
            MaybeQuantized::Quantized(q) => q.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> crate::module::ModuleParamMut<'_> {
        match self {
            MaybeQuantized::Original(m) => m.parameters_mut(),
            MaybeQuantized::Quantized(q) => q.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            MaybeQuantized::Original(m) => m.trainable_parameters(),
            MaybeQuantized::Quantized(q) => q.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            MaybeQuantized::Original(m) => m.freeze_parameters(recursive),
            MaybeQuantized::Quantized(q) => q.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            MaybeQuantized::Original(m) => m.unfreeze_parameters(recursive),
            MaybeQuantized::Quantized(q) => q.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            MaybeQuantized::Original(m) => m.all_frozen(),
            MaybeQuantized::Quantized(q) => q.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            MaybeQuantized::Original(m) => m.any_frozen(),
            MaybeQuantized::Quantized(q) => q.any_frozen(),
        }
    }
}

impl<M, Input> Module<Input> for MaybeQuantized<M>
where
    M: Quantizable + Module<Input>,
    M::Quantized:
        Module<Input, Output = <M as Module<Input>>::Output, Error = <M as Module<Input>>::Error>,
{
    type Output = <M as Module<Input>>::Output;

    type Error = <M as Module<Input>>::Error;

    fn forward(&mut self, x: Input) -> Result<Self::Output, Self::Error> {
        match self {
            MaybeQuantized::Original(m) => m.forward(x),
            MaybeQuantized::Quantized(q) => q.forward(x),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            MaybeQuantized::Original(m) => m.training_mode(mode),
            MaybeQuantized::Quantized(q) => q.training_mode(mode),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{Embedding, Linear};

    use super::*;

    #[test]
    fn test_quantizable_linear() {
        let linear = Linear::new(64, 64).unwrap();
        let mut qlinear = MaybeQuantized::new(linear);
        assert!(!qlinear.is_quantized());

        qlinear = qlinear.quantize().unwrap();
        assert!(qlinear.is_quantized());
    }

    #[test]
    fn test_quantizable_embedding() {
        let embedding = Embedding::new(64, 64).unwrap();
        let mut qembedding = MaybeQuantized::new(embedding);
        assert!(!qembedding.is_quantized());

        qembedding = qembedding.quantize().unwrap();
        assert!(qembedding.is_quantized());
    }
}
