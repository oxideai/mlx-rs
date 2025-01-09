//! Traits for quantization

use crate::{error::Exception, module::{Module, ModuleParameters}};

/// Marker trait for a quantizable module.z
pub trait QuantizableModule<'a>: Module<'a> {
    /// The quantized version of the module.
    type Quantized: Module<'a, Input = Self::Input, Output = Self::Output, Error = Self::Error>;
    /// The error associated with quantization.
    type QuantizationError: std::error::Error;

    /// Convert the module into a quantized version.
    fn quantize(self) -> Result<Self::Quantized, Self::QuantizationError>;
}

/// A wrapper for a quantizable module.
#[derive(Debug, Clone)]
pub enum Quantizable<M, Q> 
where 
    for<'a> M: QuantizableModule<'a, Quantized = Q>,
    for<'q> Q: Module<'q, Input = <M as Module<'q>>::Input, Output = <M as Module<'q>>::Output, Error = <M as Module<'q>>::Error>,
{
    /// The original module.
    Original(M),

    /// The quantized version of the module.
    Quantized(Q),
}

impl<M, Q> Quantizable<M, Q>
where 
    for<'a> M: QuantizableModule<'a, Quantized = Q>,
    for<'q> Q: Module<'q, Input = <M as Module<'q>>::Input, Output = <M as Module<'q>>::Output, Error = <M as Module<'q>>::Error>,
{
    /// Create a new [`Quantizable`] from the original module.
    pub fn new(module: M) -> Self {
        Quantizable::Original(module)
    }

    /// Convert the module into a quantized version.
    pub fn quantize<'a>(self) -> Result<Quantizable<M, Q>, <M as QuantizableModule<'a>>::QuantizationError> {
        match self {
            Quantizable::Original(m) => m.quantize().map(Quantizable::Quantized),
            Quantizable::Quantized(q) => Ok(Quantizable::Quantized(q)),
        }
    }

    /// Quantize the module with a custom quantization function.
    /// 
    /// This is useful if one would like to quantize with a custom group size or bit width.
    pub fn quantize_with<'a>(self, op: impl FnOnce(M) -> Result<Q, <M as QuantizableModule<'a>>::QuantizationError>) -> Result<Quantizable<M, Q>, <M as QuantizableModule<'a>>::QuantizationError> {
        match self {
            Quantizable::Original(m) => op(m).map(Quantizable::Quantized),
            Quantizable::Quantized(q) => Ok(Quantizable::Quantized(q)),
        }
    }

    /// Check if the module is quantized.
    pub fn is_quantized(&self) -> bool {
        match self {
            Quantizable::Original(_) => false,
            Quantizable::Quantized(_) => true,
        }
    }
}

impl<M, Q> ModuleParameters for Quantizable<M, Q> 
where 
    for<'m> M: QuantizableModule<'m, Quantized = Q>,
    for<'q> Q: Module<'q, Input = <M as Module<'q>>::Input, Output = <M as Module<'q>>::Output, Error = <M as Module<'q>>::Error>,
{
    fn parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            Quantizable::Original(m) => m.parameters(),
            Quantizable::Quantized(q) => q.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> crate::module::ModuleParamMut<'_> {
        match self {
            Quantizable::Original(m) => m.parameters_mut(),
            Quantizable::Quantized(q) => q.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            Quantizable::Original(m) => m.trainable_parameters(),
            Quantizable::Quantized(q) => q.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Quantizable::Original(m) => m.freeze_parameters(recursive),
            Quantizable::Quantized(q) => q.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Quantizable::Original(m) => m.unfreeze_parameters(recursive),
            Quantizable::Quantized(q) => q.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Quantizable::Original(m) => m.all_frozen(),
            Quantizable::Quantized(q) => q.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Quantizable::Original(m) => m.any_frozen(),
            Quantizable::Quantized(q) => q.any_frozen(),
        }
    }
}

impl<'a, M, Q> Module<'a> for Quantizable<M, Q> 
where 
    for<'m> M: QuantizableModule<'m, Quantized = Q>,
    for<'q> Q: Module<'q, Input = <M as Module<'q>>::Input, Output = <M as Module<'q>>::Output, Error = <M as Module<'q>>::Error>,
{
    type Input = <M as Module<'a>>::Input;

    type Output = <M as Module<'a>>::Output;

    type Error = <M as Module<'a>>::Error;

    fn forward(&mut self, x: Self::Input) -> Result<Self::Output, Self::Error> {
        match self {
            Quantizable::Original(m) => m.forward(x),
            Quantizable::Quantized(q) => q.forward(x),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Quantizable::Original(m) => m.training_mode(mode),
            Quantizable::Quantized(q) => q.training_mode(mode),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::Linear;

    use super::Quantizable;

    #[test]
    fn test_quantizable_linear() {
        let linear = Linear::new(64, 64).unwrap();
        let mut qlinear = Quantizable::new(linear);
        assert!(!qlinear.is_quantized());

        qlinear = qlinear.quantize().unwrap();
        assert!(qlinear.is_quantized());
    }
}