//! Traits for quantization

use crate::module::{Module, ModuleParameters};

/// Marker trait for a quantizable module.z
pub trait QuantizableModule<'a>: Module<'a> {
    /// The quantized version of the module.
    type Quantized: Module<'a, Input = Self::Input, Output = Self::Output, Error = Self::Error>;

    /// Convert the module into a quantized version.
    fn into_quantized(self) -> Self::Quantized;
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