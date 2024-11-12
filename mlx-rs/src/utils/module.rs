use crate::{module::*, Array};


/// A marker module that panics when called.
/// 
/// This is a convenience type for `MaybeQuantizable` trait implementations that are not quantized.
#[derive(Debug, Clone)]
pub struct Never;

impl ModuleParameters for Never {
    fn parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::new()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        ModuleParamMut::new()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::new()
    }
}

impl Module for Never {
    type Error = std::convert::Infallible;

    fn forward(&self, _: &Array) -> Result<Array, Self::Error> {
        unreachable!("Never module should never be called")
    }

    fn training_mode(&mut self, _: bool) { }
}