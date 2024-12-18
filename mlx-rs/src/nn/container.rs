use std::borrow::Cow;

use crate::module::{Module, UnaryModule};
use crate::{error::Exception, Array};
use mlx_macros::ModuleParameters;

/// Marker trait for items that can be used in a `Sequential` module.
///
/// It is implemented for all types that implement [`Module`] and [`std::fmt::Debug`].
pub trait SequentialModuleItem<Err>: UnaryModule<Error = Err> + std::fmt::Debug {}

impl<T, Err> SequentialModuleItem<Err> for T
where
    T: UnaryModule<Error = Err> + std::fmt::Debug,
    Err: std::error::Error + 'static,
{
}

/// A sequential layer.
///
/// It calls each layer in sequence.
#[derive(Debug, ModuleParameters)]
#[module(root = crate)]
pub struct Sequential<Err = Exception> {
    /// The layers to be called in sequence.
    #[param]
    pub layers: Vec<Box<dyn SequentialModuleItem<Err>>>,
}

impl Module<&Array> for Sequential {
    type Error = Exception;
    type Output = Array;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = Cow::Borrowed(x);

        for layer in &self.layers {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.clone()),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
    }
}

impl<Err> Default for Sequential<Err> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Err> Sequential<Err> {
    /// Creates a new [`Sequential`] module.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a layer to the sequential module.
    pub fn append<M>(mut self, layer: M) -> Self
    where
        M: UnaryModule<Error = Err> + std::fmt::Debug + 'static,
        Err: std::error::Error + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::module::ModuleParameters;

    use crate::Linear;

    #[test]
    fn test_sequential_linear_param_len() {
        use super::*;

        let model = Sequential::new()
            .append(Linear::new(2, 3).unwrap())
            .append(Linear::new(3, 1).unwrap());

        let params = model.parameters().flatten();
        assert_eq!(params.len(), 4);
    }
}
