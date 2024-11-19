use std::borrow::Cow;

use mlx_macros::ModuleParameters;
use mlx_rs::module::{Module, UnaryModule};
use mlx_rs::{error::Exception, Array};

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
pub struct Sequential<Err = Exception> {
    /// The layers to be called in sequence.
    #[param]
    pub layers: Vec<Box<dyn SequentialModuleItem<Err>>>,
}

impl<'a> Module<&'a Array> for Sequential {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = Cow::Borrowed(x);

        for layer in &mut self.layers {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.clone())
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
