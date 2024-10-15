use std::borrow::Cow;

use mlx_macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::{error::Exception, Array};

/// Marker trait for items that can be used in a `Sequential` module.
///
/// It is implemented for all types that implement [`Module`] and [`std::fmt::Debug`].
pub trait SequentialModuleItem<Err>: Module<Error = Err> + std::fmt::Debug {}

impl<T, Err> SequentialModuleItem<Err> for T
where
    T: Module<Error = Err> + std::fmt::Debug,
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
    pub layers: Param<Vec<Box<dyn SequentialModuleItem<Err>>>>,
}

impl Module for Sequential {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = Cow::Borrowed(x);

        for layer in &self.layers.value {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.deep_clone()),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl<Err> Sequential<Err> {
    /// Creates a new [`Sequential`] module.
    pub fn new() -> Self {
        Self {
            layers: Param::new(Vec::new()),
        }
    }

    /// Appends a layer to the sequential module.
    pub fn append<M>(mut self, layer: M) -> Self
    where
        M: Module<Error = Err> + std::fmt::Debug + 'static,
        Err: std::error::Error + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}
