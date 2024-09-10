use std::borrow::Cow;

use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{error::Exception, Array};

/// Marker trait for items that can be used in a `Sequential` module.
/// 
/// It is implemented for all types that implement [`Module`] and [`std::fmt::Debug`].
pub trait SequentialModuleItem: Module + std::fmt::Debug {}

impl<T> SequentialModuleItem for T where T: Module + std::fmt::Debug {} 

/// A sequential layer.
/// 
/// It calls each layer in sequence.
#[derive(Debug, ModuleParameters)]
pub struct Sequential {
    /// The layers to be called in sequence.
    #[param]
    pub layers: Param<Vec<Box<dyn SequentialModuleItem>>>,
}

impl Module for Sequential {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let mut x = Cow::Borrowed(x);

        for layer in &self.layers.value {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.deep_clone()),
        }
    }

    fn train(&mut self, mode: bool) {
        self.layers.iter_mut().for_each(|layer| layer.train(mode));
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    /// Creates a new [`Sequential`] module.
    pub fn new() -> Self {
        Self {
            layers: Param::new(Vec::new()),
        }
    }

    /// Appends a layer to the sequential module.
    pub fn append(mut self, layer: impl Module + std::fmt::Debug + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}
