use std::borrow::Cow;

use mlx_macros::ModuleParameters;
use mlx_nn_module::{Module, Param};
use mlx_rs::{error::Exception, Array};

#[derive(ModuleParameters)]
pub struct Sequential {
    #[param]
    layers: Param<Vec<Box<dyn Module>>>,
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
    pub fn new() -> Self {
        Self {
            layers: Param::new(Vec::new()),
        }
    }

    pub fn append(mut self, layer: impl Module + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}
