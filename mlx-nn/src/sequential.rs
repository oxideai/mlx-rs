use std::borrow::Cow;

use mlx_nn_module::{Module, ModuleParameters};
use mlx_rs::{error::Exception, Array};

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl ModuleParameters for Sequential {
    fn parameters(&self) -> mlx_nn_module::ModuleParamRef<'_> {
        todo!()
    }

    fn parameters_mut(&mut self) -> mlx_nn_module::ModuleParamMut<'_> {
        todo!()
    }

    fn trainable_parameters(&self) -> mlx_nn_module::ModuleParamRef<'_> {
        todo!()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let mut x = Cow::Borrowed(x);

        for layer in &self.layers {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.deep_clone()),
        }
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn append(mut self, layer: impl Module + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}
