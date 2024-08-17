use mlx_rs::{error::Exception, utils::OwnedOrRef, Array};

use crate::Module;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let mut x = OwnedOrRef::Ref(x);

        for layer in &self.layers {
            x = OwnedOrRef::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            OwnedOrRef::Owned(array) => Ok(array),
            OwnedOrRef::Ref(array) => Ok(array.deep_clone()),
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
