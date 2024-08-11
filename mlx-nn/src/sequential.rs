use mlx_rs::{error::Exception, Array};

use crate::Module;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&self, mut x: Array) -> Result<Array, Exception> {
        for layer in &self.layers {
            x = layer.forward(x)?;
        }
        Ok(x)
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
