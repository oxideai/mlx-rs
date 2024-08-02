use mlx_rs::{error::Exception, Array};

pub trait Module {
    fn forward(&self, x: Array) -> Result<Array, Exception>;
}