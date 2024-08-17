use mlx_rs::{error::Exception, Array};

pub trait Module {
    // TODO: Should we use `&Array` instead of `Array`? What if an op does nothing and just return
    // the same array?
    fn forward(&self, x: &Array) -> Result<Array, Exception>;
}
