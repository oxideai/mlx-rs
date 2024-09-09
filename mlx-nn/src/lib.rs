pub mod macros;
pub mod utils;
pub mod value_and_grad;

mod activation;
mod convolution;
mod linear;
mod sequential;

pub use activation::*;
pub use convolution::*;
pub use linear::*;
pub use sequential::*;

pub mod module {
    pub use mlx_nn_module::*;
}
