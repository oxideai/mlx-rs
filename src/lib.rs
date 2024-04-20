#![deny(unused_unsafe)]

mod array;
mod device;
mod dtype;
pub mod error;
pub mod fft;
pub mod ops;
mod stream;
mod utils;

pub use array::*;
pub use device::*;
pub use dtype::*;
pub use stream::*;

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}
}
