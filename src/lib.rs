#![deny(unused_unsafe)]

pub mod array;
pub mod device;
pub mod dtype;
pub mod stream;
mod utils;
mod ops;

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}
}
