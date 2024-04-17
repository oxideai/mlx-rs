#![deny(unused_unsafe)]

pub mod dtype;
pub mod array;
pub mod device;
mod utils;

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}
}
