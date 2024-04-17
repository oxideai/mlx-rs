#![deny(unused_unsafe, )]

pub mod device;
pub mod array;
mod utils;

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}
}