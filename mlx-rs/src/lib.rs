//! Unofficial rust bindings for the [MLX
//! framework](https://github.com/ml-explore/mlx).

#![deny(unused_unsafe, missing_debug_implementations, missing_docs)]
#![cfg_attr(test, allow(clippy::approx_constant))]

#[macro_use]
pub mod macros; // Must be first to ensure the other modules can use the macros

mod array;
pub mod builder;
mod device;
mod dtype;
pub mod error;
pub mod fast;
pub mod fft;
pub mod linalg;
pub mod losses;
pub mod module;
pub mod nested;
pub mod nn;
pub mod ops;
pub mod optimizers;
pub mod quantization;
pub mod random;
mod stream;
pub mod transforms;
pub mod utils;

pub use array::*;
pub use device::*;
pub use dtype::*;
pub use stream::*;

pub(crate) mod constants {
    /// The default length of the stack-allocated vector in `SmallVec<[T; DEFAULT_STACK_VEC_LEN]>`
    pub(crate) const DEFAULT_STACK_VEC_LEN: usize = 4;
}

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}

    impl Sealed for () {}

    impl<A> Sealed for (A,) where A: Sealed {}
    impl<A, B> Sealed for (A, B)
    where
        A: Sealed,
        B: Sealed,
    {
    }
    impl<A, B, C> Sealed for (A, B, C)
    where
        A: Sealed,
        B: Sealed,
        C: Sealed,
    {
    }
}
