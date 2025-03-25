//! Unofficial rust bindings for the [MLX
//! framework](https://github.com/ml-explore/mlx).
//! 
//! # Quick Start
//! 
//! ## Basics
//! 
//! ```rust
//! use mlx_rs::{array, Dtype};
//! 
//! let a = array!([1, 2, 3, 4]);
//! assert_eq!(a.shape(), &[4]);
//! assert_eq!(a.dtype(), Dtype::Int32);
//! 
//! let b = array!([1.0, 2.0, 3.0, 4.0]);
//! assert_eq!(b.dtype(), Dtype::Float32);
//! ```
//! 
//! Operations in MLX are lazy. Use [`Array::eval`] to evaluate the
//! the output of an operation. Operations are also automatically evaluated when
//! inspecting an array with [`Array::item`], printing an array, or attempting to
//! obtain the underlying data with [`Array::as_slice`].
//! 
//! ```rust
//! use mlx_rs::{array, transforms::eval};
//! 
//! let a = array!([1, 2, 3, 4]);
//! let b = array!([1.0, 2.0, 3.0, 4.0]);
//! 
//! let c = a + b; // c is not evaluated
//! c.eval().unwrap(); // evaluates c
//! 
//! let d = a + b;
//! println!("{:?}", d); // evaluates d
//! 
//! let e = a + b;
//! let e_slice: &[f32] = e.as_slice().unwrap(); // evaluates e
//! ```


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
