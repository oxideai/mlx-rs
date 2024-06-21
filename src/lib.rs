#![deny(unused_unsafe, missing_debug_implementations)]
#![cfg_attr(test, allow(clippy::approx_constant))]

#[macro_use]
mod macros; // Must be first to ensure the other modules can use the macros

mod array;
mod device;
mod dtype;
pub mod error;
pub mod fft;
pub mod linalg;
pub mod ops;
mod stream;
pub mod utils;

pub use array::*;
pub use device::*;
pub use dtype::*;
pub use stream::*;

// TODO: what to put in the prelude?
pub mod prelude {
    pub use crate::{
        array::Array,
        dtype::Dtype,
        ops::indexing::{Ellipsis, IndexMutOp, IndexOp, IntoStrideBy, NewAxis},
        stream::StreamOrDevice,
        utils::{OwnedOrRef, ScalarOrArray},
    };
}

pub(crate) mod constants {
    /// The default length of the stack-allocated vector in `SmallVec<[T; DEFAULT_STACK_VEC_LEN]>`
    pub(crate) const DEFAULT_STACK_VEC_LEN: usize = 4;
}

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}
}
