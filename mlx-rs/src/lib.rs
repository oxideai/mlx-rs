//! Unofficial rust bindings for the [MLX
//! framework](https://github.com/ml-explore/mlx).
//! 
//! # Table of Contents
//! 
//! - [Quick Start](#quick-start)
//! - [Lazy Evaluation](#lazy-evaluation)
//! 
//! # Quick Start
//! 
//! See also [MLX python
//! documentation](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
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
//! Operations in MLX are lazy. Use [`Array::eval`] to evaluate the the output
//! of an operation. Operations are also automatically evaluated when inspecting
//! an array with [`Array::item`], printing an array, or attempting to obtain
//! the underlying data with [`Array::as_slice`].
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
//! 
//! See [Lazy Evaluation](#lazy-evaluation) for more details.
//! 
//! ## Function and Graph Transformations
//! 
//! TODO: https://github.com/oxideai/mlx-rs/issues/214
//! 
//! TODO: also document that all `Array` in the args for function
//!       transformations
//! 
//! # Lazy Evaluation
//! 
//! See also [MLX python
//! documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
//! 
//! ## Why Lazy Evaluation
//! 
//! When you perform operations in MLX, no computation actually happens. Instead
//! a compute graph is recorded. The actual computation only happens if an
//! [`Array::eval`] is performed.
//! 
//! MLX uses lazy evaluation because it has some nice features, some of which we
//! describe below.
//! 
//! ## Transforming Compute Graphs
//! 
//! Lazy evaluation lets us record a compute graph without actually doing any
//! computations. This is useful for function transformations like
//! [`transforms::grad`] and graph optimizations.
//! 
//! Currently, MLX does not compile and rerun compute graphs. They are all
//! generated dynamically. However, lazy evaluation makes it much easier to
//! integrate compilation for future performance enhancements.
//! 
//! ## Only Compute What You Use
//! 
//! In MLX you do not need to worry as much about computing outputs that are
//! never used. For example:
//! 
//! ```rust,ignore
//! fn fun(x: &Array) -> (Array, Array) {
//!     let a = cheap_fun(x);
//!     let b = expensive_fun(x);
//!     (a, b)
//! }
//! 
//! let (y, _) = fun(&x);
//! ```
//! 
//! Here, we never actually compute the output of `expensive_fun`. Use this
//! pattern with care though, as the graph of `expensive_fun` is still built,
//! and that has some cost associated to it.
//! 
//! Similarly, lazy evaluation can be beneficial for saving memory while keeping
//! code simple. Say you have a very large model `Model` implementing
//! [`module::Module`]. You can instantiate this model with `let model =
//! Model::new()`. Typically, this will initialize all of the weights as
//! `float32`, but the initialization does not actually compute anything until
//! you perform an `eval()`. If you update the model with `float16` weights,
//! your maximum consumed memory will be half that required if eager computation
//! was used instead.
//! 
//! This pattern is simple to do in MLX thanks to lazy computation:
//! 
//! ```rust,ignore
//! let mut model = Model::new();
//! model.load_safetensors("model.safetensors").unwrap();
//! ```
//! 
//! ## When to Evaluate
//! 
//! A common question is when to use `eval()`. The trade-off is between letting
//! graphs get too large and not batching enough useful work.
//! 
//! For example
//! 
//! ```rust,ignore
//! let mut a = array!([1, 2, 3, 4]);
//! let mut b = array!([1.0, 2.0, 3.0, 4.0]);
//! 
//! for _ in 0..100 {
//!     a = a + b;
//!     a.eval()?;
//!     b = b * 2.0;
//!     b.eval()?;
//! }
//! ```
//! 
//! This is a bad idea because there is some fixed overhead with each graph
//! evaluation. On the other hand, there is some slight overhead which grows
//! with the compute graph size, so extremely large graphs (while
//! computationally correct) can be costly.
//! 
//! Luckily, a wide range of compute graph sizes work pretty well with MLX:
//! anything from a few tens of operations to many thousands of operations per
//! evaluation should be okay.
//! 
//! Most numerical computations have an iterative outer loop (e.g. the iteration
//! in stochastic gradient descent). A natural and usually efficient place to
//! use `eval()` is at each iteration of this outer loop.
//!
//! Here is a concrete example:
//! 
//! ```rust,ignore
//! for batch in dataset {
//!     // Nothing has been evaluated yet
//!     let (loss, grad) = value_and_grad_fn(&mut model, batch)?;
//! 
//!     // Still nothing has been evaluated
//!     optimizer.update(&mut model, grad)?;
//! 
//!     // Evaluate the loss and the new parameters which will
//!     // run the full gradient computation and optimizer update
//!     eval_params(model.parameters())?;
//! }
//! ```
//! 
//! An important behavior to be aware of is when the graph will be implicitly
//! evaluated. Anytime you `print` an array, or otherwise access its memory via
//! [`Array::as_slice`], the graph will be evaluated. Saving arrays via
//! [`Array::save_numpy`] or [`Array::save_safetensors`] (or any other MLX
//! saving functions) will also evaluate the array.
//! 
//! Calling [`Array::item`] on a scalar array will also evaluate it. In the
//! example above, printing the loss (`println!("{:?}", loss)`) or pushing the
//! loss scalar to a [`Vec`] (`losses.push(loss.item::<f32>())`) would cause a
//! graph evaluation. If these lines are before evaluating the loss and module
//! parameters, then this will be a partial evaluation, computing only the
//! forward pass.
//! 
//! Also, calling `eval()` on an array or set of arrays multiple times is
//! perfectly fine. This is effectively a no-op.
//! 
//! **Warning**: Using scalar arrays for control-flow will cause an evaluation.
//! 
//! ```rust,ignore
//! fn fun(x: &Array) -> Array {
//!     let (h, y) = first_layer(x);
//! 
//!     if y.gt(array!(0.5)).unwrap().item() {
//!         second_layer_a(h)
//!     } else {
//!         second_layer_b(h)
//!     }
//! }
//! ```
//! 
//! Using arrays for control flow should be done with care. The above example
//! works and can even be used with gradient transformations. However, this can
//! be very inefficient if evaluations are done too frequently.

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
