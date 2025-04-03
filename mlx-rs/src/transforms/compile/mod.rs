//! Compilation of functions.
//!
//! See also [MLX python
//! documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
//!
//! MLX has a [`compile()`] function transformation which compiles computation
//! graphs. Function compilation results in smaller graphs by merging common
//! work and fusing certain operations. In many cases this can lead to big
//! improvements in run-time and memory use.
//!
//! Getting started with compile() is simple, but there are some edge cases that
//! are good to be aware of for more complex graphs and advanced usage.
//!
//! **WARN**: Because function transforms including compilation works on the
//! computation graph, the user must ensure that all `Array`s are passed as
//! inputs to the function/closure. Closures with captured `Array`s may not work
//! as expected and may lead to undefined behavior.
//!
//! # Basic usage
//!
//! ```rust
//! use mlx_rs::{Array, array, transforms::compile::compile, error::Exception};
//!
//! let fun = |(x, y): (&Array, &Array)| -> Result<Array, Exception> {
//!    mlx_rs::exp!(x.negative()?)?.add(y)
//! };
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! // Regular call, no compilation
//! let result = fun((&x, &y)).unwrap();
//! // Prints: array(2.36788, dtype=float32)
//! println!("{:?}", result);
//!
//! // Compile the function
//! let compiled_fun = compile(fun, None);
//! let result = compiled_fun((&x, &y)).unwrap();
//! // Prints: array(2.36788, dtype=float32)
//! println!("{:?}", result);
//! ```
//!
//! The output of both the regular function and the compiled function is the
//! same up to numerical precision.
//!
//! The first time you call a compiled function, MLX will build the compute
//! graph, optimize it, and generate and compile code. This can be relatively
//! slow. However, MLX will cache compiled functions, so calling a compiled
//! function multiple times will not initiate a new compilation. This means you
//! should typically compile functions that you plan to use more than once.
//!
//! ```rust
//! use mlx_rs::{Array, array, transforms::compile::compile};
//!
//! let fun = |(x, y): (&Array, &Array)| {
//!    mlx_rs::exp!(x.negative()?)?.add(y)
//! };
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! let compiled_fun = compile(fun, None);
//!
//! // Compiled here
//! let result = compiled_fun((&x, &y)).unwrap();
//!
//! // Not compiled again
//! let result = compiled_fun((&x, &y)).unwrap();
//!
//! // Not compiled again
//! let compiled_fun2 = compile(fun, None);
//! ```
//!
//! There are some important cases to be aware of that can cause a function to
//! be recompiled:
//!
//! - Changing the shape or number of dimensions
//! - Changing the type of any of the inputs
//! - Changing the number of inputs to the function
//!
//! In certain cases only some of the compilation stack will be rerun (for
//! example when changing the shapes) and in other cases the full compilation
//! stack will be rerun (for example when changing the types). In general you
//! should avoid compiling functions too frequently.
//!
//! Another idiom to watch out for is compiling functions which get created and
//! destroyed frequently. This can happen, for example, when compiling an
//! closure in a loop.
//!
//! # Pure Functions
//!
//! Compiled functions are intended to be pure; that is they should not have
//! side effects. For example:
//!
//! ```rust,no_run
//! use mlx_rs::{Array, array, transforms::compile::compile};
//!
//! let mut state = vec![];
//!
//! let fun = |(x, y): (&Array, &Array)| {
//!     let z = x + y;
//!     state.push(z);
//!     mlx_rs::exp!(z)
//! };
//!
//! let compiled = compile(fun, None);
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! // This will crash or lead to undefined behavior
//! let result = compiled((&x, &y)).unwrap();
//! println!("{:?}", result);
//! ```
//!
//! Use [`compile_with_state()`] to compile functions that have side effects and
//! pass the state as an mutable reference.
//!
//! ```rust
//! let mut state = vec![];
//!
//! let fun = |state: &mut Vec<Array>, (x, y): (&Array, &Array)| {
//!     let z = x + y;
//!     state.push(z);
//!     mlx_rs::exp!(z)
//! };
//!
//! let compiled = compile_with_state(fun, None);
//! let result = compiled(&mut state, (&x, &y)).unwrap();
//! println!("{:?}", result);
//! println!("{:?}", state);
//! ```
//!
//! This is particularly useful for compiling a function which includes an
//! update to a container of arrays, as is commonly done when training the
//! parameters of a [`crate::module::Module`].
//!
//! See mlx-rs/mlx-tests/tests/test_compile_with_state.rs for more examples.
//!

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{Closure, Guarded, VectorArray};
use crate::Array;

#[allow(clippy::module_inception)]
mod compile;
mod compile_with_state;

pub use compile::*;
pub use compile_with_state::*;

/// Globally enable the compilation of functions.
///
/// Default is enabled.
pub fn enable_compile() {
    unsafe {
        mlx_sys::mlx_enable_compile();
    }
}

/// Globally disable the compilation of functions.
///
/// Default is enabled.
pub fn disable_compile() {
    unsafe {
        mlx_sys::mlx_disable_compile();
    }
}

/// Clear the memory cache.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_detail_compile_clear_cache();
    }
}

/// A compiled function that can be called.
#[derive(Debug, Clone)]
pub struct Compiled<F, G> {
    f_marker: std::marker::PhantomData<F>,
    state: CompiledState<G>,
}

#[derive(Debug, Clone)]
struct CompiledState<F> {
    f: F,
    shapeless: bool,
    id: usize,
}

impl<F> Drop for CompiledState<F> {
    fn drop(&mut self) {
        unsafe {
            // remove the compiled structure from the back end
            mlx_sys::mlx_detail_compile_erase(self.id);
        }
    }
}

fn type_id_to_usize<T>(_val: &T) -> usize
where
    T: 'static,
{
    // hash type id to usize
    let type_id = std::any::TypeId::of::<T>();
    let mut hasher = DefaultHasher::new();
    type_id.hash(&mut hasher);
    hasher.finish() as usize
}

fn update_by_replace_with_ref_to_new_array(src: &mut Array, new_array: &Array) {
    unsafe {
        mlx_sys::mlx_array_set(&mut src.as_ptr() as *mut _, new_array.as_ptr());
    }
}
