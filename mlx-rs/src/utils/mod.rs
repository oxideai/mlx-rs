use guard::Guarded;
use mlx_sys::mlx_vector_array;

use crate::module::ModuleParameters;
use crate::{complex64, error::Exception, Array, FromNested};
use std::collections::HashMap;
use std::{marker::PhantomData, rc::Rc};

/// Success status code from the c binding
pub(crate) const SUCCESS: i32 = 0;
pub(crate) const FAILURE: i32 = 1;

#[cfg(feature = "io")]
pub(crate) mod io;

pub(crate) mod guard;

pub(crate) fn resolve_index_signed_unchecked(index: i32, len: i32) -> i32 {
    if index < 0 {
        len.saturating_add(index)
    } else {
        index
    }
}

pub(crate) fn resolve_index_unchecked(index: i32, len: usize) -> usize {
    if index.is_negative() {
        (len as i32 + index) as usize
    } else {
        index as usize
    }
}

/// Helper method to convert an optional slice of axes to a Vec covering all axes.
pub(crate) fn axes_or_default_to_all<'a>(axes: impl IntoOption<&'a [i32]>, ndim: i32) -> Vec<i32> {
    match axes.into_option() {
        Some(axes) => axes.to_vec(),
        None => {
            let axes: Vec<i32> = (0..ndim).collect();
            axes
        }
    }
}

pub(crate) struct VectorArray {
    c_vec: mlx_sys::mlx_vector_array,
}

impl VectorArray {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_vector_array {
        self.c_vec
    }

    pub(crate) fn try_from_iter(
        iter: impl Iterator<Item = impl AsRef<Array>>,
    ) -> Result<Self, Exception> {
        VectorArray::try_from_op(|res| unsafe {
            let mut status = SUCCESS;
            for arr in iter {
                status = mlx_sys::mlx_vector_array_append_value(*res, arr.as_ref().as_ptr());
                if status != SUCCESS {
                    return status;
                }
            }
            status
        })
    }

    pub(crate) fn try_into_values<T>(self) -> Result<T, Exception>
    where
        T: FromIterator<Array>,
    {
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.c_vec);
            (0..size)
                .map(|i| {
                    Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, self.c_vec, i))
                })
                .collect::<Result<T, Exception>>()
        }
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_vector_array_free(self.c_vec) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// A helper trait that is just like `Into<Option<T>>` but improves ergonomics by allowing
/// implicit conversion from &[T; N] to &[T].
pub trait IntoOption<T> {
    fn into_option(self) -> Option<T>;
}

impl<T> IntoOption<T> for Option<T> {
    fn into_option(self) -> Option<T> {
        self
    }
}

impl<T> IntoOption<T> for T {
    fn into_option(self) -> Option<T> {
        Some(self)
    }
}

impl<'a, T, const N: usize> IntoOption<&'a [T]> for &'a [T; N] {
    fn into_option(self) -> Option<&'a [T]> {
        Some(self)
    }
}

impl<'a, T> IntoOption<&'a [T]> for &'a Vec<T> {
    fn into_option(self) -> Option<&'a [T]> {
        Some(self)
    }
}

pub trait ScalarOrArray<'a> {
    type Array: AsRef<Array> + 'a;

    fn into_owned_or_ref_array(self) -> Self::Array;
}

impl ScalarOrArray<'_> for Array {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        self
    }
}

impl<'a> ScalarOrArray<'a> for &'a Array {
    type Array = &'a Array;

    // TODO: clippy would complain about `as_array`. Is there a better name?
    fn into_owned_or_ref_array(self) -> &'a Array {
        self
    }
}

impl ScalarOrArray<'static> for bool {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_bool(self)
    }
}

impl ScalarOrArray<'static> for i32 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_int(self)
    }
}

impl ScalarOrArray<'static> for f32 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_float(self)
    }
}

impl ScalarOrArray<'static> for complex64 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_complex(self)
    }
}

impl<T> ScalarOrArray<'static> for T
where
    Array: FromNested<T>,
{
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_nested(self)
    }
}

#[derive(Debug)]
pub(crate) struct Closure<'a> {
    c_closure: mlx_sys::mlx_closure,
    lt_marker: PhantomData<&'a ()>,
}

impl<'a> Closure<'a> {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_closure {
        self.c_closure
    }

    pub(crate) fn new<F>(closure: F) -> Self
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'a,
    {
        let c_closure = new_mlx_closure(closure);
        Self {
            c_closure,
            lt_marker: PhantomData,
        }
    }

    pub(crate) fn new_fallible<F>(closure: F) -> Self
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    {
        let c_closure = new_mlx_fallible_closure(closure);
        Self {
            c_closure,
            lt_marker: PhantomData,
        }
    }
}

impl Drop for Closure<'_> {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_closure_free(self.c_closure) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// Helper method to create a mlx_closure from a Rust closure.
fn new_mlx_closure<'a, F>(closure: F) -> mlx_sys::mlx_closure
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // Box the closure to keep it on the heap
    let boxed = Box::new(closure);

    // Create a raw pointer from the Box, transferring ownership to C
    let raw = Box::into_raw(boxed);
    let payload = raw as *mut std::ffi::c_void;

    unsafe {
        mlx_sys::mlx_closure_new_func_payload(Some(trampoline::<F>), payload, Some(noop_dtor))
    }
}

fn new_mlx_fallible_closure<'a, F>(closure: F) -> mlx_sys::mlx_closure
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let boxed = Box::new(closure);
    let raw = Box::into_raw(boxed);
    let payload = raw as *mut std::ffi::c_void;

    unsafe {
        mlx_sys::mlx_closure_new_func_payload(
            Some(trampoline_fallible::<F>),
            payload,
            Some(noop_dtor),
        )
    }
}

/// Function to create a new (+1 reference) mlx_vector_array from a vector of Array
fn new_mlx_vector_array(arrays: Vec<Array>) -> mlx_sys::mlx_vector_array {
    unsafe {
        let result = mlx_sys::mlx_vector_array_new();
        let ctx_ptrs: Vec<mlx_sys::mlx_array> = arrays.iter().map(|array| array.as_ptr()).collect();
        mlx_sys::mlx_vector_array_append_data(result, ctx_ptrs.as_ptr(), arrays.len());
        result
    }
}

fn mlx_vector_array_values(
    vector_array: mlx_sys::mlx_vector_array,
) -> Result<Vec<Array>, Exception> {
    unsafe {
        let size = mlx_sys::mlx_vector_array_size(vector_array);
        (0..size)
            .map(|index| {
                Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, vector_array, index))
            })
            .collect()
    }
}

extern "C" fn trampoline<'a, F>(
    ret: *mut mlx_vector_array,
    vector_array: mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    unsafe {
        let raw_closure: *mut F = payload as *mut _;
        // Let the box take care of freeing the closure
        let mut closure = Box::from_raw(raw_closure);
        let arrays = match mlx_vector_array_values(vector_array) {
            Ok(arrays) => arrays,
            Err(_) => {
                return FAILURE;
            }
        };
        let result = closure(&arrays);
        // We should probably keep using new_mlx_vector_array here instead of VectorArray
        // since we probably don't want to drop the arrays in the closure
        *ret = new_mlx_vector_array(result);

        SUCCESS
    }
}

extern "C" fn trampoline_fallible<'a, F>(
    ret: *mut mlx_vector_array,
    vector_array: mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    unsafe {
        let raw_closure: *mut F = payload as *mut _;
        let mut closure = Box::from_raw(raw_closure);
        let arrays = match mlx_vector_array_values(vector_array) {
            Ok(arrays) => arrays,
            Err(_) => {
                return FAILURE;
            }
        };
        let result = closure(&arrays);
        match result {
            Ok(result) => {
                *ret = new_mlx_vector_array(result);
                SUCCESS
            }
            Err(_) => FAILURE,
        }
    }
}

extern "C" fn noop_dtor(_data: *mut std::ffi::c_void) {}

pub(crate) fn get_mut_or_insert_with<'a, T>(
    map: &'a mut HashMap<Rc<str>, T>,
    key: &Rc<str>,
    f: impl FnOnce() -> T,
) -> &'a mut T {
    if !map.contains_key(key) {
        map.insert(key.clone(), f());
    }

    map.get_mut(key).unwrap()
}

/// Helper trait for compiling a function that takes a Module and/or an Optimizer
/// 
/// This is automatically implemented for all types that implement ModuleParameters
pub trait Updatable {
    fn updatable_parameters(&self) -> Vec<&Array>;
}

impl<T> Updatable for T
where 
    T: ModuleParameters 
{
    fn updatable_parameters(&self) -> Vec<&Array> {
        self.parameters().flatten().values().copied().collect()
    }
}
