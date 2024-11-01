use crate::{complex64, error::Exception, Array, FromNested};
use mlx_sys::mlx_tuple_array_array;
use std::collections::HashMap;
use std::ffi::CStr;
use std::{ffi::NulError, marker::PhantomData, os::raw::c_void};

/// Helper method to get a string representation of an mlx object.
pub(crate) fn mlx_describe(ptr: *mut ::std::os::raw::c_void) -> Option<String> {
    let mlx_description = unsafe { mlx_sys::mlx_tostring(ptr) };
    let c_str = unsafe { mlx_sys::mlx_string_data(mlx_description) };

    let description = if c_str.is_null() {
        None
    } else {
        Some(unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        })
    };

    unsafe { mlx_sys::mlx_free(mlx_description as *mut std::ffi::c_void) };

    description
}

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

    pub(crate) unsafe fn from_ptr(c_vec: mlx_sys::mlx_vector_array) -> Self {
        Self { c_vec }
    }

    pub(crate) fn from_iter(iter: impl Iterator<Item = impl AsRef<Array>>) -> Self {
        unsafe {
            let c_vec = mlx_sys::mlx_vector_array_new();
            for arr in iter {
                mlx_sys::mlx_vector_array_add_value(c_vec, arr.as_ref().as_ptr())
            }
            Self { c_vec }
        }
    }

    pub(crate) fn into_values<T>(self) -> T
    where
        T: FromIterator<Array>,
    {
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.c_vec);
            (0..size)
                .map(|i| {
                    let c_array = mlx_sys::mlx_vector_array_get(self.c_vec, i);
                    Array::from_ptr(c_array)
                })
                .collect::<T>()
        }
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_vec as *mut c_void) }
    }
}

pub(crate) struct MlxString(mlx_sys::mlx_string);

impl MlxString {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_string {
        self.0
    }
}

impl<'a> TryFrom<&'a str> for MlxString {
    type Error = NulError;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let c_str = std::ffi::CString::new(s)?;
        let ptr = unsafe { mlx_sys::mlx_string_new(c_str.as_ptr()) };
        Ok(Self(ptr))
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.0 as *mut c_void) }
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

impl<'a> ScalarOrArray<'a> for Array {
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
    pub(crate) fn from_ptr(c_closure: mlx_sys::mlx_closure) -> Self {
        Self {
            c_closure,
            lt_marker: PhantomData,
        }
    }

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

impl<'a> Drop for Closure<'a> {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_closure as *mut _) }
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
        mlx_sys::mlx_closure_new_with_payload(Some(trampoline::<F>), payload, Some(noop_dtor))
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
        mlx_sys::mlx_fallible_closure_new_with_payload(
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
        mlx_sys::mlx_vector_array_add_data(result, ctx_ptrs.as_ptr(), arrays.len());
        result
    }
}

fn mlx_vector_array_values(vector_array: mlx_sys::mlx_vector_array) -> Vec<Array> {
    unsafe {
        let size = mlx_sys::mlx_vector_array_size(vector_array);
        (0..size)
            .map(|index| {
                // ctx is a +1 reference, the array takes ownership
                let c_array = mlx_sys::mlx_vector_array_get(vector_array, index);
                Array::from_ptr(c_array)
            })
            .collect()
    }
}

extern "C" fn trampoline<'a, F>(
    vector_array: mlx_sys::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> mlx_sys::mlx_vector_array
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    unsafe {
        let raw_closure: *mut F = payload as *mut _;
        // Let the box take care of freeing the closure
        let mut closure = Box::from_raw(raw_closure);
        let arrays = mlx_vector_array_values(vector_array);
        let result = closure(&arrays);
        // We should probably keep using new_mlx_vector_array here instead of VectorArray
        // since we probably don't want to drop the arrays in the closure
        new_mlx_vector_array(result)
    }
}

extern "C" fn trampoline_fallible<'a, F>(
    vector_array: mlx_sys::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> mlx_sys::mlx_vector_array_result
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    use std::ffi::CString;

    unsafe {
        let raw_closure: *mut F = payload as *mut _;
        let mut closure = Box::from_raw(raw_closure);
        let arrays = mlx_vector_array_values(vector_array);
        let result = closure(&arrays);
        match result {
            Ok(result) => {
                let c_result = new_mlx_vector_array(result);
                mlx_sys::mlx_vector_array_result_new_ok(c_result)
            }
            Err(exception) => {
                let what = CString::new(exception.what).unwrap();
                let mlx_string = mlx_sys::mlx_string_new(what.as_ptr());
                mlx_sys::mlx_vector_array_result_new_err(mlx_string)
            }
        }
    }
}

extern "C" fn noop_dtor(_data: *mut std::ffi::c_void) {}

pub(crate) struct TupleArrayArray {
    c_tuple: mlx_tuple_array_array,
}

impl Drop for TupleArrayArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_tuple as *mut c_void) }
    }
}

impl TupleArrayArray {
    pub(crate) unsafe fn from_ptr(c_tuple: mlx_tuple_array_array) -> Self {
        Self { c_tuple }
    }

    pub(crate) fn into_values(self) -> (Array, Array) {
        unsafe {
            let c_array1 = mlx_sys::mlx_tuple_array_array_get_0(self.c_tuple);
            let c_array2 = mlx_sys::mlx_tuple_array_array_get_1(self.c_tuple);
            (Array::from_ptr(c_array1), Array::from_ptr(c_array2))
        }
    }
}

pub(crate) struct TupleArrayArrayArray {
    c_tuple: mlx_sys::mlx_tuple_array_array_array,
}

impl Drop for TupleArrayArrayArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_tuple as *mut c_void) }
    }
}

impl TupleArrayArrayArray {
    pub(crate) unsafe fn from_ptr(c_tuple: mlx_sys::mlx_tuple_array_array_array) -> Self {
        Self { c_tuple }
    }

    pub(crate) fn into_values(self) -> (Array, Array, Array) {
        unsafe {
            let c_array1 = mlx_sys::mlx_tuple_array_array_array_get_0(self.c_tuple);
            let c_array2 = mlx_sys::mlx_tuple_array_array_array_get_1(self.c_tuple);
            let c_array3 = mlx_sys::mlx_tuple_array_array_array_get_2(self.c_tuple);
            (
                Array::from_ptr(c_array1),
                Array::from_ptr(c_array2),
                Array::from_ptr(c_array3),
            )
        }
    }
}

pub(crate) struct TupleVectorArrayVectorArray {
    c_tuple: mlx_sys::mlx_tuple_vector_array_vector_array,
}

impl Drop for TupleVectorArrayVectorArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_tuple as *mut c_void) }
    }
}

impl TupleVectorArrayVectorArray {
    pub(crate) unsafe fn from_ptr(c_tuple: mlx_sys::mlx_tuple_vector_array_vector_array) -> Self {
        Self { c_tuple }
    }

    pub(crate) fn into_values(self) -> (Vec<Array>, Vec<Array>) {
        unsafe {
            let c_array1 = mlx_sys::mlx_tuple_vector_array_vector_array_get_0(self.c_tuple);
            let c_array2 = mlx_sys::mlx_tuple_vector_array_vector_array_get_1(self.c_tuple);
            let array1 = VectorArray::from_ptr(c_array1).into_values();
            let array2 = VectorArray::from_ptr(c_array2).into_values();

            (array1, array2)
        }
    }
}

pub(crate) fn mlx_map_array_values(
    mlx_map: mlx_sys::mlx_map_string_to_array,
) -> HashMap<String, Array> {
    let mut result = HashMap::new();

    unsafe {
        let iterator = mlx_sys::mlx_map_string_to_array_iterate(mlx_map);

        while !mlx_sys::mlx_map_string_to_array_iterator_end(iterator) {
            let mlx_key = mlx_sys::mlx_map_string_to_array_iterator_key(iterator);
            let key = CStr::from_ptr(mlx_sys::mlx_string_data(mlx_key))
                .to_string_lossy()
                .into_owned();

            let mlx_array_ctx = mlx_sys::mlx_map_string_to_array_iterator_value(iterator);
            let array = Array::from_ptr(mlx_array_ctx);
            result.insert(key, array);

            mlx_sys::mlx_free(mlx_key as *mut c_void);
            mlx_sys::mlx_map_string_to_array_iterator_next(iterator);
        }

        mlx_sys::free(iterator as *mut c_void);
    }

    result
}

pub(crate) fn mlx_map_string_values(
    mlx_map: mlx_sys::mlx_map_string_to_string,
) -> HashMap<String, String> {
    let mut result = HashMap::new();

    unsafe {
        let iterator = mlx_sys::mlx_map_string_to_string_iterate(mlx_map);

        while !mlx_sys::mlx_map_string_to_string_iterator_end(iterator) {
            let mlx_key = mlx_sys::mlx_map_string_to_string_iterator_key(iterator);
            let key = CStr::from_ptr(mlx_sys::mlx_string_data(mlx_key))
                .to_string_lossy()
                .into_owned();

            let mlx_value = mlx_sys::mlx_map_string_to_string_iterator_value(iterator);
            let value = CStr::from_ptr(mlx_sys::mlx_string_data(mlx_value))
                .to_string_lossy()
                .into_owned();

            result.insert(key, value);

            mlx_sys::mlx_free(mlx_key as *mut c_void);
            mlx_sys::mlx_free(mlx_value as *mut c_void);
            mlx_sys::mlx_map_string_to_string_iterator_next(iterator);
        }

        mlx_sys::mlx_free(iterator as *mut c_void);
    }

    result
}

pub(crate) fn new_mlx_array_map(
    hashmap: &HashMap<String, Array>,
) -> mlx_sys::mlx_map_string_to_array {
    let mlx_map = unsafe { mlx_sys::mlx_map_string_to_array_new() };

    for (key, array) in hashmap {
        let mlx_key = MlxString::try_from(key.as_str()).unwrap();
        unsafe {
            mlx_sys::mlx_map_string_to_array_insert(mlx_map, mlx_key.as_ptr(), array.as_ptr());
        }
    }

    mlx_map
}

pub(crate) fn new_mlx_string_map(
    hashmap: &HashMap<String, String>,
) -> mlx_sys::mlx_map_string_to_string {
    let mlx_map = unsafe { mlx_sys::mlx_map_string_to_string_new() };

    for (key, value) in hashmap {
        let mlx_key = MlxString::try_from(key.as_str()).unwrap();
        let mlx_value = MlxString::try_from(value.as_str()).unwrap();
        unsafe {
            mlx_sys::mlx_map_string_to_string_insert(mlx_map, mlx_key.as_ptr(), mlx_value.as_ptr());
        }
    }

    mlx_map
}
