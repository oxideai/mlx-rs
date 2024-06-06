use std::os::raw::c_void;

use mlx_sys::mlx_vector_array;

use crate::Array;

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
    c_vec: mlx_vector_array,
}

impl VectorArray {
    pub(crate) fn as_ptr(&self) -> mlx_vector_array {
        self.c_vec
    }

    pub(crate) unsafe fn from_ptr(c_vec: mlx_vector_array) -> Self {
        Self { c_vec }
    }

    pub(crate) fn from_iter(iter: impl Iterator<Item = impl AsRef<Array>>) -> Self {
        unsafe {
            let c_vec = mlx_sys::mlx_vector_array_new();
            for arr in iter {
                mlx_sys::mlx_vector_array_add(c_vec, arr.as_ref().as_ptr())
            }
            Self { c_vec }
        }
    }

    pub(crate) fn into_values(self) -> Vec<Array> {
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.c_vec);
            (0..size)
                .map(|i| {
                    let c_array = mlx_sys::mlx_vector_array_get(self.c_vec, i);
                    Array::from_ptr(c_array)
                })
                .collect()
        }
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_vec as *mut c_void) }
    }
}

/// A custom type for internal use with `Array` only that is essentially `Cow` but doens't require
/// the `Clone`
pub(crate) enum OwnedOrRef<'a, T> {
    Owned(T),
    Ref(&'a T),
}

impl<'a, T> AsRef<T> for OwnedOrRef<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            OwnedOrRef::Owned(array) => array,
            OwnedOrRef::Ref(array) => array,
        }
    }
}

impl<'a, T> std::ops::Deref for OwnedOrRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
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

