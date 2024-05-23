use std::{cell::Cell, ffi::c_char};

use crate::Dtype;
use thiserror::Error;

#[derive(Error, PartialEq, Debug)]
pub enum ItemError {
    #[error("not a scalar array")]
    NotScalar,
}

/// Error associated with `Array::try_as_slice()`
#[derive(Debug, PartialEq, Error)]
pub enum AsSliceError {
    /// The underlying data pointer is null.
    ///
    /// This is likely because the array has not been evaluated yet.
    #[error("The data pointer is null.")]
    Null,

    /// The output dtype does not match the data type of the array.
    #[error("dtype mismatch: expected {expecting:?}, found {found:?}")]
    DtypeMismatch { expecting: Dtype, found: Dtype },
}

#[derive(Debug, Error)]
#[error("{what}")]
pub struct Exception {
    what: String,
}

thread_local! {
    pub static LAST_MLX_ERROR: Cell<*const c_char> = const { Cell::new(std::ptr::null()) };
    pub static IS_HANDLER_SET: Cell<bool> = const { Cell::new(false) };
}

#[no_mangle]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    LAST_MLX_ERROR.with(|last_error| {
        last_error.set(msg);
    });
}

#[no_mangle]
extern "C" fn noop_mlx_error_handler_data_deleter(_data: *mut std::ffi::c_void) {}

pub fn setup_mlx_error_handler() {
    let handler = default_mlx_error_handler;
    let data_ptr = LAST_MLX_ERROR.with(|last_error| last_error.as_ptr() as *mut std::ffi::c_void);
    let dtor = noop_mlx_error_handler_data_deleter;
    unsafe {
        mlx_sys::mlx_set_error_handler(Some(handler), data_ptr, Some(dtor));
    }

    IS_HANDLER_SET.with(|is_set| is_set.set(true));
}

pub(crate) fn is_mlx_error_handler_set() -> bool {
    IS_HANDLER_SET.with(|is_set| is_set.get())
}

pub(crate) fn get_and_clear_last_mlx_error() -> Option<Exception> {
    LAST_MLX_ERROR.with(|last_error| {
        let last_err_ptr = last_error.replace(std::ptr::null());
        if last_err_ptr.is_null() {
            return None;
        }

        let last_err = unsafe { std::ffi::CStr::from_ptr(last_err_ptr) };
        Some(Exception {
            what: last_err.to_string_lossy().into_owned(),
        })
    })
}

#[derive(Debug, Error)]
#[error(
    "GPU sort cannot handle sort axis of >= 2M elements, got array with sort axis size {size} "
)]
pub struct ArrayTooLargeForGpuError {
    pub size: usize,
}

#[derive(Debug, Error)]
pub enum SortError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error(transparent)]
    ArrayTooLargeForGpu(#[from] ArrayTooLargeForGpuError),
}

#[derive(Debug, Error)]
pub enum SortAllError {
    #[error(transparent)]
    ArrayTooLargeForGpu(#[from] ArrayTooLargeForGpuError),
}

#[derive(Debug, Error)]
#[error("Received invalid kth {kth} along axis {axis} for array with shape {shape:?}")]
pub struct InvalidKthError {
    pub kth: i32,
    pub axis: i32,
    pub shape: Vec<i32>,
}

#[derive(Debug, Error)]
pub enum PartitionError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error(transparent)]
    InvalidKth(#[from] InvalidKthError),
}

#[derive(Debug, Error)]
pub enum PartitionAllError {
    #[error(transparent)]
    InvalidKth(#[from] InvalidKthError),
}
