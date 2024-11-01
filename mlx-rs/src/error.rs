use std::{cell::Cell, ffi::c_char};

use crate::Dtype;
use libc::strdup;
use thiserror::Error;

#[derive(Error, PartialEq, Debug)]
pub enum ItemError {
    #[error("not a scalar array")]
    NotScalar,

    #[error(transparent)]
    Exception(#[from] Exception),
}

#[derive(Error, PartialEq, Debug)]
pub enum IoError {
    #[error("Path must point to a local file")]
    NotFile,

    #[error("Path contains invalid UTF-8")]
    InvalidUtf8,

    #[error("Path contains null bytes")]
    NullBytes,

    #[error("No file extension found")]
    NoExtension,

    #[error("Unsupported file format")]
    UnsupportedFormat,

    #[error("Unable to open file")]
    UnableToOpenFile,

    #[error("Unable to allocate memory")]
    AllocationError,

    #[error(transparent)]
    Exception(#[from] Exception),
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

    #[error(transparent)]
    Exception(#[from] Exception),
}

#[derive(Debug, PartialEq, Error)]
#[error("{what:?}")]
pub struct Exception {
    pub(crate) what: String,
}

impl Exception {
    pub fn what(&self) -> &str {
        &self.what
    }

    /// Creates a new exception with the given message.
    pub fn custom(what: impl Into<String>) -> Self {
        Self { what: what.into() }
    }
}

impl From<&str> for Exception {
    fn from(what: &str) -> Self {
        Self {
            what: what.to_string(),
        }
    }
}

thread_local! {
    static LAST_MLX_ERROR: Cell<*const c_char> = const { Cell::new(std::ptr::null()) };
    static IS_HANDLER_SET: Cell<bool> = const { Cell::new(false) };
}

#[no_mangle]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    unsafe {
        LAST_MLX_ERROR.with(|last_error| {
            last_error.set(strdup(msg));
        });
    }
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

        let last_err = unsafe {
            std::ffi::CStr::from_ptr(last_err_ptr)
                .to_string_lossy()
                .into_owned()
        };
        unsafe {
            libc::free(last_err_ptr as *mut libc::c_void);
        }
        Some(Exception { what: last_err })
    })
}

#[cfg(test)]
mod tests {
    use crate::array;

    #[test]
    fn test_exception() {
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);

        let result = a.add(&b);
        let error = result.expect_err("Expected error");

        // The full error message would also contain the full path to the original c++ file,
        // so we just check for a substring
        assert!(error
            .what()
            .contains("Shapes (3) and (2) cannot be broadcast."))
    }
}
