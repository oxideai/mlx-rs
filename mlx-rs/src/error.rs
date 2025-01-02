//! Custom error types and handler for the c ffi

use crate::Dtype;
use libc::strdup;
use std::convert::Infallible;
use std::ffi::NulError;
use std::sync::Once;
use std::{cell::Cell, ffi::c_char};
use thiserror::Error;

/// Type alias for a `Result` with an `Exception` error type.
pub type Result<T> = std::result::Result<T, Exception>;

/// Error with io operations
#[derive(Error, PartialEq, Debug)]
pub enum IoError {
    /// Path must point to a local file
    #[error("Path must point to a local file")]
    NotFile,

    /// Path contains invalid UTF-8
    #[error("Path contains invalid UTF-8")]
    InvalidUtf8,

    /// Path contains null bytes
    #[error("Path contains null bytes")]
    NullBytes,

    /// No file extension found
    #[error("No file extension found")]
    NoExtension,

    /// Unsupported file format
    #[error("Unsupported file format")]
    UnsupportedFormat,

    /// Unable to open file
    #[error("Unable to open file")]
    UnableToOpenFile,

    /// Unable to allocate memory
    #[error("Unable to allocate memory")]
    AllocationError,

    /// Null error
    #[error(transparent)]
    NulError(#[from] NulError),

    /// Exception
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
    DtypeMismatch {
        /// The expected data type.
        expecting: Dtype,

        /// The actual data type
        found: Dtype,
    },

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Exception. Most will come from the C API.
#[derive(Debug, PartialEq, Error)]
#[error("{what:?}")]
pub struct Exception {
    pub(crate) what: String,
}

impl Exception {
    /// The error message.
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

impl From<Infallible> for Exception {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<Exception> for String {
    fn from(e: Exception) -> Self {
        e.what
    }
}

thread_local! {
    static CLOSURE_ERROR: Cell<Option<Exception>> = const { Cell::new(None) };
    static LAST_MLX_ERROR: Cell<*const c_char> = const { Cell::new(std::ptr::null()) };
    pub(crate) static INIT_ERR_HANDLER: Once = const { Once::new() };
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

pub(crate) fn setup_mlx_error_handler() {
    let handler = default_mlx_error_handler;
    let data_ptr = LAST_MLX_ERROR.with(|last_error| last_error.as_ptr() as *mut std::ffi::c_void);
    let dtor = noop_mlx_error_handler_data_deleter;
    unsafe {
        mlx_sys::mlx_set_error_handler(Some(handler), data_ptr, Some(dtor));
    }
}

pub(crate) fn set_closure_error(err: Exception) {
    CLOSURE_ERROR.with(|closure_error| closure_error.set(Some(err)));
}

pub(crate) fn get_and_clear_closure_error() -> Option<Exception> {
    CLOSURE_ERROR.with(|closure_error| closure_error.replace(None))
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

/// Error with building a cross-entropy loss function
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrossEntropyBuildError {
    /// Label smoothing factor must be in the range [0, 1)
    #[error("Label smoothing factor must be in the range [0, 1)")]
    InvalidLabelSmoothingFactor,
}

impl From<CrossEntropyBuildError> for Exception {
    fn from(value: CrossEntropyBuildError) -> Self {
        Exception::custom(format!("{}", value))
    }
}

/// Error with building a RmsProp optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RmsPropBuildError {
    /// Alpha must be non-negative
    #[error("alpha must be non-negative")]
    NegativeAlpha,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEpsilon,
}

/// Error with building an AdaDelta optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdaDeltaBuildError {
    /// Rho must be non-negative
    #[error("rho must be non-negative")]
    NegativeRho,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEps,
}

/// Error with building an Adafactor optimizer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdafactorBuildError {
    /// Either learning rate is provided or relative step is set to true.
    #[error("Either learning rate is provided or relative step is set to true")]
    LrIsNoneAndRelativeStepIsFalse,
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
