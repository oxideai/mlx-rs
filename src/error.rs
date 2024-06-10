use std::convert::Infallible;
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
#[error("{ord} norm is not implemented")]
pub struct OrdNotImplementedError<'a> {
    pub ord: &'a str,
}

impl From<Infallible> for OrdNotImplementedError<'static> {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

#[derive(Debug, Error)]
pub enum NormError<'a> {
    #[error(transparent)]
    Ord(OrdNotImplementedError<'a>),

    #[error("Too many axes for norm operation")]
    TooManyAxes,

    #[error("Singular value norms are not implemented")]
    SingularValueNormNotImplemented,

    #[error("Invalid ord {ord} for matrix norm")]
    InvalidMatrixOrd { ord: crate::linalg::Ord },

    #[error("Norm {ord} only supported for matrices")]
    OrdRequiresMatrix { ord: crate::linalg::Ord },

    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),
}

impl<'a> From<OrdNotImplementedError<'a>> for NormError<'a> {
    fn from(err: OrdNotImplementedError<'a>) -> Self {
        NormError::Ord(err)
    }
}

#[derive(Debug, Error)]
pub enum QrError {
    #[error("Arrays must type f32. Received array with dtype {dtype:?}")]
    DtypeNotSupported { dtype: Dtype },

    #[error("Arrays must have >= 2 dimensions. Received array with {ndim} dimensions")]
    InvalidShape { ndim: usize },

    #[error("Support for non-square matrices NYI")]
    NonSquareMatrix,
}

#[derive(Debug, Error)]
pub enum SvdError {
    #[error("Arrays must type f32. Received array with dtype {dtype:?}")]
    DtypeNotSupported { dtype: Dtype },

    #[error("Arrays must have >= 2 dimensions. Received array with {ndim} dimensions")]
    InvalidShape { ndim: usize },
}

#[derive(Debug, Error)]
pub enum InvError {
    #[error("Arrays must type f32. Received array with dtype {dtype:?}")]
    DtypeNotSupported { dtype: Dtype },

    #[error("Arrays must have >= 2 dimensions. Received array with {ndim} dimensions")]
    InvalidShape { ndim: usize },

    #[error("Support for non-square matrices NYI")]
    NonSquareMatrix,
}
