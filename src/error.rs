use std::{cell::RefCell, ffi::c_char};

use crate::Dtype;
use thiserror::Error;

#[derive(Error, PartialEq, Debug)]
pub enum MlxError {
    #[error("data store error: {0}")]
    DataStore(#[from] DataStoreError),
    #[error("operation error: {0}")]
    Operation(#[from] OperationError),
    #[error("as slice error: {0}")]
    AsSlice(#[from] AsSliceError),
}

#[derive(Error, PartialEq, Debug)]
pub enum DataStoreError {
    #[error("negative dimension: {0}")]
    NegativeDimensions(String),

    #[error("negative integer: {0}")]
    NegativeInteger(String),

    #[error("broadcast error")]
    BroadcastError,

    #[error("not a scalar array")]
    NotScalar,
}

#[derive(Error, PartialEq, Debug)]
pub enum OperationError {
    #[error("operation not supported: {0}")]
    NotSupported(String),

    #[error("wrong input: {0}")]
    WrongInput(String),

    #[error("wrong dimensions: {0}")]
    WrongDimensions(String),

    #[error("axis out of bounds: axis {axis} for array with {dim} dimensions")]
    AxisOutOfBounds { axis: i32, dim: usize },
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

#[derive(Error, Debug, PartialEq)]
pub enum FftError {
    #[error("fftn requires at least one dimension")]
    ScalarArray,

    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("Shape and axes/axis have different sizes")]
    IncompatibleShapeAndAxes { shape_size: usize, axes_size: usize },

    #[error("Received duplicate axis")]
    DuplicateAxis,

    #[error("Invalid output size requested")]
    InvalidOutputSize,
}

#[derive(Error, Debug, PartialEq)]
#[error("Invalid axis {axis} for array with {ndim} dimensions")]
pub struct InvalidAxisError {
    pub axis: i32,
    pub ndim: usize,
}

#[derive(Debug, PartialEq, Error)]
pub enum TakeError {
    #[error("Cannot do a non-empty take from an array with zero elements.")]
    NonEmptyTakeFromEmptyArray,

    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),
}

#[derive(Debug, PartialEq, Error)]
pub enum TakeAlongAxisError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error(
        "Indices of dimension {indices_ndim} does not match the array of dimension {array_ndim}"
    )]
    IndicesDimensionMismatch {
        array_ndim: usize,
        indices_ndim: usize,
    },
}

#[derive(Debug, PartialEq, Error)]
pub enum ExpandDimsError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("Received duplicate axis")]
    DuplicateAxis,
}

#[derive(Debug, Error)]
#[error("Invalid number of indices or strides for array with dimension {ndim}")]
pub struct SliceError {
    pub ndim: usize,
}

#[derive(Debug, Error)]
pub enum FlattenError {
    #[error("Start axis must be less than or equal to end axis. Found start: {start}, end: {end}")]
    StartAxisGreaterThanEndAxis { start: i32, end: i32 },

    #[error(transparent)]
    InvalidStartAxis(InvalidAxisError),

    #[error(transparent)]
    InvalidEndAxis(InvalidAxisError),
}

#[derive(Debug, Error)]
pub enum ReshapeError<'a> {
    #[error("Can only infer one dimension")]
    MultipleInferredDims,

    #[error("Cannot infer the shape of an empty array")]
    EmptyArray,

    #[error("Cannot reshape array of size {size} into shape {shape:?}")]
    InvalidShape { size: usize, shape: &'a [i32] },
}

#[derive(Debug, Error)]
pub enum SqueezeError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("Cannot squeeze axis {axis} with size {size} which is not equal to 1")]
    AxisSizeGreaterThanOne { axis: i32, size: i32 },

    #[error("Received duplicate axis")]
    DuplicateAxis,
}

#[derive(Debug, Error)]
pub enum TransposeError {
    #[error("Received {num_axes} axes for array with {ndim} dimensions")]
    InvalidArgument { num_axes: usize, ndim: usize },

    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("Received duplicate axis")]
    DuplicateAxis,
}

#[derive(Debug, Error)]
#[error("Cannot broadcast array of shape {src_shape:?} into shape {dst_shape:?}")]
pub struct BroadcastError<'a> {
    pub src_shape: &'a [i32],
    pub dst_shape: &'a [i32],
}

#[derive(Debug, Error)]
pub enum ConcatenateError {
    #[error("No arrays provided for concatenation")]
    NoInputArray,

    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are {0:?}")]
    InvalidShapes(Vec<Vec<i32>>),
}

#[derive(Debug, Error)]
pub enum PadError {
    #[error("Invalid number of padding sizes passed to pad with axes of size {axes_size}")]
    InvalidWidths { axes_size: usize },

    #[error("Invalid padding size {size:?} for axis {axis}. Padding sizes must be non-negative")]
    NegativeWidth { axis: usize, size: (i32, i32) },
}

// #[derive(Debug, Error)]
// pub enum StackError {
//     #[error("No arrays provided for stacking")]
//     NoInputArray,

//     #[error("All arrays must have the same shape")]
//     InvalidShapes,
// }

// #[derive(Debug, Error)]
// pub enum SplitEqualError {
//     #[error(transparent)]
//     InvalidAxis(#[from] InvalidAxisError),

//     #[error("Cannot split array of size {size} into {num_splits} equal parts")]
//     InvalidNumSplits { size: usize, num_splits: i32 },
// }

#[derive(Debug, Error)]
#[error("{what}")]
pub struct Exception {
    what: String,
}

thread_local! {
    pub static LAST_MLX_ERROR: RefCell<Option<String>> = RefCell::new(None);
    pub static IS_HANDLER_SET: RefCell<bool> = RefCell::new(false);
}

#[no_mangle]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    LAST_MLX_ERROR.with(|last_error| {
        let mut last_error = last_error.borrow_mut();
        let c_str = unsafe { std::ffi::CStr::from_ptr(msg) };
        *last_error = Some(c_str.to_string_lossy().into_owned());
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

    IS_HANDLER_SET.with(|is_set| *is_set.borrow_mut() = true);
}

pub(crate) fn is_mlx_error_handler_set() -> bool {
    IS_HANDLER_SET.with(|is_set| *is_set.borrow())
}

pub(crate) fn get_and_clear_last_mlx_error() -> Option<Exception> {
    LAST_MLX_ERROR.with(|last_error| {
        let mut last_error = last_error.borrow_mut();
        let last_error = last_error.take();
        last_error.map(|what| Exception { what })
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_set_error_handler() {
        super::setup_mlx_error_handler();
        assert!(super::is_mlx_error_handler_set());
    }
}
