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
}

#[derive(Error, PartialEq, Debug)]
pub enum OperationError {
    #[error("operation not supported: {0}")]
    NotSupported(String),

    #[error("wrong input: {0}")]
    WrongInput(String),

    #[error("wrong dimensions: {0}")]
    WrongDimensions(String),

    #[error("axis out of bounds: {0}")]
    AxisOutOfBounds(String),
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
pub enum FftnError {
    #[error("fftn requires at least one dimension")]
    ScalarArray,

    #[error("Invalid axis received for array with {ndim} dimensions")]
    InvalidAxis { ndim: usize },

    #[error("Shape and axis have different sizes")]
    ShapeAxisMismatch,

    #[error("Duplcated axis received: {axis}")]
    DuplicateAxis { axis: i32 },
}
