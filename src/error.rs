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

    #[error("Duplicate axis received: {axis}")]
    DuplicateAxis { axis: i32 },

    #[error("Invalid output size requested")]
    InvalidOutputSize,
}

#[derive(Error, Debug, PartialEq)]
#[error("Invalid axis {axis} for array with {ndim} dimensions")]
pub struct InvalidAxisError {
    pub axis: i32,
    pub ndim: usize,
}

#[derive(Error, Debug, PartialEq)]
#[error("Received duplicate axis {axis}")]
pub struct DuplicateAxisError {
    pub axis: i32,
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

    #[error(transparent)]
    DuplicateAxis(#[from] DuplicateAxisError),
}

#[derive(Debug, Error)]
#[error("Invalid number of indices or strides for array with dimension {ndim}")]
pub struct SliceError {
    pub ndim: usize,
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
