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

    #[error("Invalid axis received for array with {ndim} dimensions")]
    InvalidAxis { ndim: usize },

    #[error("Shape and axes/axis have different sizes")]
    IncompatibleShapeAndAxes { shape_size: usize, axes_size: usize },

    #[error("Duplicate axis received: {axis}")]
    DuplicateAxis { axis: i32 },

    #[error("Invalid output size requested")]
    InvalidOutputSize,
}

#[derive(Error, Debug, PartialEq)]
pub enum LinAlgError {
    #[error("Too many axes for norm operation")]
    TooManyAxes,

    #[error("Singular value norms are not implemented")]
    SingularValueNormNotImplemented,

    #[error("Matrix norm with ord={ord} is not supported")]
    InvalidMatrixF64Ord { ord: f64 },

    #[error("Matrix norm with ord={ord} is not supported")]
    InvalidMatrixStrOrd { ord: &'static str },

    #[error("Norm ord={ord} only supported for matrices")]
    RequiresMatrix { ord: &'static str },

    #[error("Nuclear norm is not implemented")]
    NotYetImplemented,
}
