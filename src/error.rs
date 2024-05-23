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

#[derive(Debug, Error)]
pub enum StackError {
    #[error("No arrays provided for stacking")]
    NoInputArray,

    #[error("All arrays must have the same shape")]
    InvalidShapes,
}

#[derive(Debug, Error)]
pub enum SplitEqualError {
    #[error(transparent)]
    InvalidAxis(#[from] InvalidAxisError),

    #[error("Cannot split array of size {size} into {num_splits} equal parts")]
    InvalidNumSplits { size: usize, num_splits: i32 },
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
