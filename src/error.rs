use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataStoreError {
    #[error("negative dimension: {0}")]
    NegativeDimensions(String),

    #[error("negative integer: {0}")]
    NegativeInteger(String),
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
    #[error("Desired output dtype does not match the data type of the array.")]
    DtypeMismatch,
}

#[derive(Error, Debug)]
pub enum FftError {
    #[error("fftn requires at least one dimension")]
    ScalarArray,

    #[error("Invalid axis received for array with {0} dimensions")]
    InvalidAxis(usize),
}