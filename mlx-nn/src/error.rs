//! Error types for mlx-nn.

use mlx_rs::error::Exception;
use thiserror::Error;

/// Error type for mlx-nn.
#[derive(Debug, Error)]
pub enum Error {
    /// Error from the MLX library.
    #[error(transparent)]
    Exception(#[from] Exception),

    /// Other error.
    #[error(transparent)]
    Other(Box<dyn std::error::Error>),
}

/// Error associated with the `Dropout2d` module.
#[derive(Debug, Error)]
pub enum Dropout2dError {
    /// Expecting `input.ndim()` to be 3 or 4.
    #[error("Expecting `input.ndim()` to be 3 or 4")]
    NdimNotSupported,
}

impl From<Dropout2dError> for Error {
    fn from(e: Dropout2dError) -> Self {
        Error::Other(Box::new(e))
    }
}

/// Error associated with the `Dropout3d` module.
#[derive(Debug, Error)]
pub enum Dropout3dError {
    /// Expecting `input.ndim()` to be 4 or 5.
    #[error("Expecting `input.ndim()` to be 4 or 5")]
    NdimNotSupported,
}

impl From<Dropout3dError> for Error {
    fn from(e: Dropout3dError) -> Self {
        Error::Other(Box::new(e))
    }
}