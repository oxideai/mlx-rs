//! Custom error types for mlx-nn

use mlx_rs::error::Exception;
use thiserror::Error;

/// Error with building a dropout layer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DropoutBuildError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,
}

/// Error with building a MultiHeadAttention module
#[derive(Debug, PartialEq, Error)]
pub enum MultiHeadAttentionBuildError {
    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

/// Error with building a transformer
#[derive(Debug, PartialEq, Error)]
pub enum TransformerBulidError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,

    /// Invalid number of heads
    #[error("Invalid number of heads: {0}")]
    InvalidNumHeads(i32),

    /// Exceptions
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<DropoutBuildError> for TransformerBulidError {
    fn from(e: DropoutBuildError) -> Self {
        match e {
            DropoutBuildError::InvalidProbability => Self::InvalidProbability,
        }
    }
}

impl From<MultiHeadAttentionBuildError> for TransformerBulidError {
    fn from(e: MultiHeadAttentionBuildError) -> Self {
        match e {
            MultiHeadAttentionBuildError::InvalidNumHeads(n) => Self::InvalidNumHeads(n),
            MultiHeadAttentionBuildError::Exception(e) => Self::Exception(e),
        }
    }
}
