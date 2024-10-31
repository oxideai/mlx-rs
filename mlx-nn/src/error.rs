//! Custom error types for mlx-nn

use thiserror::Error;

/// Error with building a dropout layer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DropoutBuildError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,
}
