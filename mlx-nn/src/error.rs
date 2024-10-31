//! Custom error types for mlx-nn

use mlx_rs::error::Exception;
use thiserror::Error;

/// Error with building a cross-entropy loss function
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrossEntropyBuildError {
    /// Label smoothing factor must be in the range [0, 1)
    #[error("Label smoothing factor must be in the range [0, 1)")]
    InvalidLabelSmoothingFactor,
}

impl From<CrossEntropyBuildError> for Exception {
    fn from(value: CrossEntropyBuildError) -> Self {
        Exception::custom(format!("{}", value))
    }
}

/// Error with building a RmsProp optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RmsPropBuildError {
    /// Alpha must be non-negative
    #[error("alpha must be non-negative")]
    NegativeAlpha,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEpsilon,
}

/// Error with building a dropout layer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DropoutBuildError {
    /// Dropout probability must be in the range [0, 1)
    #[error("Dropout probability must be in the range [0, 1)")]
    InvalidProbability,
}

/// Error with building an AdaDelta optimizer
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdaDeltaBuildError {
    /// Rho must be non-negative
    #[error("rho must be non-negative")]
    NegativeRho,

    /// Epsilon must be non-negative
    #[error("epsilon must be non-negative")]
    NegativeEps,
}
