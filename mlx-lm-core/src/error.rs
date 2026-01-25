//! Error types for mlx-lm-core

use mlx_rs::error::Exception;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Weight loading error: {0}")]
    LoadWeights(#[from] mlx_rs::error::IoError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

impl From<std::convert::Infallible> for Error {
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}

/// Convenience Result type alias
pub type Result<T> = std::result::Result<T, Error>;
