use hf_hub::api::sync::ApiError;
use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Api(#[from] ApiError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError)
}