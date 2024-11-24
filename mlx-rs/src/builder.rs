//! Defines helper traits for builder pattern

/// Helper trait for buildable types
pub trait Buildable: Sized {
    type Builder: Builder<Self>;
}

/// Helper trait for builder
pub trait Builder<T> {
    type Error: std::error::Error;

    fn build(self) -> Result<T, Self::Error>;
}