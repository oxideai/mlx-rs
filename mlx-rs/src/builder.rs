//! Defines helper traits for builder pattern

/// Helper trait for buildable types
pub trait Buildable: Sized {
    /// The builder type for this buildable type
    type Builder: Builder<Self>;
}

/// Helper trait for builder
pub trait Builder<T: Buildable> {
    /// Error with building
    type Error: std::error::Error;

    /// Build the type
    fn build(self) -> Result<T, Self::Error>;
}
