//! Extension crate for the `mlx` with a focus on improving the ergonomics.

// We are re-exporting here so that we can refer to `mlx_rs` in the macros even if the user
// doesn't have it in their dependencies or renamed it.
pub use mlx_rs;

#[cfg(feature = "macros")]
mod macros;
