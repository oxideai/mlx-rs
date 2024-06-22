use mlx_macros::default_device;

use crate::{Array, Stream, StreamOrDevice};

impl Array {
    /// Extract a diagonal or construct a diagonal matrix.
    ///
    /// If self is 1-D then a diagonal matrix is constructed with self on the `k`-th diagonal. If
    /// self is 2-D then the `k`-th diagonal is returned.
    ///
    /// # Params:
    /// 
    /// - `k`: the diagonal to extract or construct
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diag_device(&self, k: impl Into<Option<i32>>, stream: impl AsRef<Stream>) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_diag(
                self.c_array,
                k.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            ))
        }
    }

    /// Return specified diagonals.
    ///
    /// If self is 2-D, then a 1-D array containing the diagonal at the given `offset` is returned.
    ///
    /// If self has more than two dimensions, then `axis1` and `axis2` determine the 2D subarrays
    /// from which diagonals are extracted. The new shape is the original shape with `axis1` and
    /// `axis2` removed and a new dimension inserted at the end corresponding to the diagonal.
    ///
    /// # Params:
    /// 
    /// - `offset`: offset of the diagonal.  Can be positive or negative
    /// - `axis1`: first axis of the 2-D sub-array from which the diagonals should be taken
    /// - `axis2`: second axis of the 2-D sub-array from which the diagonals should be taken
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diagonal_device(
        &self,
        offset: impl Into<Option<i32>>,
        axis1: impl Into<Option<i32>>,
        axis2: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_diagonal(
                self.c_array,
                offset.into().unwrap_or(0),
                axis1.into().unwrap_or(0),
                axis2.into().unwrap_or(1),
                stream.as_ref().as_ptr(),
            ))
        }
    }
}

/// See [`Array::diag`]
#[default_device]
pub fn diag_device(a: &Array, k: impl Into<Option<i32>>, stream: impl AsRef<Stream>) -> Array {
    a.diag_device(k, stream)
}

/// See [`Array::diagonal`]
#[default_device]
pub fn diagonal_device(
    a: &Array,
    offset: impl Into<Option<i32>>,
    axis1: impl Into<Option<i32>>,
    axis2: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    a.diagonal_device(offset, axis1, axis2, stream)
}