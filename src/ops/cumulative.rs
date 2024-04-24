use crate::error::OperationError;
use crate::{Array, StreamOrDevice};
use mlx_macros::default_device;

impl Array {
    /// Return the cumulative maximum of the elements along the given axis.
    ///
    /// # Example
    /// ```rust
    /// use mlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(0, None, None);
    /// ```
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    #[default_device]
    pub fn cummax_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_cummax_device(axis, reverse, inclusive, stream)
            .unwrap()
    }

    /// Return the cumulative maximum of the elements along the given axis without checking the inputs.
    ///
    /// # Example
    /// ```rust
    /// use mlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = unsafe { array.cummax_unchecked(0, None, None) };
    /// ```
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check that the axis is within the bounds of the array.
    /// It also does not check that the array can be reshaped to a flat array if the axis is not specified.
    #[default_device]
    pub unsafe fn cummax_device_unchecked(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            match axis.into() {
                Some(axis) => Array::from_ptr(mlx_sys::mlx_cummax(
                    self.c_array,
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )),
                None => {
                    // we make this an array instead of using the pointer directly
                    // so that Rust will drop it when it goes out of scope
                    let shape = &[-1];
                    let flat = Array::from_ptr(mlx_sys::mlx_reshape(
                        self.c_array,
                        shape.as_ptr(),
                        1,
                        stream.as_ptr(),
                    ));

                    Array::from_ptr(mlx_sys::mlx_cummax(
                        flat.c_array,
                        0,
                        reverse.into().unwrap_or(false),
                        inclusive.into().unwrap_or(true),
                        stream.as_ptr(),
                    ))
                }
            }
        }
    }

    /// Return the cumulative maximum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Example
    /// ```rust
    /// use mlx::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.try_cummax(0, None, None).unwrap();
    /// ```
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    #[default_device]
    pub fn try_cummax_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Result<Array, OperationError> {
        let axis = axis.into();

        if let Some(axis) = axis.into() {
            if axis >= self.ndim() as i32 || axis < -(self.ndim() as i32) {
                return Err(OperationError::AxisOutOfBounds {
                    axis,
                    dim: self.ndim(),
                });
            }
        }

        Ok(unsafe { self.cummax_device_unchecked(axis, reverse, inclusive, stream) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_cummax() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let mut result = array.cummax(0, None, None);
        result.eval();
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);

        let mut result = array.cummax(1, None, None);
        result.eval();
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 9]);

        let mut result = array.cummax(None, None, None);
        result.eval();
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 8, 9]);

        let mut result = array.cummax(0, Some(true), None);
        result.eval();
        assert_eq!(result.as_slice::<i32>(), &[5, 9, 4, 9]);

        let mut result = array.cummax(0, None, Some(true));
        result.eval();
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);
    }

    #[test]
    fn test_cummax_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.try_cummax(2, None, None);
        assert!(result.is_err());
    }
}
