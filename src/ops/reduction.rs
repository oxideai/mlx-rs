use crate::array::Array;
use crate::error::OperationError;
use crate::stream::StreamOrDevice;
use mlx_macros::default_device;

impl Array {
    /// An `and` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.all(&[0][..], None);
    ///
    /// b.eval();
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    #[default_device]
    pub fn all_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_all_device(axes, keep_dims, stream).unwrap()
    }

    /// An `and` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = unsafe { a.all_axes_unchecked(&[0][..], None) };
    ///
    /// b.eval();
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn all_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        let axes = match axes.into() {
            Some(axes) => axes.to_vec(),
            None => {
                let axes: Vec<i32> = (0..self.ndim() as i32).collect();
                axes
            }
        };

        Array::from_ptr(mlx_sys::mlx_all_axes(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ptr(),
        ))
    }

    /// An `and` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.try_all_axes(&[0][..], None).unwrap();
    ///
    /// b.eval();
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    #[default_device]
    pub fn try_all_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Result<Array, OperationError> {
        let axes = match axes.into() {
            Some(axes) => axes.to_vec(),
            None => {
                let axes: Vec<i32> = (0..self.ndim() as i32).collect();
                axes
            }
        };

        let ndim = self.shape().len() as i32;
        let mut axes_set = std::collections::HashSet::new();
        for axis in axes.clone() {
            let ax = if axis < 0 { axis + ndim } else { axis };
            if ax < 0 || ax >= ndim {
                return Err(OperationError::AxisOutOfBounds(format!(
                    "Invalid axis {} for array with {} dimensions",
                    axis, ndim
                )));
            }

            axes_set.insert(ax);
        }

        if axes_set.len() != axes.len() {
            return Err(OperationError::WrongInput(format!(
                "Duplicate axes in {:?}",
                axes
            )));
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_all_axes(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ptr(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let mut all = array.all(&[0][..], None);

        all.eval();
        let results: &[bool] = all.as_slice();
        assert_eq!(results, &[false, true, true, true]);
    }

    #[test]
    fn test_all_axes_out_of_bounds() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[12]);
        let result = array.try_all(&[1][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_axes_duplicate_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let result = array.try_all(&[0, 0][..], None);
        assert!(result.is_err());
    }
}
