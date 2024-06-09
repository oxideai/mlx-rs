use crate::error::Exception;
use crate::{Array, Stream, StreamOrDevice};
use mlx_macros::default_device;

impl Array {
    /// Return the cumulative maximum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cummax_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                match axis.into() {
                    Some(axis) => {
                        mlx_sys::mlx_cummax(
                            self.c_array,
                            axis,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                    None => {
                        let shape = &[-1];
                        let flat = self.reshape_device(shape, &stream)?;
                        mlx_sys::mlx_cummax(
                            flat.c_array,
                            0,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                }
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    /// Return the cumulative minimum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative minimum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative minimum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
    /// let result = array.cummin(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cummin_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                match axis.into() {
                    Some(axis) => {
                        mlx_sys::mlx_cummin(
                            self.c_array,
                            axis,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                    None => {
                        let shape = &[-1];
                        let flat = self.reshape_device(shape, &stream)?;
                        mlx_sys::mlx_cummin(
                            flat.c_array,
                            0,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                }
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    /// Return the cumulative product of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative product over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative product is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [20, 72]] -- cumulative min along the columns
    /// let result = array.cumprod(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cumprod_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                match axis.into() {
                    Some(axis) => {
                        mlx_sys::mlx_cumprod(
                            self.c_array,
                            axis,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                    None => {
                        let shape = &[-1];
                        let flat = self.reshape_device(shape, &stream)?;
                        mlx_sys::mlx_cumprod(
                            flat.c_array,
                            0,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                }
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    /// Return the cumulative sum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    /// - axis: Optional axis to compute the cumulative sum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative sum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [9, 17]] -- cumulative min along the columns
    /// let result = array.cumsum(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cumsum_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                match axis.into() {
                    Some(axis) => {
                        mlx_sys::mlx_cumsum(
                            self.c_array,
                            axis,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                    None => {
                        let shape = &[-1];
                        let flat = self.reshape_device(shape, &stream)?;
                        mlx_sys::mlx_cumsum(
                            flat.c_array,
                            0,
                            reverse.into().unwrap_or(false),
                            inclusive.into().unwrap_or(true),
                            stream.as_ref().as_ptr(),
                        )
                    }
                }
            };

            Ok(Array::from_ptr(c_array))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_cummax() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let mut result = array.cummax(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);

        let mut result = array.cummax(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 9]);

        let mut result = array.cummax(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 8, 9]);

        let mut result = array.cummax(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 9, 4, 9]);

        let mut result = array.cummax(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);
    }

    #[test]
    fn test_cummax_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummax(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cummin() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let mut result = array.cummin(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 8]);

        let mut result = array.cummin(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 5, 4, 4]);

        let mut result = array.cummin(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 5, 4, 4]);

        let mut result = array.cummin(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[4, 8, 4, 9]);

        let mut result = array.cummin(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 8]);
    }

    #[test]
    fn test_cummin_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummin(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumprod() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let mut result = array.cumprod(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 20, 72]);

        let mut result = array.cumprod(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 40, 4, 36]);

        let mut result = array.cumprod(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 40, 160, 1440]);

        let mut result = array.cumprod(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[20, 72, 4, 9]);

        let mut result = array.cumprod(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 20, 72]);
    }

    #[test]
    fn test_cumprod_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumprod(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumsum() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let mut result = array.cumsum(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 9, 17]);

        let mut result = array.cumsum(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 13, 4, 13]);

        let mut result = array.cumsum(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 13, 17, 26]);

        let mut result = array.cumsum(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[9, 17, 4, 9]);

        let mut result = array.cumsum(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 9, 17]);
    }

    #[test]
    fn test_cumsum_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumsum(2, None, None);
        assert!(result.is_err());
    }
}
