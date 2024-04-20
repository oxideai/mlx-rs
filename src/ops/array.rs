use crate::array::Array;
use crate::stream::StreamOrDevice;
use mlx_macros::default_device;

impl Array {
    /// Element-wise absolute value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
    /// let mut result = array.abs();
    ///
    /// result.eval();
    /// let data: &[i32] = result.as_slice();
    /// // data == [1, 2, 3, 4, 5]
    /// ```
    ///
    /// # Params
    ///
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn abs_device(&self, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe { Array::from_ptr(mlx_sys::mlx_abs(self.c_array, ctx)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs() {
        let data = [1i32, 2, -3, -4, -5];
        let array = Array::from_slice(&data, &[5]);
        let mut result = array.abs();

        result.eval();
        let data: &[i32] = result.as_slice();
        assert_eq!(data, [1, 2, 3, 4, 5]);

        // test that previous array is not modified and valid
        let data: &[i32] = array.as_slice();
        assert_eq!(data, [1, 2, -3, -4, -5]);
    }
}
