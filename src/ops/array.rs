use crate::array::Array;
use crate::stream::StreamOrDevice;
use mlx_macros::default_device;

impl Array {
    /// Element-wise absolute value.
    ///
    /// # Params
    ///
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn abs_device(&self, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe { Array::from_ptr(mlx_sys::mlx_abs(self.c_array, ctx)) }
    }

    /// Element-wise addition.
    ///
    /// Add two arrays with <doc:broadcasting>.
    pub fn add_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_add(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
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
        let data: &[i32] = result.as_slice().unwrap();
        assert_eq!(data, [1, 2, 3, 4, 5]);

        // test that previous array is not modified and valid
        let data: &[i32] = array.as_slice().unwrap();
        assert_eq!(data, [1, 2, -3, -4, -5]);
    }

    #[test]
    fn test_add_device() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = add(&a, &b);
        c.eval();

        let c_data: &[f32] = c.as_slice().unwrap();
        assert_eq!(c_data, &[5.0, 7.0, 9.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice().unwrap();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice().unwrap();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }
}
