use crate::array::ArrayElement;
use crate::{array::Array, stream::StreamOrDevice};

impl Array {
    /// Construct an array of zeros.
    ///
    /// Example:
    ///
    /// ```rust
    /// use mlx::{array::Array, stream::StreamOrDevice};
    /// Array::zeros::<f32>(&[5, 10], StreamOrDevice::default());
    /// ```
    ///
    /// # Parameters:
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    pub fn zeros<T: ArrayElement>(shape: &[usize], stream: StreamOrDevice) -> Array {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_zeros(
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                ctx,
            ))
        }
    }

    /// Construct an array of ones.
    ///
    /// Example:
    ///
    /// ```rust
    /// use mlx::{array::Array, stream::StreamOrDevice};
    /// Array::ones::<f32>(&[5, 10], StreamOrDevice::default());
    /// ```
    ///
    /// # Parameters:
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    pub fn ones<T: ArrayElement>(shape: &[usize], stream: StreamOrDevice) -> Array {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_ones(
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                ctx,
            ))
        }
    }

    /// Create an identity matrix or a general diagonal matrix.
    ///
    /// Example:
    ///
    /// ```rust
    /// use mlx::{array::Array, stream::StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::eye::<f32>(10, None, None, StreamOrDevice::default());
    /// ```
    ///
    /// # Parameters:
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    /// - stream: stream or device to evaluate on
    pub fn eye<T: ArrayElement>(n: i32, m: Option<i32>, k: Option<i32>, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_eye(
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                ctx,
            ))
        }
    }

    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```rust
    /// use mlx::{array::Array, stream::StreamOrDevice};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::full::<f32>(&[5, 4], 7f32.into(), StreamOrDevice::default());
    /// ```
    ///
    /// # Parameters:
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    /// - stream: stream or device to evaluate on
    pub fn full<T: ArrayElement>(shape: &[usize], values: Array, stream: StreamOrDevice) -> Array {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_full(
                shape.as_ptr(),
                shape.len(),
                values.c_array,
                T::DTYPE.into(),
                ctx,
            ))
        }
    }

    /// Create a square identity matrix.
    ///
    /// Example:
    ///
    /// ```rust
    /// use mlx::{array::Array, stream::StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::identity::<f32>(10, StreamOrDevice::default());
    /// ```
    ///
    /// # Parameters:
    /// - n: number of rows and columns in the output
    /// - stream: stream or device to evaluate on
    pub fn identity<T: ArrayElement>(n: i32, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_identity(n, T::DTYPE.into(), ctx))
        }
    }

    /// Element-wise absolute value.
    ///
    /// # Parameters:
    /// - stream: stream or device to evaluate on
    pub fn abs(&self, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_abs(self.c_array, ctx))
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use super::*;
    use crate::dtype::Dtype;

    #[test]
    fn test_zeros() {
        let mut array = Array::zeros::<f32>(&[2, 3], StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_ones() {
        let mut array = Array::ones::<f16>(&[2, 3], StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float16);

        array.eval();
        let data: &[f16] = array.as_slice().unwrap();
        assert_eq!(data, &[f16::from_f32(1.0); 6]);
    }

    #[test]
    fn test_eye() {
        let mut array = Array::eye::<f32>(3, None, None, StreamOrDevice::default());
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let mut array = Array::full::<f32>(&[2, 3], 7f32.into(), StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[7.0; 6]);
    }

    #[test]
    fn test_full_array() {
        let source = Array::zeros::<f32>(&[1, 3], StreamOrDevice::default());
        let mut array = Array::full::<f32>(&[2, 3], source, StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_identity() {
        let mut array = Array::identity::<f32>(3, StreamOrDevice::default());
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_abs() {
        let data = [1i32, 2, -3, -4, -5];
        let array = Array::from_slice(&data, &[5]);
        let mut result = array.abs(StreamOrDevice::default());

        result.eval();
        let data: &[i32] = result.as_slice().unwrap();
        assert_eq!(data, [1, 2, 3, 4, 5]);
    }
}
