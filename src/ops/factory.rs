use crate::array::ArrayElement;
use crate::error::Exception;
use crate::prelude::ScalarOrArray;
use crate::Stream;
use crate::{array::Array, stream::StreamOrDevice};
use mlx_macros::default_device;
use num_traits::NumCast;

impl Array {
    /// Construct an array of zeros returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// Array::zeros_device::<f32>(&[5, 10], StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn zeros_device<T: ArrayElement>(
        shape: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_zeros(
                    shape.as_ptr(),
                    shape.len(),
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Construct an array of ones returning an error if shape is invalid.
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// Array::ones_device::<f32>(&[5, 10], StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn ones_device<T: ArrayElement>(
        shape: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_ones(
                    shape.as_ptr(),
                    shape.len(),
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Create an identity matrix or a general diagonal matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::eye_device::<f32>(10, None, None, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn eye_device<T: ArrayElement>(
        n: i32,
        m: impl Into<Option<i32>>,
        k: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_eye(
                    n,
                    m.into().unwrap_or(n),
                    k.into().unwrap_or(0),
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    /// Construct an array with the given value returning an error if shape is invalid.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::full_device::<f32>(&[5, 4], 7.0f32, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn full_device<'a, T: ArrayElement>(
        shape: &[i32],
        values: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_full(
                    shape.as_ptr(),
                    shape.len(),
                    values.into_owned_or_ref_array().as_ref().as_ptr(),
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Create a square identity matrix returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::identity_device::<f32>(10, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn identity_device<T: ArrayElement>(
        n: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_identity(n, T::DTYPE.into(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Generates ranges of numbers.
    ///
    /// Generate numbers in the half-open interval `[start, stop)` in increments of `step`.
    ///
    /// # Params
    ///
    /// - `start`: Starting value which defaults to `0`.
    /// - `stop`: Stopping value.
    /// - `step`: Increment which defaults to `1`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    ///
    /// // Create a 1-D array with values from 0 to 50
    /// let r = Array::arange::<f32, _>(None, 50, None);
    /// ```
    #[default_device]
    pub fn arange_device<T, U>(
        start: impl Into<Option<U>>,
        stop: U,
        step: impl Into<Option<U>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception>
    where
        T: ArrayElement,
        U: NumCast,
    {
        let start: f64 = start.into().and_then(NumCast::from).unwrap_or(0.0);
        let stop: f64 = NumCast::from(stop).unwrap();
        let step: f64 = step.into().and_then(NumCast::from).unwrap_or(1.0);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_arange(
                    start,
                    stop,
                    step,
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - start: start value
    /// - stop: stop value
    /// - count: number of samples -- defaults to 50 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = Array::linspace_device::<f32, _>(0, 50, None, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn linspace_device<T, U>(
        start: U,
        stop: U,
        count: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception>
    where
        T: ArrayElement,
        U: NumCast,
    {
        let count = count.into().unwrap_or(50);
        let start_f32 = NumCast::from(start).unwrap();
        let stop_f32 = NumCast::from(stop).unwrap();

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_linspace(
                    start_f32,
                    stop_f32,
                    count,
                    T::DTYPE.into(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Repeat an array along a specified axis returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_device::<i32>(source, 4, 1, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn repeat_device<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_repeat(
                    array.c_array,
                    count,
                    axis,
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Repeat a flattened array along axis 0 returning an error if params are invalid.
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_all_device::<i32>(source, 4, StreamOrDevice::default()).unwrap();
    /// ```
    #[default_device]
    pub fn repeat_all_device<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_repeat_all(
                    array.c_array,
                    count,
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal -- defaults to 0 if not specified
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, StreamOrDevice};
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = Array::tri_device::<f32>(5, None, None, StreamOrDevice::default());
    /// ```
    #[default_device]
    pub fn tri_device<T: ArrayElement>(
        n: i32,
        m: impl Into<Option<i32>>,
        k: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_tri(
                n,
                m.into().unwrap_or(n),
                k.into().unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ref().as_ptr(),
            ))
        }
    }
}

/// See [`Array::zeros`]
#[default_device]
pub fn zeros_device<T: ArrayElement>(
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::zeros_device::<T>(shape, stream)
}

/// See [`Array::ones`]
#[default_device]
pub fn ones_device<T: ArrayElement>(
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::ones_device::<T>(shape, stream)
}

/// See [`Array::eye`]
#[default_device]
pub fn eye_device<T: ArrayElement>(
    n: i32,
    m: Option<i32>,
    k: Option<i32>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::eye_device::<T>(n, m, k, stream)
}

/// See [`Array::full`]
#[default_device]
pub fn full_device<'a, T: ArrayElement>(
    shape: &[i32],
    values: impl ScalarOrArray<'a>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::full_device::<T>(shape, values, stream)
}

/// See [`Array::identity`]
#[default_device]
pub fn identity_device<T: ArrayElement>(
    n: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::identity_device::<T>(n, stream)
}

/// See [`Array::arange`]
#[default_device]
pub fn arange_device<T, U>(
    start: impl Into<Option<U>>,
    stop: U,
    step: impl Into<Option<U>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception>
where
    T: ArrayElement,
    U: NumCast,
{
    Array::arange_device::<T, U>(start, stop, step, stream)
}

/// See [`Array::linspace`]
#[default_device]
pub fn linspace_device<T, U>(
    start: U,
    stop: U,
    count: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception>
where
    T: ArrayElement,
    U: NumCast,
{
    Array::linspace_device::<T, U>(start, stop, count, stream)
}

/// See [`Array::repeat`]
#[default_device]
pub fn repeat_device<T: ArrayElement>(
    array: Array,
    count: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::repeat_device::<T>(array, count, axis, stream)
}

/// See [`Array::repeat_all`]
#[default_device]
pub fn repeat_all_device<T: ArrayElement>(
    array: Array,
    count: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    Array::repeat_all_device::<T>(array, count, stream)
}

/// See [`Array::tri`]
#[default_device]
pub fn tri_device<T: ArrayElement>(
    n: i32,
    m: Option<i32>,
    k: Option<i32>,
    stream: impl AsRef<Stream>,
) -> Array {
    Array::tri_device::<T>(n, m, k, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use half::f16;

    #[test]
    fn test_zeros() {
        let mut array = Array::zeros::<f32>(&[2, 3]).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_zeros_try() {
        let array = Array::zeros::<f32>(&[2, 3]);
        assert!(array.is_ok());

        let array = Array::zeros::<f32>(&[-1, 3]);
        assert!(array.is_err());
    }

    #[test]
    fn test_ones() {
        let mut array = Array::ones::<f16>(&[2, 3]).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float16);

        let data: &[f16] = array.as_slice();
        assert_eq!(data, &[f16::from_f32(1.0); 6]);
    }

    #[test]
    fn test_eye() {
        let mut array = Array::eye::<f32>(3, None, None).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let mut array = Array::full::<f32>(&[2, 3], 7f32).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[7.0; 6]);
    }

    #[test]
    fn test_full_array() {
        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::cpu()).unwrap();
        let mut array = Array::full::<f32>(&[2, 3], source).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        float_eq::float_eq!(*data, [0.0; 6], abs <= [1e-6; 6]);
    }

    #[test]
    fn test_full_try() {
        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::default()).unwrap();
        let array = Array::full::<f32>(&[2, 3], source);
        assert!(array.is_ok());

        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::default()).unwrap();
        let array = Array::full::<f32>(&[-1, 3], source);
        assert!(array.is_err());
    }

    #[test]
    fn test_identity() {
        let mut array = Array::identity::<f32>(3).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_arange() {
        let mut array = Array::arange::<f32, _>(None, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
        assert_eq!(data, expected.as_slice());

        let mut array = Array::arange::<i32, _>(0, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        let expected: Vec<i32> = (0..50).collect();
        assert_eq!(data, expected.as_slice());

        let result = Array::arange::<bool, _>(None, 50, None);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(f64::NEG_INFINITY, 50.0, None);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(0.0, f64::INFINITY, None);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(0.0, 50.0, f32::NAN);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(f32::NAN, 50.0, None);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(0.0, f32::NAN, None);
        assert!(result.is_err());

        let result = Array::arange::<f32, _>(0, i32::MAX as i64 + 1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace_int() {
        let mut array = Array::linspace::<f32, _>(0, 50, None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        assert_eq!(data, expected.as_slice());
    }

    #[test]
    fn test_linspace_float() {
        let mut array = Array::linspace::<f32, _>(0., 50., None).unwrap();
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        assert_eq!(data, expected.as_slice());
    }

    #[test]
    fn test_linspace_try() {
        let array = Array::linspace::<f32, _>(0, 50, None);
        assert!(array.is_ok());

        let array = Array::linspace::<f32, _>(0, 50, Some(-1));
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let mut array = Array::repeat::<i32>(source, 4, 1).unwrap();
        assert_eq!(array.shape(), &[2, 8]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, 4, 1);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat::<i32>(source, -1, 1);
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat_all() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let mut array = Array::repeat_all::<i32>(source, 4).unwrap();
        assert_eq!(array.shape(), &[16]);
        assert_eq!(array.dtype(), Dtype::Int32);

        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_all_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_all::<i32>(source, 4);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::repeat_all::<i32>(source, -1);
        assert!(array.is_err());
    }

    #[test]
    fn test_tri() {
        let mut array = Array::tri::<f32>(3, None, None);
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }
}
