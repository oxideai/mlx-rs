use crate::array::ArrayElement;
use crate::error::DataStoreError;
use crate::{array::Array, stream::StreamOrDevice};
use mlx_macros::default_device;
use num_traits::NumCast;

impl Array {
    /// Construct an array of zeros.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// Array::zeros_device::<f32>(&[5, 10], StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    #[default_device]
    pub fn zeros_device<T: ArrayElement>(shape: &[i32], stream: StreamOrDevice) -> Array {
        // TODO: Can we make use of full() here?
        Self::try_zeros_device::<T>(shape, stream).unwrap()
    }

    /// Construct an array of zeros without validating shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// unsafe { Array::zeros_device_unchecked::<f32>(&[5, 10], StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that the shape has no negative dimensions.
    #[default_device]
    pub unsafe fn zeros_device_unchecked<T: ArrayElement>(
        shape: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        // TODO: Can we make use of full() here?
        unsafe {
            Array::from_ptr(mlx_sys::mlx_zeros(
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                stream.as_ptr(),
            ))
        }
    }

    /// Construct an array of zeros returning an error if shape is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// Array::try_zeros_device::<f32>(&[5, 10], StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    #[default_device]
    pub fn try_zeros_device<T: ArrayElement>(
        shape: &[i32],
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        // TODO: Can we make use of full() here?
        if shape.iter().any(|&i| i < 0) {
            return Err(DataStoreError::NegativeDimensions(
                "negative dimensions in shape not allowed".to_string(),
            ));
        }

        Ok(unsafe { Self::zeros_device_unchecked::<T>(shape, stream) })
    }

    /// Construct an array of ones.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// Array::ones_device::<f32>(&[5, 10], StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    #[default_device]
    pub fn ones_device<T: ArrayElement>(shape: &[i32], stream: StreamOrDevice) -> Array {
        // TODO: Can we make use of full() here?
        Array::try_ones_device::<T>(shape, stream).unwrap()
    }

    /// Construct an array of ones without validating shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// unsafe { Array::ones_device_unchecked::<f32>(&[5, 10], StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that the shape has no negative dimensions.
    #[default_device]
    pub unsafe fn ones_device_unchecked<T: ArrayElement>(
        shape: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        // TODO: Can we make use of full() here?
        Array::from_ptr(mlx_sys::mlx_ones(
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            stream.as_ptr(),
        ))
    }

    /// Construct an array of ones returning an error if shape is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// Array::try_ones_device::<f32>(&[5, 10], StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - shape: Desired shape
    /// - stream: Stream or device to evaluate on
    pub fn try_ones_device<T: ArrayElement>(
        shape: &[i32],
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        // TODO: Can we make use of full() here?
        if shape.iter().any(|&i| i < 0) {
            return Err(DataStoreError::NegativeDimensions(
                "negative dimensions in shape not allowed".to_string(),
            ));
        }

        Ok(unsafe { Self::ones_device_unchecked::<T>(shape, stream) })
    }

    /// Create an identity matrix or a general diagonal matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::eye_device::<f32>(10, None, None, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn eye_device<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: StreamOrDevice,
    ) -> Array {
        Self::try_eye_device::<T>(n, m, k, stream).unwrap()
    }

    /// Create an identity matrix or a general diagonal matrix without validating params.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = unsafe { Array::eye_device_unchecked::<f32>(10, None, None, StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that n and m are positive.
    #[default_device]
    pub unsafe fn eye_device_unchecked<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_eye(
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ptr(),
            ))
        }
    }

    /// Create an identity matrix or a general diagonal matrix returning an error if params are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::try_eye_device::<f32>(10, None, None, StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal - defaults to 0 if not specified
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_eye_device<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if n < 0 || m.unwrap_or(n) < 0 {
            return Err(DataStoreError::NegativeInteger(format!(
                "m and n must be positive, got m: {}, n: {}",
                m.unwrap_or(n),
                n
            )));
        }

        Ok(unsafe { Self::eye_device_unchecked::<T>(n, m, k, stream) })
    }

    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::full_device::<f32>(&[5, 4], 7f32.into(), StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn full_device<T: ArrayElement>(
        shape: &[i32],
        values: Array,
        stream: StreamOrDevice,
    ) -> Array {
        Self::try_full_device::<T>(shape, values, stream).unwrap()
    }

    /// Construct an array with the given value without validating shape.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [5, 4] array filled with 7
    /// let r = unsafe { Array::full_device_unchecked::<f32>(&[5, 4], 7f32.into(), StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that the shape has no negative dimensions.
    #[default_device]
    pub unsafe fn full_device_unchecked<T: ArrayElement>(
        shape: &[i32],
        values: Array,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_full(
                shape.as_ptr(),
                shape.len(),
                values.c_array,
                T::DTYPE.into(),
                stream.as_ptr(),
            ))
        }
    }

    /// Construct an array with the given value returning an error if shape is invalid.
    ///
    /// Constructs an array of size `shape` filled with `values`. If `values`
    /// is an [Array] it must be [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting) to the given `shape`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [5, 4] array filled with 7
    /// let r = Array::try_full_device::<f32>(&[5, 4], 7f32.into(), StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - shape: shape of the output array
    /// - values: values to be broadcast into the array
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_full_device<T: ArrayElement>(
        shape: &[i32],
        values: Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if shape.iter().any(|&i| i < 0) {
            return Err(DataStoreError::NegativeDimensions(
                "negative dimensions in shape not allowed".to_string(),
            ));
        }

        Ok(unsafe { Self::full_device_unchecked::<T>(shape, values, stream) })
    }

    /// Create a square identity matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::identity_device::<f32>(10, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn identity_device<T: ArrayElement>(n: i32, stream: StreamOrDevice) -> Array {
        Self::eye_device::<T>(n, Some(n), None, stream)
    }

    /// Create a square identity matrix without validating params.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = unsafe { Array::identity_device_unchecked::<f32>(10, StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that n is positive.
    pub unsafe fn identity_device_unchecked<T: ArrayElement>(
        n: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe { Self::eye_device_unchecked::<T>(n, Some(n), None, stream) }
    }

    /// Create a square identity matrix returning an error if params are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = Array::try_identity_device::<f32>(10, StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows and columns in the output
    /// - stream: stream or device to evaluate on
    pub fn try_identity_device<T: ArrayElement>(
        n: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        Self::try_eye_device::<T>(n, Some(n), None, stream)
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = Array::linspace_device::<f32, _>(0, 50, None, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - start: start value
    /// - stop: stop value
    /// - count: number of samples -- defaults to 50 if not specified
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn linspace_device<T, U>(
        start: U,
        stop: U,
        count: Option<i32>,
        stream: StreamOrDevice,
    ) -> Array
    where
        T: ArrayElement,
        U: NumCast,
    {
        Self::try_linspace_device::<T, U>(start, stop, count, stream).unwrap()
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` without validating params.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = unsafe { Array::linspace_device_unchecked::<f32, _>(0, 50, None, StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - start: start value
    /// - stop: stop value
    /// - count: number of samples -- defaults to 50 if not specified
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that count is positive.
    #[default_device]
    pub unsafe fn linspace_device_unchecked<T, U>(
        start: U,
        stop: U,
        count: Option<i32>,
        stream: StreamOrDevice,
    ) -> Array
    where
        T: ArrayElement,
        U: NumCast,
    {
        let start_f32 = NumCast::from(start).unwrap();
        let stop_f32 = NumCast::from(stop).unwrap();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_linspace(
                start_f32,
                stop_f32,
                count.unwrap_or(50),
                T::DTYPE.into(),
                stream.as_ptr(),
            ))
        }
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` returning an error if params are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = Array::try_linspace_device::<f32, _>(0, 50, None, StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - start: start value
    /// - stop: stop value
    /// - count: number of samples -- defaults to 50 if not specified
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_linspace_device<T, U>(
        start: U,
        stop: U,
        count: Option<i32>,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError>
    where
        T: ArrayElement,
        U: NumCast,
    {
        let count = count.unwrap_or(50);
        if count < 0 {
            return Err(DataStoreError::NegativeInteger(format!(
                "count must be positive, got {}",
                count
            )));
        }

        Ok(unsafe { Self::linspace_device_unchecked::<T, U>(start, stop, Some(count), stream) })
    }

    /// Repeat an array along a specified axis.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_device::<i32>(source, 4, 1, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn repeat_device<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Array {
        Self::try_repeat_device::<T>(array, count, axis, stream).unwrap()
    }

    /// Repeat an array along a specified axis without validating params.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = unsafe { Array::repeat_device_unchecked::<i32>(source, 4, 1, StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that count is positive.
    #[default_device]
    pub unsafe fn repeat_device_unchecked<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_repeat(
                array.c_array,
                count,
                axis,
                stream.as_ptr(),
            ))
        }
    }

    /// Repeat an array along a specified axis returning an error if params are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::try_repeat_device::<i32>(source, 4, 1, StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - axis: axis to repeat along
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_repeat_device<T: ArrayElement>(
        array: Array,
        count: i32,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if count < 0 {
            return Err(DataStoreError::NegativeInteger(format!(
                "count must be positive, got {}",
                count
            )));
        }

        Ok(unsafe { Self::repeat_device_unchecked::<T>(array, count, axis, stream) })
    }

    /// Repeat a flattened array along axis 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::repeat_all_device::<i32>(source, 4, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn repeat_all_device<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: StreamOrDevice,
    ) -> Array {
        Self::try_repeat_all_device::<T>(array, count, stream).unwrap()
    }

    /// Repeat a flattened array along axis 0 without validating params.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = unsafe { Array::repeat_all_device_unchecked::<i32>(source, 4, StreamOrDevice::default()) };
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// The caller must ensure that count is positive.
    #[default_device]
    pub unsafe fn repeat_all_device_unchecked<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_repeat_all(
                array.c_array,
                count,
                stream.as_ptr(),
            ))
        }
    }

    /// Repeat a flattened array along axis 0 returning an error if params are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // repeat a 4 element array 4 times along axis 0
    /// let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
    /// let r = Array::try_repeat_all_device::<i32>(source, 4, StreamOrDevice::default()).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - array: array to repeat
    /// - count: number of times to repeat
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_repeat_all_device<T: ArrayElement>(
        array: Array,
        count: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if count < 0 {
            return Err(DataStoreError::NegativeInteger(format!(
                "count must be positive, got {}",
                count
            )));
        }

        Ok(unsafe { Self::repeat_all_device_unchecked::<T>(array, count, stream) })
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::{Array, StreamOrDevice};
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = Array::tri_device::<f32>(5, None, None, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - n: number of rows in the output
    /// - m: number of columns in the output -- equal to `n` if not specified
    /// - k: index of the diagonal -- defaults to 0 if not specified
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn tri_device<T: ArrayElement>(
        n: i32,
        m: Option<i32>,
        k: Option<i32>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_tri(
                n,
                m.unwrap_or(n),
                k.unwrap_or(0),
                T::DTYPE.into(),
                stream.as_ptr(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use half::f16;

    #[test]
    fn test_zeros() {
        let mut array = Array::zeros::<f32>(&[2, 3]);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_zeros_try() {
        let array = Array::try_zeros::<f32>(&[2, 3]);
        assert!(array.is_ok());

        let array = Array::try_zeros::<f32>(&[-1, 3]);
        assert!(array.is_err());
    }

    #[test]
    fn test_ones() {
        let mut array = Array::ones::<f16>(&[2, 3]);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float16);

        array.eval();
        let data: &[f16] = array.as_slice();
        assert_eq!(data, &[f16::from_f32(1.0); 6]);
    }

    #[test]
    fn test_eye() {
        let mut array = Array::eye::<f32>(3, None, None);
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_full_scalar() {
        let mut array = Array::full::<f32>(&[2, 3], 7f32.into());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[7.0; 6]);
    }

    #[test]
    fn test_full_array() {
        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::default());
        let mut array = Array::full::<f32>(&[2, 3], source);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[0.0; 6]);
    }

    #[test]
    fn test_full_try() {
        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::default());
        let array = Array::try_full::<f32>(&[2, 3], source);
        assert!(array.is_ok());

        let source = Array::zeros_device::<f32>(&[1, 3], StreamOrDevice::default());
        let array = Array::try_full::<f32>(&[-1, 3], source);
        assert!(array.is_err());
    }

    #[test]
    fn test_identity() {
        let mut array = Array::identity::<f32>(3);
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_linspace_int() {
        let mut array = Array::linspace::<f32, _>(0, 50, None);
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        assert_eq!(data, expected.as_slice());
    }

    #[test]
    fn test_linspace_float() {
        let mut array = Array::linspace::<f32, _>(0., 50., None);
        assert_eq!(array.shape(), &[50]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        let expected: Vec<f32> = (0..50).map(|x| x as f32 * (50.0 / 49.0)).collect();
        assert_eq!(data, expected.as_slice());
    }

    #[test]
    fn test_linspace_try() {
        let array = Array::try_linspace::<f32, _>(0, 50, None);
        assert!(array.is_ok());

        let array = Array::try_linspace::<f32, _>(0, 50, Some(-1));
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let mut array = Array::repeat::<i32>(source, 4, 1);
        assert_eq!(array.shape(), &[2, 8]);
        assert_eq!(array.dtype(), Dtype::Int32);

        array.eval();
        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::try_repeat::<i32>(source, 4, 1);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::try_repeat::<i32>(source, -1, 1);
        assert!(array.is_err());
    }

    #[test]
    fn test_repeat_all() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let mut array = Array::repeat_all::<i32>(source, 4);
        assert_eq!(array.shape(), &[16]);
        assert_eq!(array.dtype(), Dtype::Int32);

        array.eval();
        let data: &[i32] = array.as_slice();
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_all_try() {
        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::try_repeat_all::<i32>(source, 4);
        assert!(array.is_ok());

        let source = Array::from_slice(&[0, 1, 2, 3], &[2, 2]);
        let array = Array::try_repeat_all::<i32>(source, -1);
        assert!(array.is_err());
    }

    #[test]
    fn test_tri() {
        let mut array = Array::tri::<f32>(3, None, None);
        assert_eq!(array.shape(), &[3, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }
}
