use mlx_macros::default_device;

use crate::{array::Array, error::FftError, stream::StreamOrDevice, utils::{resolve_index, resolve_index_unchecked}};

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - a: The input array.
/// - n: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - axis: Axis along which to perform the FFT. The default is -1.
///
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
/// let s = StreamOrDevice::cpu();
/// let mut result = unsafe { fft_device_unchecked(&array, 4, 0, s) };
/// result.eval();
///
/// assert_eq!(result.dtype(), Dtype::Complex64);
///
/// let expected = &[
///     complex64::new(10.0, 0.0),
///     complex64::new(-2.0, 2.0),
///     complex64::new(-2.0, 0.0),
///     complex64::new(-2.0, -2.0),
/// ];
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
///
/// // test that previous array is not modified and valid
/// let data: &[f32] = array.as_slice();
/// assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub unsafe fn fft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let axis = axis.into().unwrap_or(-1);
    let n = n.into().unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

/// One dimensional discrete Fourier Transform.
///
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
///
/// // Error case
/// let scalar_array = Array::from_float(1.0);
/// let result = try_fft_device(&scalar_array, 0, 0, StreamOrDevice::cpu());
/// assert!(result.is_err());
///
/// let result = try_fft_device(&array, 4, 2, StreamOrDevice::cpu());
/// assert!(result.is_err());
///
/// // Success case
/// let mut result = try_fft_device(&array, 4, 0, StreamOrDevice::cpu()).unwrap();
/// result.eval();
///
/// assert_eq!(result.dtype(), Dtype::Complex64);
///
/// let expected = &[
///     complex64::new(10.0, 0.0),
///     complex64::new(-2.0, 2.0),
///     complex64::new(-2.0, 0.0),
///     complex64::new(-2.0, -2.0),
/// ];
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
///
/// // test that previous array is not modified and valid
/// let data: &[f32] = array.as_slice();
/// assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn try_fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    if a.ndim() < 1 {
        return Err(FftError::ScalarArray);
    }

    let axis = axis.into().unwrap_or(-1);
    let axis_index = resolve_index(axis, a.ndim());
    let n = n.into().unwrap_or(a.shape()[axis_index]);

    Ok(unsafe { fft_device_unchecked(a, Some(n), Some(axis), stream) })
}

/// One dimensional discrete Fourier Transform.
/// 
/// # Panic
/// 
/// Panics if the input array is a scalar or if the axis is invalid.
/// 
/// See [`try_fft_device`] for more details.
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_fft_device(a, n, axis, stream).unwrap()
}

#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn fft2_device_unchecked(
    a: &Array,
    n: &[i32],
    axes: &[i32],
    stream: StreamOrDevice,
) -> Array {
    let num_n = n.len();
    let num_axes = axes.len();

    let n_ptr = n.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = mlx_sys::mlx_fft_fft2(
            a.c_array,
            n_ptr,
            num_n,
            axes_ptr,
            num_axes,
            stream.as_ptr()
        );

        Array::from_ptr(c_array)
    }
}

// TODO: test out of bound indexing
#[cfg(test)]
mod tests {
    #[test]
    fn test_fft_unchecked() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let mut result = unsafe { fft_unchecked(&array, 4, 0) };
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // The original array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fft_device_unchecked() {
        use crate::{Array, StreamOrDevice, complex64, fft::*};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let s = StreamOrDevice::cpu();
        let mut result = unsafe { fft_device_unchecked(&array, 4, 0, s) };
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_try_fft() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fft(&scalar_array, 0, 0);
        assert!(result.is_err());

        let result = try_fft(&array, 4, 2);
        assert!(result.is_err());

        // Success case
        let mut result = try_fft(&array, 4, 0).unwrap();
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_try_fft_device() {
        use crate::{Array, StreamOrDevice, complex64, fft::*};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fft_device(&scalar_array, 0, 0, StreamOrDevice::cpu());
        assert!(result.is_err());

        let result = try_fft_device(&array, 4, 2, StreamOrDevice::cpu());
        assert!(result.is_err());

        // Success case
        let mut result = try_fft_device(&array, 4, 0, StreamOrDevice::cpu()).unwrap();
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fft() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        // Success case
        let mut result = fft(&array, 4, 0);
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fft_device() {
        use crate::{Array, Dtype, StreamOrDevice, complex64, fft::*};

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        // Success case
        let mut result = fft_device(&array, 4, 0, StreamOrDevice::cpu());
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }
}
