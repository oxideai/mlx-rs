use mlx_macros::default_device;

use crate::{array::Array, error::FftError, stream::StreamOrDevice};

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
        if axis.is_negative() {
            // TODO: replace with unchecked_add when it's stable
            let index = (a.ndim() as i32)
                .checked_add(axis)
                .unwrap()
                // index may still be negative
                .max(0) as usize;
            a.shape()[index]
        } else {
            // # Safety: positive i32 is always smaller than usize::MAX
            a.shape()[axis as usize]
        }
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
    let (n, axis) = if axis.is_negative() {
        if axis.abs() as usize > a.ndim() {
            return Err(FftError::InvalidAxis(a.ndim()));
        }
        let index = a.ndim() - axis.abs() as usize;
        let n = n.into().unwrap_or(a.shape()[index]);
        (n, axis)
    } else {
        if axis as usize >= a.ndim() {
            return Err(FftError::InvalidAxis(a.ndim()));
        }
        let n = n.into().unwrap_or(a.shape()[axis as usize]);
        (n, axis)
    };

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
