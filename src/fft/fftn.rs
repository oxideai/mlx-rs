use mlx_macros::default_device;

use crate::{array::Array, error::FftError, stream::StreamOrDevice};

use super::resolve_size_and_axis_unchecked;

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
#[default_device(device = "cpu")]
pub unsafe fn fft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n, axis);
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
#[default_device(device = "cpu")]
pub fn try_fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (n, axis) = super::try_resolve_size_and_axis(a, n, axis)?;
    unsafe { Ok(fft_device_unchecked(a, Some(n), Some(axis), stream)) }
}

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
///
/// # Panic
///
/// Panics if the input array is a scalar or if the axis is invalid.
///
/// See [`try_fft_device`] for more details.
#[default_device(device = "cpu")]
pub fn fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_fft_device(a, n, axis, stream).unwrap()
}

fn fft2_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_fft2(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Two dimensional discrete Fourier Transform.
///
/// # Param
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
///  with zeros to match `n`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub unsafe fn fft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    let (valid_s, valid_axes) = super::resolve_sizes_and_axes_unchecked(a, s, axes);
    fft2_device_inner(a, &valid_s, &valid_axes, stream)
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `n`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn try_fft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let valid_axes = axes.into().unwrap_or(&[-2, -1]);
    let (valid_s, valid_axes) = super::try_resolve_sizes_and_axes(a, s, valid_axes)?;
    Ok(fft2_device_inner(a, &valid_s, &valid_axes, stream))
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `n`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [`try_fft2_device`] for more details.
#[default_device(device = "cpu")]
pub fn fft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_fft2_device(a, s, axes, stream).unwrap()
}

#[inline]
fn fftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_fftn(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());

        Array::from_ptr(c_array)
    }
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///  padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
///  if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///   over the last `len(s)` axes are or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub unsafe fn fftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let (valid_s, valid_axes) = super::resolve_sizes_and_axes_unchecked(a, s, axes);
    fftn_device_inner(a, &valid_s, &valid_axes, stream)
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub fn try_fftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = super::try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(fftn_device_inner(a, &valid_s, &valid_axes, stream))
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
///
/// # Panic
///
/// - if the input array is a scalar array
/// - if the axes are not unique
/// - if the shape and axes have different sizes
/// - if the output sizes are invalid (<= 0)
/// - if more axes are provided than the array has
///
/// See [`try_fftn_device`] for more details.
#[default_device(device = "cpu")]
pub fn fftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_fftn_device(a, s, axes, stream).unwrap()
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
    fn test_fft2_unchecked() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let n = [2, 2];
        let axes = [-2, -1];
        let mut result = unsafe { fft2_unchecked(&array, &n[..], &axes[..]) };
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_try_fft2() {
        use crate::{complex64, error::FftError, fft::*, Array};

        let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fft2(&scalar_array, None, None);
        assert_eq!(result.unwrap_err(), FftError::ScalarArray);

        let result = try_fft2(&array, &[2, 2, 2][..], &[0, 1, 2][..]);
        assert_eq!(result.unwrap_err(), FftError::InvalidAxis { ndim: 2 });

        let result = try_fft2(&array, &[2, 2][..], &[-1][..]);
        assert_eq!(
            result.unwrap_err(),
            FftError::IncompatibleShapeAndAxes {
                shape_size: 2,
                axes_size: 1,
            }
        );

        let result = try_fft2(&array, None, &[-2, -2][..]);
        assert_eq!(result.unwrap_err(), FftError::DuplicateAxis { axis: -2 });

        let result = try_fft2(&array, &[-2, 2][..], None);
        assert_eq!(result.unwrap_err(), FftError::InvalidOutputSize);

        // Success case
        let mut result = try_fft2(&array, None, None).unwrap();
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let expected = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fft2() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let n = [2, 2];
        let axes = [-2, -1];
        let mut result = fft2(&array, Some(&n[..]), Some(&axes[..]));
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fftn_unchecked() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::ones::<f32>(&[3, 3]);
        let mut result = unsafe { fftn_unchecked(&array, None, None) };
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let expected = &[
            complex64::new(9.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];
        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0; 9]);
    }

    #[test]
    fn test_try_fftn() {
        use crate::{complex64, error::FftError, fft::*, Array};

        let array = Array::ones::<f32>(&[3, 3, 3]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fftn(&scalar_array, None, None);
        assert_eq!(result.unwrap_err(), FftError::ScalarArray);

        let result = try_fftn(&array, &[3, 3, 3, 3][..], &[0, 1, 2, 3][..]);
        assert_eq!(result.unwrap_err(), FftError::InvalidAxis { ndim: 3 });

        let result = try_fftn(&array, &[3, 3, 3][..], &[-1][..]);
        assert_eq!(
            result.unwrap_err(),
            FftError::IncompatibleShapeAndAxes {
                shape_size: 3,
                axes_size: 1,
            }
        );

        let result = try_fftn(&array, None, &[-2, -2][..]);
        assert_eq!(result.unwrap_err(), FftError::DuplicateAxis { axis: -2 });

        let result = try_fftn(&array, &[-2, 2][..], None);
        assert_eq!(result.unwrap_err(), FftError::InvalidOutputSize);

        // Success case
        let mut result = try_fftn(&array, None, None).unwrap();
        result.eval();

        assert_eq!(result.dtype(), crate::dtype::Dtype::Complex64);

        let mut expected = vec![complex64::new(0.0, 0.0); 27];
        expected[0] = complex64::new(27.0, 0.0);

        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0; 27]);
    }

    #[test]
    fn test_fftn() {
        use crate::{complex64, fft::*, Array, Dtype};

        let array = Array::ones::<f32>(&[3, 3, 3]);
        let mut result = fftn(&array, None, None);
        result.eval();

        assert_eq!(result.dtype(), Dtype::Complex64);

        let mut expected = vec![complex64::new(0.0, 0.0); 27];
        expected[0] = complex64::new(27.0, 0.0);

        assert_eq!(result.as_slice::<complex64>(), &expected[..]);

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, &[1.0; 27]);
    }
}
