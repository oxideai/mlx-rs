use mlx_macros::default_device;

use crate::{array::Array, error::FftError, stream::StreamOrDevice, utils::IntoOption, Stream};

use super::utils::{
    resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked, try_resolve_size_and_axis,
    try_resolve_sizes_and_axes,
};

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_fft_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn fft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    unsafe {
        let c_array = mlx_sys::mlx_fft_fft(a.c_array, n, axis, stream.as_ref().as_ptr());
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
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let (n, axis) = try_resolve_size_and_axis(a, n.into(), axis.into())?;
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
    stream: impl AsRef<Stream>,
) -> Array {
    try_fft_device(a, n, axis, stream).unwrap()
}

/// Two dimensional discrete Fourier Transform.
///
/// # Param
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
///  with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_fft2_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn fft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    fftn_device_unchecked(a, s, axes, stream)
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn try_fft2_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    try_fftn_device(a, s, axes, stream)
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [`try_fft2_device`] for more details.
#[default_device(device = "cpu")]
pub fn fft2_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_fft2_device(a, s, axes, stream).unwrap()
}

#[inline]
fn fftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: impl AsRef<Stream>) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = mlx_sys::mlx_fft_fftn(
            a.c_array,
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            stream.as_ref().as_ptr(),
        );

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
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_fftn_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn fftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    let (valid_s, valid_axes) =
        resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
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
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s.into_option(), axes.into_option())?;
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
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_fftn_device(a, s, axes, stream).unwrap()
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_ifft_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn ifft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    unsafe {
        let c_array = mlx_sys::mlx_fft_ifft(a.c_array, n, axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[default_device(device = "cpu")]
pub fn try_ifft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let (n, axis) = try_resolve_size_and_axis(a, n.into(), axis.into())?;
    unsafe { Ok(ifft_device_unchecked(a, n, axis, stream)) }
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[default_device(device = "cpu")]
pub fn ifft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_ifft_device(a, n, axis, stream).unwrap()
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_ifft2_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn ifft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    ifftn_device_unchecked(a, s, axes, stream)
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn try_ifft2_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    try_ifftn_device(a, s, axes, stream)
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [try_ifft2_device] for more details.
#[default_device(device = "cpu")]
pub fn ifft2_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_ifft2_device(a, s, axes, stream).unwrap()
}

fn ifftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: impl AsRef<Stream>) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = mlx_sys::mlx_fft_ifftn(
            a.c_array,
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///  padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
///  if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///   over the last `len(s)` axes are or all axes if `s` is also `None`.
///
/// # Safety
///
/// This function is unsafe because it does not check if the input arguments are valid. See
/// [`try_ifftn_device`] for a safe version of this function.
#[default_device(device = "cpu")]
pub unsafe fn ifftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    let (valid_s, valid_axes) =
        resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
    ifftn_device_inner(a, &valid_s, &valid_axes, stream)
}

/// n-dimensional inverse discrete Fourier Transform.
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
pub fn try_ifftn_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s.into_option(), axes.into_option())?;
    Ok(ifftn_device_inner(a, &valid_s, &valid_axes, stream))
}

/// n-dimensional inverse discrete Fourier Transform.
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
/// Panics if the input arguments are invalid. See [try_ifftn_device] for more details.
#[default_device(device = "cpu")]
pub fn ifftn_device<'a>(
    a: &'a Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_ifftn_device(a, s, axes, stream).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{complex64, fft::*, Array, Dtype};

    #[test]
    fn test_fft() {
        const FFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const FFT_SHAPE: &[i32] = &[4];
        const FFT_EXPECTED: &[complex64; 4] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];

        let mut array = Array::from_slice(FFT_DATA, FFT_SHAPE);
        let mut fft = fft(&array, None, None);

        assert_eq!(fft.dtype(), Dtype::Complex64);
        assert_eq!(fft.as_slice::<complex64>(), FFT_EXPECTED);

        let mut ifft = ifft(&fft, None, None);

        assert_eq!(ifft.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft.as_slice::<complex64>(),
            FFT_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // The original array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT_DATA);
    }

    #[test]
    fn test_fft2() {
        const FFT2_DATA: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        const FFT2_SHAPE: &[i32] = &[2, 2];
        const FFT2_EXPECTED: &[complex64; 4] = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let mut array = Array::from_slice(FFT2_DATA, FFT2_SHAPE);
        let mut fft2 = fft2(&array, None, None);

        assert_eq!(fft2.dtype(), Dtype::Complex64);
        assert_eq!(fft2.as_slice::<complex64>(), FFT2_EXPECTED);

        let mut ifft2 = ifft2(&fft2, None, None);

        assert_eq!(ifft2.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft2.as_slice::<complex64>(),
            FFT2_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT2_DATA);
    }

    #[test]
    fn test_fftn() {
        const FFTN_DATA: &[f32] = &[1.0; 8];
        const FFTN_SHAPE: &[i32] = &[2, 2, 2];
        const FFTN_EXPECTED: &[complex64; 8] = &[
            complex64::new(8.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let mut array = Array::from_slice(FFTN_DATA, FFTN_SHAPE);
        let mut fftn = fftn(&array, None, None);

        assert_eq!(fftn.dtype(), Dtype::Complex64);
        assert_eq!(fftn.as_slice::<complex64>(), FFTN_EXPECTED);

        let mut ifftn = ifftn(&fftn, None, None);

        assert_eq!(ifftn.dtype(), Dtype::Complex64);
        assert_eq!(
            ifftn.as_slice::<complex64>(),
            FFTN_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFTN_DATA);
    }
}
