use mlx_macros::default_device;

use crate::{error::FftError, Array, StreamOrDevice};

use super::utils::{
    resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked, try_resolve_size_and_axis,
    try_resolve_sizes_and_axes,
};

/// One dimensional discrete Fourier Transform on a real input.
///
/// The output has the same shape as the input except along `axis` in which case it has size `n // 2
/// + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[default_device(device = "cpu")]
pub unsafe fn rfft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n, axis);
    unsafe {
        let c_array = mlx_sys::mlx_fft_rfft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

/// One dimensional discrete Fourier Transform on a real input.
///
/// The output has the same shape as the input except along `axis` in which case it has size `n // 2
/// + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[default_device(device = "cpu")]
pub fn try_rfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (n, axis) = try_resolve_size_and_axis(a, n, axis)?;
    unsafe { Ok(rfft_device_unchecked(a, n, axis, stream)) }
}

/// One dimensional discrete Fourier Transform on a real input.
///
/// The output has the same shape as the input except along `axis` in which case it has size `n // 2
/// + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [try_rfft_device()] for more information.
#[default_device(device = "cpu")]
pub fn rfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_rfft_device(a, n, axis, stream).unwrap()
}

/// Two dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub unsafe fn rfft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    rfftn_device_unchecked(a, s, axes, stream)
}

/// Two dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn try_rfft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    try_rfftn_device(a, s, axes, stream)
}

/// Two dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [try_rfft2_device()] for more information.
#[default_device(device = "cpu")]
pub fn rfft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_rfft2_device(a, s, axes, stream).unwrap()
}

fn rfftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_rfftn(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

/// n-dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is over
///   the last `len(s)` axes or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub unsafe fn rfftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let (valid_s, valid_axes) = resolve_sizes_and_axes_unchecked(a, s, axes);
    rfftn_device_inner(a, &valid_s, &valid_axes, stream)
}

/// n-dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is over
///   the last `len(s)` axes or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub fn try_rfftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(rfftn_device_inner(a, &valid_s, &valid_axes, stream))
}

/// n-dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in `axes` in which case
/// it has sizes from `s`. The last axis in `axes` is treated as the real axis and will have size
/// `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array. If the array is complex it will be silently cast to a real type.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is over
///   the last `len(s)` axes or all axes if `s` is also `None`.
///
/// # Panic
///
/// Panics if the input arguments are invalid. See [try_rfftn_device] for more information.
#[default_device(device = "cpu")]
pub fn rfftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_rfftn_device(a, s, axes, stream).unwrap()
}

/// The inverse of [`rfft()`].
///
/// The output has the same shape as the input except along axis in which case it has size n.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n // 2 + 1`. The default value is `a.shape[axis] // 2 + 1`.
/// - `axis`: Axis along which to perform the FFT. The default is `-1`.
#[default_device(device = "cpu")]
pub unsafe fn irfft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n, axis);
    unsafe {
        let c_array = mlx_sys::mlx_fft_irfft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

/// The inverse of [`rfft()`].
///
/// The output has the same shape as the input except along axis in which case it has size n.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n // 2 + 1`. The default value is `a.shape[axis] // 2 + 1`.
/// - `axis`: Axis along which to perform the FFT. The default is `-1`.
#[default_device(device = "cpu")]
pub fn try_irfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (n, axis) = try_resolve_size_and_axis(a, n, axis)?;
    unsafe { Ok(irfft_device_unchecked(a, n, axis, stream)) }
}

/// The inverse of [`rfft()`].
///
/// The output has the same shape as the input except along axis in which case it has size n.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n // 2 + 1`. The default value is `a.shape[axis] // 2 + 1`.
/// - `axis`: Axis along which to perform the FFT. The default is `-1`.
#[default_device(device = "cpu")]
pub fn irfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_irfft_device(a, n, axis, stream).unwrap()
}

/// The inverse of [`rfft2()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub unsafe fn irfft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    irfftn_device_unchecked(a, s, axes, stream)
}

/// The inverse of [`rfft2()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn try_irfft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    try_irfftn_device(a, s, axes, stream)
}

/// The inverse of [`rfft2()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[default_device(device = "cpu")]
pub fn irfft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_irfft2_device(a, s, axes, stream).unwrap()
}

fn irfftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_irfftn(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

/// The inverse of [`rfftn()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///  over the last `len(s)` axes or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub unsafe fn irfftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let (valid_s, valid_axes) = resolve_sizes_and_axes_unchecked(a, s, axes);
    irfftn_device_inner(a, &valid_s, &valid_axes, stream)
}

/// The inverse of [`rfftn()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///  over the last `len(s)` axes or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub fn try_irfftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(irfftn_device_inner(a, &valid_s, &valid_axes, stream))
}

/// The inverse of [`rfftn()`].
///
/// Note the input is generally complex. The dimensions of the input specified in `axes` are padded
/// or truncated to match the sizes from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.len()-1] // 2 + 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///   padded with zeros to match the sizes in `s` except for the last axis which has size
///   `s[s.len()-1] // 2 + 1`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///  over the last `len(s)` axes or all axes if `s` is also `None`.
#[default_device(device = "cpu")]
pub fn irfftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_irfftn_device(a, s, axes, stream).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{complex64, fft::*, Array, Dtype};

    #[test]
    fn test_rfft() {
        const RFFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const RFFT_SHAPE: &[i32] = &[4];
        const RFFT_N: i32 = 4;
        const RFFT_AXIS: i32 = -1;
        const RFFT_EXPECTED: &[complex64] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
        ];

        let a = Array::from_slice(RFFT_DATA, RFFT_SHAPE);
        let mut rfft = rfft(&a, RFFT_N, RFFT_AXIS);
        rfft.eval();
        assert_eq!(rfft.dtype(), Dtype::Complex64);
        assert_eq!(rfft.as_slice::<complex64>(), RFFT_EXPECTED);

        let mut irfft = irfft(&rfft, RFFT_N, RFFT_AXIS);
        irfft.eval();
        assert_eq!(irfft.dtype(), Dtype::Float32);
        assert_eq!(irfft.as_slice::<f32>(), RFFT_DATA);
    }

    #[test]
    fn test_rfft2() {
        const RFFT2_DATA: &[f32] = &[1.0; 4];
        const RFFT2_SHAPE: &[i32] = &[2, 2];
        const RFFT2_EXPECTED: &[complex64] = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let a = Array::from_slice(RFFT2_DATA, RFFT2_SHAPE);
        let mut rfft2 = rfft2(&a, None, None);
        rfft2.eval();
        assert_eq!(rfft2.dtype(), Dtype::Complex64);
        assert_eq!(rfft2.as_slice::<complex64>(), RFFT2_EXPECTED);

        let mut irfft2 = irfft2(&rfft2, None, None);
        irfft2.eval();
        assert_eq!(irfft2.dtype(), Dtype::Float32);
        assert_eq!(irfft2.as_slice::<f32>(), RFFT2_DATA);
    }

    #[test]
    fn test_rfftn() {
        const RFFTN_DATA: &[f32] = &[1.0; 8];
        const RFFTN_SHAPE: &[i32] = &[2, 2, 2];
        const RFFTN_EXPECTED: &[complex64] = &[
            complex64::new(8.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let a = Array::from_slice(RFFTN_DATA, RFFTN_SHAPE);
        let mut rfftn = rfftn(&a, None, None);
        rfftn.eval();
        assert_eq!(rfftn.dtype(), Dtype::Complex64);
        assert_eq!(rfftn.as_slice::<complex64>(), RFFTN_EXPECTED);

        let mut irfftn = irfftn(&rfftn, None, None);
        irfftn.eval();
        assert_eq!(irfftn.dtype(), Dtype::Float32);
        assert_eq!(irfftn.as_slice::<f32>(), RFFTN_DATA);
    }
}
