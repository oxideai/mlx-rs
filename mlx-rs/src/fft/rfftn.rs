use mlx_macros::default_device;

use crate::{error::Exception, utils::IntoOption, Array, Stream, StreamOrDevice};

use super::utils::{resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked};

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
pub fn rfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_rfft(a.c_array, n, axis, stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
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
pub fn rfft2_device<'a>(
    a: &Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_rfft2(
                a.c_array,
                s_ptr,
                num_s,
                axes_ptr,
                num_axes,
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
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
pub fn rfftn_device<'a>(
    a: &Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_rfftn(
                a.c_array,
                s_ptr,
                num_s,
                axes_ptr,
                num_axes,
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
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
pub fn irfft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let n = n.into();
    let axis = axis.into();
    let modify_n = n.is_none();
    let (mut n, axis) = resolve_size_and_axis_unchecked(a, n, axis);
    if modify_n {
        n = (n - 1) * 2;
    }
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_irfft(a.c_array, n, axis, stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
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
    a: &Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let s = s.into_option();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let modify_last_axis = s.is_none();

    let (mut s, axes) = resolve_sizes_and_axes_unchecked(a, s, Some(axes));
    if modify_last_axis {
        let end = s.len() - 1;
        s[end] = (s[end] - 1) * 2;
    }

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_irfft2(
                a.c_array,
                s_ptr,
                num_s,
                axes_ptr,
                num_axes,
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
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
pub fn irfftn_device<'a>(
    a: &Array,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let s = s.into_option();
    let axes = axes.into_option();
    let modify_last_axis = s.is_none();

    let (mut s, axes) = resolve_sizes_and_axes_unchecked(a, s, axes);
    if modify_last_axis {
        let end = s.len() - 1;
        s[end] = (s[end] - 1) * 2;
    }

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_fft_irfftn(
                a.c_array,
                s_ptr,
                num_s,
                axes_ptr,
                num_axes,
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[cfg(test)]
mod tests {
    use crate::{complex64, Array, Dtype};

    #[test]
    fn test_rfft() {
        const RFFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const RFFT_N: i32 = 4;
        const RFFT_SHAPE: &[i32] = &[RFFT_N];
        const RFFT_AXIS: i32 = -1;
        const RFFT_EXPECTED: &[complex64] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
        ];

        let a = Array::from_slice(RFFT_DATA, RFFT_SHAPE);
        let rfft = super::rfft(&a, RFFT_N, RFFT_AXIS).unwrap();
        assert_eq!(rfft.dtype(), Dtype::Complex64);
        assert_eq!(rfft.as_slice::<complex64>(), RFFT_EXPECTED);

        let irfft = super::irfft(&rfft, RFFT_N, RFFT_AXIS).unwrap();
        assert_eq!(irfft.dtype(), Dtype::Float32);
        assert_eq!(irfft.as_slice::<f32>(), RFFT_DATA);
    }

    #[test]
    fn test_rfft_shape_with_default_params() {
        const IN_N: i32 = 8;
        const OUT_N: i32 = IN_N / 2 + 1;

        let a = Array::ones::<f32>(&[IN_N]).unwrap();
        let rfft = super::rfft(&a, None, None).unwrap();
        assert_eq!(rfft.shape(), &[OUT_N]);
    }

    #[test]
    fn test_irfft_shape_with_default_params() {
        const IN_N: i32 = 8;
        const OUT_N: i32 = (IN_N - 1) * 2;

        let a = Array::ones::<f32>(&[IN_N]).unwrap();
        let irfft = super::irfft(&a, None, None).unwrap();
        assert_eq!(irfft.shape(), &[OUT_N]);
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
        let rfft2 = super::rfft2(&a, None, None).unwrap();
        assert_eq!(rfft2.dtype(), Dtype::Complex64);
        assert_eq!(rfft2.as_slice::<complex64>(), RFFT2_EXPECTED);

        let irfft2 = super::irfft2(&rfft2, None, None).unwrap();
        assert_eq!(irfft2.dtype(), Dtype::Float32);
        assert_eq!(irfft2.as_slice::<f32>(), RFFT2_DATA);
    }

    #[test]
    fn test_rfft2_shape_with_default_params() {
        const IN_SHAPE: &[i32] = &[6, 6];
        const OUT_SHAPE: &[i32] = &[6, 6 / 2 + 1];

        let a = Array::ones::<f32>(IN_SHAPE).unwrap();
        let rfft2 = super::rfft2(&a, None, None).unwrap();
        assert_eq!(rfft2.shape(), OUT_SHAPE);
    }

    #[test]
    fn test_irfft2_shape_with_default_params() {
        const IN_SHAPE: &[i32] = &[6, 6];
        const OUT_SHAPE: &[i32] = &[6, (6 - 1) * 2];

        let a = Array::ones::<f32>(IN_SHAPE).unwrap();
        let irfft2 = super::irfft2(&a, None, None).unwrap();
        assert_eq!(irfft2.shape(), OUT_SHAPE);
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
        let rfftn = super::rfftn(&a, None, None).unwrap();
        assert_eq!(rfftn.dtype(), Dtype::Complex64);
        assert_eq!(rfftn.as_slice::<complex64>(), RFFTN_EXPECTED);

        let irfftn = super::irfftn(&rfftn, None, None).unwrap();
        assert_eq!(irfftn.dtype(), Dtype::Float32);
        assert_eq!(irfftn.as_slice::<f32>(), RFFTN_DATA);
    }

    #[test]
    fn test_fftn_shape_with_default_params() {
        const IN_SHAPE: &[i32] = &[6, 6, 6];
        const OUT_SHAPE: &[i32] = &[6, 6, 6 / 2 + 1];

        let a = Array::ones::<f32>(IN_SHAPE).unwrap();
        let rfftn = super::rfftn(&a, None, None).unwrap();
        assert_eq!(rfftn.shape(), OUT_SHAPE);
    }

    #[test]
    fn test_irfftn_shape_with_default_params() {
        const IN_SHAPE: &[i32] = &[6, 6, 6];
        const OUT_SHAPE: &[i32] = &[6, 6, (6 - 1) * 2];

        let a = Array::ones::<f32>(IN_SHAPE).unwrap();
        let irfftn = super::irfftn(&a, None, None).unwrap();
        assert_eq!(irfftn.shape(), OUT_SHAPE);
    }
}
