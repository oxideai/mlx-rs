use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    array::Array, error::FftnError, stream::StreamOrDevice, utils::resolve_index_unchecked,
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
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
///
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
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
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn try_fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftnError> {
    let (n, axis) = super::try_resolve_size_and_axis(a, n, axis)?;
    Ok(unsafe { fft_device_unchecked(a, Some(n), Some(axis), stream) })
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
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn fft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_fft_device(a, n, axis, stream).unwrap()
}

fn fft2_device_inner(a: &Array, n: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_axes = axes.len();
    let num_n = n.len();

    let n_ptr = n.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_fft2(a.c_array, n_ptr, num_n, axes_ptr, num_axes, stream.as_ptr());
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
///
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
/// let mut result = unsafe {
///     fft2_device_unchecked(&array, &[2, 2][..], &[-2,-1][..], StreamOrDevice::cpu())
/// };
/// result.eval();
///
/// assert_eq!(result.dtype(), Dtype::Complex64);
///
/// let expected = &[
///    complex64::new(4.0, 0.0),
///    complex64::new(0.0, 0.0),
///    complex64::new(0.0, 0.0),
///    complex64::new(0.0, 0.0),
/// ];
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub unsafe fn fft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    let mut valid_n = SmallVec::<[i32; 2]>::new();
    match s.into() {
        Some(s) => valid_n.extend_from_slice(&s),
        None => {
            for axis in axes {
                let axis_index = resolve_index_unchecked(*axis, a.ndim());
                valid_n.push(a.shape()[axis_index]);
            }
        }
    }

    fft2_device_inner(a, &valid_n, axes, stream)
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
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
/// let mut result = try_fft2_device(&array, None, None, StreamOrDevice::cpu()).unwrap();
/// result.eval();
/// assert_eq!(result.dtype(), Dtype::Complex64);
/// let expected = &[
///     complex64::new(4.0, 0.0),
///     complex64::new(0.0, 0.0),
///     complex64::new(0.0, 0.0),
///     complex64::new(0.0, 0.0),
/// ];
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn try_fft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftnError> {
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
/// - if the input array is a scalar array
/// - if the shape and axes have different sizes
/// - if more axes are provided than the array has
/// - if the output sizes are invalid (<= 0)
/// - if the axes are not unique
///
/// See [`try_fft2_device`] for more details.
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
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

/// N-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
///  padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
///  if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
///   over the last `len(s)` axes are or all axes if `s` is also None.
///
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::ones::<f32>(&[3, 3, 3]);
///
/// let mut result = unsafe { fftn_device_unchecked(&array, None, None, StreamOrDevice::cpu()) };
/// result.eval();
///
/// assert_eq!(result.dtype(), Dtype::Complex64);
///
/// let mut expected = vec![complex64::new(0.0, 0.0); 27];
/// expected[0] = complex64::new(27.0, 0.0);
///
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub unsafe fn fftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let (valid_s, valid_axes) = match (s.into(), axes.into()) {
        (Some(s), Some(axes)) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (Some(s), None) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
        (None, Some(axes)) => {
            let valid_s = axes
                .iter()
                .map(|&axis| {
                    let axis_index = resolve_index_unchecked(axis, a.ndim());
                    a.shape()[axis_index]
                })
                .collect();
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (None, None) => {
            let valid_s: SmallVec<[i32; 4]> = (0..a.ndim()).map(|axis| a.shape()[axis]).collect();
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
    };

    fftn_device_inner(a, &valid_s, &valid_axes, stream)
}

/// N-dimensional discrete Fourier Transform.
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
/// # Example
///
/// ```rust
/// use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
///
/// let array = Array::ones::<f32>(&[3, 3, 3]);
///
/// let mut result = try_fftn(&array, None, None).unwrap();
/// result.eval();
///
/// assert_eq!(result.dtype(), Dtype::Complex64);
///
/// let mut expected = vec![complex64::new(0.0, 0.0); 27];
/// expected[0] = complex64::new(27.0, 0.0);
///
/// assert_eq!(result.as_slice::<complex64>(), &expected[..]);
/// ```
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
pub fn try_fftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftnError> {
    let (valid_s, valid_axes) = super::try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(fftn_device_inner(a, &valid_s, &valid_axes, stream))
}

/// N-dimensional discrete Fourier Transform.
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
#[default_device(device = "cpu")] // fft is not implemented on GPU yet
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
        use crate::{complex64, error::FftnError, fft::*, Array};

        let array = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fft2(&scalar_array, None, None);
        assert_eq!(result.unwrap_err(), FftnError::ScalarArray);

        let result = try_fft2(&array, &[2, 2, 2][..], &[0, 1, 2][..]);
        assert_eq!(result.unwrap_err(), FftnError::InvalidAxis { ndim: 2 });

        let result = try_fft2(&array, &[2, 2][..], &[-1][..]);
        assert_eq!(
            result.unwrap_err(),
            FftnError::IncompatibleShapeAndAxes {
                shape_size: 2,
                axes_size: 1,
            }
        );

        let result = try_fft2(&array, None, &[-2, -2][..]);
        assert_eq!(result.unwrap_err(), FftnError::DuplicateAxis { axis: -2 });

        let result = try_fft2(&array, &[-2, 2][..], None);
        assert_eq!(result.unwrap_err(), FftnError::InvalidOutputSize);

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
        use crate::{complex64, error::FftnError, fft::*, Array};

        let array = Array::ones::<f32>(&[3, 3, 3]);

        // Error case
        let scalar_array = Array::from_float(1.0);
        let result = try_fftn(&scalar_array, None, None);
        assert_eq!(result.unwrap_err(), FftnError::ScalarArray);

        let result = try_fftn(&array, &[3, 3, 3, 3][..], &[0, 1, 2, 3][..]);
        assert_eq!(result.unwrap_err(), FftnError::InvalidAxis { ndim: 3 });

        let result = try_fftn(&array, &[3, 3, 3][..], &[-1][..]);
        assert_eq!(
            result.unwrap_err(),
            FftnError::IncompatibleShapeAndAxes {
                shape_size: 3,
                axes_size: 1,
            }
        );

        let result = try_fftn(&array, None, &[-2, -2][..]);
        assert_eq!(result.unwrap_err(), FftnError::DuplicateAxis { axis: -2 });

        let result = try_fftn(&array, &[-2, 2][..], None);
        assert_eq!(result.unwrap_err(), FftnError::InvalidOutputSize);

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
