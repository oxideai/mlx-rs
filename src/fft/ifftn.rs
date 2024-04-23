use mlx_macros::default_device;

use crate::{error::FftError, Array, StreamOrDevice};

use super::{
    resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked, try_resolve_size_and_axis,
    try_resolve_sizes_and_axes,
};

#[default_device(device = "cpu")]
pub unsafe fn ifft_device_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let (n, axis) = resolve_size_and_axis_unchecked(a, n, axis);
    unsafe {
        let c_array = mlx_sys::mlx_fft_ifft(a.c_array, n, axis, stream.stream.c_stream);
        Array::from_ptr(c_array)
    }
}

#[default_device(device = "cpu")]
pub fn try_ifft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (n, axis) = try_resolve_size_and_axis(a, n, axis)?;
    unsafe { Ok(ifft_device_unchecked(a, n, axis, stream)) }
}

#[default_device(device = "cpu")]
pub fn ifft_device(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_ifft_device(a, n, axis, stream).unwrap()
}

fn ifft2_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_ifft2(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device(device = "cpu")]
pub unsafe fn ifft2_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    let (valid_s, valid_axes) = resolve_sizes_and_axes_unchecked(a, s, axes);
    ifft2_device_inner(a, &valid_s, &valid_axes, stream)
}

#[default_device(device = "cpu")]
pub fn try_ifft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let axes = axes.into().unwrap_or(&[-2, -1]);
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(ifft2_device_inner(a, &valid_s, &valid_axes, stream))
}

#[default_device(device = "cpu")]
pub fn ifft2_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_ifft2_device(a, s, axes, stream).unwrap()
}

fn ifftn_device_inner(a: &Array, s: &[i32], axes: &[i32], stream: StreamOrDevice) -> Array {
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    unsafe {
        let c_array =
            mlx_sys::mlx_fft_ifftn(a.c_array, s_ptr, num_s, axes_ptr, num_axes, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device(device = "cpu")]
pub unsafe fn ifftn_device_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    let (valid_s, valid_axes) = resolve_sizes_and_axes_unchecked(a, s, axes);
    ifftn_device_inner(a, &valid_s, &valid_axes, stream)
}

#[default_device(device = "cpu")]
pub fn try_ifftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Result<Array, FftError> {
    let (valid_s, valid_axes) = try_resolve_sizes_and_axes(a, s, axes)?;
    Ok(ifftn_device_inner(a, &valid_s, &valid_axes, stream))
}

#[default_device(device = "cpu")]
pub fn ifftn_device<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
    stream: StreamOrDevice,
) -> Array {
    try_ifftn_device(a, s, axes, stream).unwrap()
}
