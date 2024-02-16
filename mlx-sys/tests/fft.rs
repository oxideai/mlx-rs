use mlx_sys::array::ffi::*;
use mlx_sys::fft::ffi::*;
use mlx_sys::cxx_vec;

#[test]
fn test_fftn_shape_axes() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![2, 2];
    let arr = array_from_slice_float32(&data, &shape);
    let axes = cxx_vec![0, 1];
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let fft = fftn_shape_axes(&arr, &shape, &axes, stream_or_device).unwrap();
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let _ifft = ifftn_shape_axes(&fft, &shape, &axes, stream_or_device).unwrap();
}

#[test]
fn test_fftn_axes() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![2, 2];
    let arr = array_from_slice_float32(&data, &shape);
    let axes = cxx_vec![0, 1];
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let fft = fftn_axes(&arr, &axes, stream_or_device).unwrap();
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let _ifft = ifftn_axes(&fft, &axes, stream_or_device).unwrap();
}

#[test]
fn test_fftn() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![2, 2];
    let arr = array_from_slice_float32(&data, &shape);
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let fft = fftn(&arr, stream_or_device).unwrap();
    let stream_or_device = mlx_sys::StreamOrDevice::default();
    let _ifft = ifftn(&fft, stream_or_device).unwrap();
}

// TODO: all others are just calling fftn and ifftn with different parameters