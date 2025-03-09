//! This contains the tests for some of the exported macros.
//!
//! This is mainly a sanity check to ensure that the exported macros are working as expected.

use mlx_rs::{
    array, complex64, ops::{arange, reshape}, Array, Dtype, StreamOrDevice
};

// Try two functions that don't have any optional arguments.

#[test]
fn test_ops_arithmetic_abs() {
    let data = array!([1i32, 2, -3, -4, -5]);
    let result = mlx_rs::abs!(&data).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5]);

    let stream = StreamOrDevice::cpu();
    let result = mlx_rs::abs!(data, stream = stream).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_ops_arithmetic_add() {
    let data1 = array!([1i32, 2, 3, 4, 5]);
    let data2 = array!([1i32, 2, 3, 4, 5]);
    let result = mlx_rs::add!(&data1, &data2).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[2, 4, 6, 8, 10]);

    let stream = StreamOrDevice::cpu();
    let result = mlx_rs::add!(data1, data2, stream = stream).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[2, 4, 6, 8, 10]);
}

// Try a function that has optional arguments.

#[test]
fn test_ops_arithmetic_tensordot() {
    let x = reshape(arange::<_, f32>(None, 60.0, None).unwrap(), &[3, 4, 5]).unwrap();
    let y = reshape(arange::<_, f32>(None, 24.0, None).unwrap(), &[4, 3, 2]).unwrap();
    let axes = (&[1, 0], &[0, 1]);
    let z = mlx_rs::tensordot!(&x, &y, axes = axes).unwrap();
    let expected = Array::from_slice(
        &[4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306],
        &[5, 2],
    );
    assert_eq!(z, expected);

    let stream = StreamOrDevice::cpu();
    let z = mlx_rs::tensordot!(x, y, axes = axes, stream = stream).unwrap();
    assert_eq!(z, expected);
}

// Test functions defined in `mlx_rs::ops` module.

#[test]
fn test_ops_convolution_conv1d() {
    let input = array!(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        shape = [1, 5, 2]
    );
    let weight = array!(
        [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0],
        shape = [2, 3, 2]
    );

    let result = mlx_rs::conv1d!(
        &input,
        &weight,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
    )
    .unwrap();

    let expected = array!([12.0, 8.0, 17.0, 13.0, 22.0, 18.0], shape = [1, 3, 2]);
    assert_eq!(result, expected);
}

#[test]
fn test_ops_factory_arange() {
    // Without specifying start and step
    let array = mlx_rs::arange!(stop = 50).unwrap();
    assert_eq!(array.shape(), &[50]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());

    // With specifying start and step
    let array = mlx_rs::arange!(start = 1.0, stop = 50.0, step = 2.0).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());

    let stream = StreamOrDevice::cpu();
    let array = mlx_rs::arange!(start = 1.0, stop = 50.0, step = 2.0, stream = stream).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());
}

// Test functions defined in `mlx_rs::fft` module.

#[test]
fn test_fft_fft() {
    const FFT_EXPECTED: &[complex64; 4] = &[
        complex64::new(10.0, 0.0),
        complex64::new(-2.0, 2.0),
        complex64::new(-2.0, 0.0),
        complex64::new(-2.0, -2.0),
    ];

    let data = array!([1.0, 2.0, 3.0, 4.0]);
    let fft = mlx_rs::fft!(&data).unwrap();

    assert_eq!(fft.dtype(), Dtype::Complex64);
    assert_eq!(fft.as_slice::<complex64>(), FFT_EXPECTED);
}

// Test functions defined in `mlx_rs::linalg` module.

#[test]
fn test_linalg_norm() {
    let a = array!([1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    let norm = mlx_rs::norm!(&a).unwrap();
    assert_eq!(norm.item::<f32>(), 5.477225575051661);
}

// Test functions defined in `mlx_rs::random` module.

#[test]
fn test_random_uniform() {
    let value = mlx_rs::uniform!(0.0, 1.0, shape=&[1]).unwrap();
    assert_eq!(value.shape(), &[1]);
    assert!(value.item::<f32>() >= 0.0 && value.item::<f32>() <= 1.0);
}

#[test]
fn test_random_normal() {
    let value = mlx_rs::normal!(shape=&[1]).unwrap();
    assert_eq!(value.shape(), &[1]);
    assert!(value.item::<f32>() >= -10.0 && value.item::<f32>() <= 10.0);
}