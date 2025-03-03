//! This contains the tests for some of the exported macros.
//! 
//! This is mainly a sanity check to ensure that the exported macros are working as expected.

use mlx_rs::{array, ops::{arange, reshape}, Array, StreamOrDevice};

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