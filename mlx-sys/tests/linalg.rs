use cxx::let_cxx_string;

use mlx_sys::cxx_vec;
use mlx_sys::utils::CloneCxxVector;

#[test]
fn test_norm_ord() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![4];
    let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
    let ord = 2.0;
    let axis = mlx_sys::Optional::None;
    let keepdims = false;
    let s = mlx_sys::utils::StreamOrDevice::default();
    let _result = mlx_sys::linalg::ffi::norm_ord(&arr, ord, &axis, keepdims, s).unwrap();
}

#[test]
fn test_norm_ord_axis() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![4];
    let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
    let ord = 2.0;
    let axis = 0;
    let keepdims = false;
    let s = mlx_sys::utils::StreamOrDevice::default();
    let _result = mlx_sys::linalg::ffi::norm_ord_axis(&arr, ord, axis, keepdims, s).unwrap();
}

#[test]
fn test_norm_str_ord() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![2,2];
    let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
    let_cxx_string!(ord = "fro");
    let axis = mlx_sys::Optional::None;
    let keepdims = false;
    let s = mlx_sys::utils::StreamOrDevice::default();
    let _result = mlx_sys::linalg::ffi::norm_str_ord(&arr, &ord, &axis, keepdims, s).unwrap();
}

// // TODO: it only supports matrix but only one axis is taken, so it will always throw an exception
// #[test]
// fn test_norm_str_ord_axis() {
//     let data = [1.0, 2.0, 3.0, 4.0];
//     let shape = cxx_vec![2,2];
//     let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
//     let_cxx_string!(ord = "inf");
//     let axis = 0;
//     let keepdims = false;
//     let s = mlx_sys::StreamOrDevice::default();
//     let _result = mlx_sys::linalg::ffi::norm_str_ord_axis(&arr, &ord, axis, keepdims, s).unwrap();
// }

#[test]
fn test_norm() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![4];
    let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
    let axis = mlx_sys::Optional::None;
    let keepdims = false;
    let s = mlx_sys::utils::StreamOrDevice::default();
    let _result = mlx_sys::linalg::ffi::norm(&arr, &axis, keepdims, s).unwrap();
}

#[test]
fn test_norm_axis() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = cxx_vec![4];
    let arr = mlx_sys::array::ffi::array_from_slice_float32(&data, shape.clone());
    let axis = 0;
    let keepdims = false;
    let s = mlx_sys::utils::StreamOrDevice::default();
    let _result = mlx_sys::linalg::ffi::norm_axis(&arr, axis, keepdims, s).unwrap();
}