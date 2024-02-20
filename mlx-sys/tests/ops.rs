use mlx_sys::{cxx_vec, dtype::ffi::*, ops::ffi::*};

#[test]
fn test_arange_f64() {
    let _arr = arange_f64(0.0, 1.0, 1.0, Default::default()).unwrap();
}

#[test]
fn test_arange_start_stop_dtype() {
    let dtype = dtype_float32();
    let _arr = arange_start_stop_dtype(0.0, 10.0, dtype, Default::default()).unwrap();
}

#[test]
fn test_arange_start_stop_f64() {
    let _arr = arange_start_stop_f64(0.0, 10.0, Default::default()).unwrap();
}

#[test]
fn test_arange_stop_dtype() {
    let dtype = dtype_float32();
    let _arr = arange_stop_dtype(10.0, dtype, Default::default()).unwrap();
}

#[test]
fn test_arange_stop_f64() {
    let _arr = arange_stop_f64(10.0, Default::default()).unwrap();
}

#[test]
fn test_arange_i32() {
    let _arr = arange_i32(0, 10, 1, Default::default()).unwrap();
}

#[test]
fn test_arange_start_stop_i32() {
    let _arr = arange_start_stop_i32(0, 10, Default::default()).unwrap();
}

#[test]
fn test_arange_stop_i32() {
    let _arr = arange_stop_i32(10, Default::default()).unwrap();
}

#[test]
fn test_linspace() {
    let dtype = dtype_float32();
    let _arr = linspace(0.0, 10.0, 100, dtype, Default::default()).unwrap();
}

#[test]
fn test_astype() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let _converted = astype(&original, dtype_int32(), s).unwrap();
}

#[test]
fn test_as_strided() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let shape = cxx_vec![2, 5];
    let strides = cxx_vec![1, 2];
    let offset = 0;
    let _converted = as_strided(&original, &shape, &strides, offset, s).unwrap();
}

#[test]
fn test_copy() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let _copied = copy(&original, s).unwrap();
}