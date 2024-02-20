use cxx::CxxVector;
use mlx_sys::{
    array::ffi::array_new_float32, cxx_vec, dtype::ffi::*, ops::ffi::*, types::{bfloat16::bfloat16_t, complex64::complex64_t, float16::float16_t}, utils::ffi::push_array, Optional
};

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
    let _converted = as_strided(&original, shape, strides, offset, s).unwrap();
}

#[test]
fn test_copy() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let _copied = copy(&original, s).unwrap();
}

#[test]
fn test_full_dtype() {
    let shape = cxx_vec![2, 5];
    let dtype = dtype_float32();
    let s = Default::default();
    let vals = array_new_float32(1.0);
    let _arr = full_dtype(&shape, &vals, dtype, s).unwrap();
}

#[test]
fn test_full() {
    let shape = cxx_vec![2, 5];
    let s = Default::default();
    let vals = array_new_float32(1.0);
    let _arr = full(&shape, &vals, s).unwrap();
}

#[test]
fn test_full_bool() {
    let shape = cxx_vec![2, 5];
    let val = true;
    let s = Default::default();
    let _arr = full_bool(&shape, val, s).unwrap();
}

#[test]
fn test_full_uint8() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_uint8(&shape, val, s).unwrap();
}

#[test]
fn test_full_uint16() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_uint16(&shape, val, s).unwrap();
}

#[test]
fn test_full_uint32() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_uint32(&shape, val, s).unwrap();
}

#[test]
fn test_full_uint64() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_uint64(&shape, val, s).unwrap();
}

#[test]
fn test_full_int8() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_int8(&shape, val, s).unwrap();
}

#[test]
fn test_full_int16() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_int16(&shape, val, s).unwrap();
}

#[test]
fn test_full_int32() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_int32(&shape, val, s).unwrap();
}

#[test]
fn test_full_int64() {
    let shape = cxx_vec![2, 5];
    let val = 1;
    let s = Default::default();
    let _arr = full_int64(&shape, val, s).unwrap();
}

#[test]
fn test_full_float16() {
    let shape = cxx_vec![2, 5];
    let val = float16_t { bits: 0x01 };
    let s = Default::default();
    let _arr = full_float16(&shape, val, s).unwrap();
}

#[test]
fn test_full_bfloat16() {
    let shape = cxx_vec![2, 5];
    let val = bfloat16_t { bits: 0x01 };
    let s = Default::default();
    let _arr = full_bfloat16(&shape, val, s).unwrap();
}

#[test]
fn test_full_float32() {
    let shape = cxx_vec![2, 5];
    let val = 1.0;
    let s = Default::default();
    let _arr = full_float32(&shape, val, s).unwrap();
}

#[test]
fn test_full_complex64() {
    let shape = cxx_vec![2, 5];
    let val = complex64_t { re: 1.0, im: 1.0 };
    let s = Default::default();
    let _arr = full_complex64(&shape, val, s).unwrap();
}

#[test]
fn test_zeros_dtype() {
    let shape = cxx_vec![2, 5];
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = zeros_dtype(&shape, dtype, s).unwrap();
}

#[test]
fn test_zeros() {
    let shape = cxx_vec![2, 5];
    let s = Default::default();
    let _arr = zeros(&shape, s).unwrap();
}

#[test]
fn test_zeros_like() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let _arr = zeros_like(&original, s).unwrap();
}

#[test]
fn test_ones_dtype() {
    let shape = cxx_vec![2, 5];
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = ones_dtype(&shape, dtype, s).unwrap();
}

#[test]
fn test_ones() {
    let shape = cxx_vec![2, 5];
    let s = Default::default();
    let _arr = ones(&shape, s).unwrap();
}

#[test]
fn test_ones_like() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let _arr = ones_like(&original, s).unwrap();
}

#[test]
fn test_eye_n_m_k_dtype() {
    let n = 5;
    let m = 5;
    let k = 0;
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = eye_n_m_k_dtype(n, m, k, dtype, s).unwrap();
}

#[test]
fn test_eye_n_dtype() {
    let n = 5;
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = eye_n_dtype(n, dtype, s).unwrap();
}

#[test]
fn test_eye_n_m() {
    let n = 5;
    let m = 5;
    let s = Default::default();
    let _arr = eye_n_m(n, m, s).unwrap();
}

#[test]
fn test_eye_n_m_k() {
    let n = 5;
    let m = 5;
    let k = 0;
    let s = Default::default();
    let _arr = eye_n_m_k(n, m, k, s).unwrap();
}

#[test]
fn test_eye_n() {
    let n = 5;
    let s = Default::default();
    let _arr = eye_n(n, s).unwrap();
}

#[test]
fn test_identity_dtype() {
    let n = 5;
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = identity_dtype(n, dtype, s).unwrap();
}

#[test]
fn test_identity() {
    let n = 5;
    let s = Default::default();
    let _arr = identity(n, s).unwrap();
}

#[test]
fn test_tri_n_m_k() {
    let n = 5;
    let m = 5;
    let k = 0;
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = tri_n_m_k(n, m, k, dtype, s).unwrap();
}

#[test]
fn test_tri_n() {
    let n = 5;
    let dtype = dtype_float32();
    let s = Default::default();
    let _arr = tri_n(n, dtype, s).unwrap();
}

#[test]
fn test_tril() {
    let shape = cxx_vec![5, 5];
    let s = Default::default();
    let x = ones(&shape, s).unwrap();
    let k = 3;
    let _arr = tril(x, k, s).unwrap();
}

#[test]
fn test_triu() {
    let shape = cxx_vec![5, 5];
    let s = Default::default();
    let x = ones(&shape, s).unwrap();
    let k = 3;
    let _arr = triu(x, k, s).unwrap();
}

#[test]
fn test_reshape() {
    let s = Default::default();
    let original = arange_start_stop_f64(0.0, 10.0, s).unwrap();
    let shape = cxx_vec![2, 5];
    let _reshaped = reshape(&original, shape, s).unwrap();
}

#[test]
fn test_flatten_start_axis_end_axis() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let ones = ones(&shape, s).unwrap();
    let _flattened = flatten_start_axis_end_axis(&ones, 0, 1, s).unwrap();
}

#[test]
fn test_flatten() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let ones = ones(&shape, s).unwrap();
    let _flattened = flatten(&ones, s).unwrap();
}

#[test]
fn test_squeeze_axes() {
    let s = Default::default();
    let shape = cxx_vec![2, 1, 5];
    let ones = ones(&shape, s).unwrap();
    let axes = cxx_vec![1];
    let _squeezed = squeeze_axes(&ones, &axes, s).unwrap();
}

#[test]
fn test_squeeze_axis() {
    let s = Default::default();
    let shape = cxx_vec![2, 1, 5];
    let ones = ones(&shape, s).unwrap();
    let _squeezed = squeeze_axis(&ones, 1, s).unwrap();
}

#[test]
fn test_squeeze() {
    let s = Default::default();
    let shape = cxx_vec![2, 1, 5];
    let ones = ones(&shape, s).unwrap();
    let _squeezed = squeeze(&ones, s).unwrap();
}

#[test]
fn test_expand_dims_at_axes() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let ones = ones(&shape, s).unwrap();
    let axes = cxx_vec![1];
    let _expanded = expand_dims_at_axes(&ones, &axes, s).unwrap();
}

#[test]
fn test_expand_dims_at_axis() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let ones = ones(&shape, s).unwrap();
    let _expanded = expand_dims_at_axis(&ones, 1, s).unwrap();
}

#[test]
fn test_slice_start_stop_strides() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let start = cxx_vec![0];
    let stop = cxx_vec![5];
    let strides = cxx_vec![1];
    let _sliced = slice_start_stop_strides(&arr, start, stop, strides, s).unwrap();
}

#[test]
fn test_slice() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let start = cxx_vec![0];
    let stop = cxx_vec![5];
    let _sliced = slice(&arr, &start, &stop, s).unwrap();
}

#[test]
fn test_split_n_at_axis() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let num_splits = 2;
    let axis = 0;
    let _splitted = split_n_at_axis(&arr, num_splits, axis, s).unwrap();
}

#[test]
fn test_split_n() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let num_splits = 2;
    let _splitted = split_n(&arr, num_splits, s).unwrap();
}

#[test]
fn test_split_at_indices_along_axis() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let indices = cxx_vec![2];
    let axis = 0;
    let _splitted = split_at_indices_along_axis(&arr, &indices, axis, s).unwrap();
}

#[test]
fn test_split_at_indices() {
    let s = Default::default();
    let arr = arange_f64(0.0, 10.0, 1.0, s).unwrap();
    let indices = cxx_vec![2];
    let _splitted = split_at_indices(&arr, &indices, s).unwrap();
}

#[test]
fn test_clip() {
    let s = Default::default();
    let dtype = dtype_float32();
    let arr = arange_start_stop_dtype(0.0, 10.0, dtype, s).unwrap();
    let a_min = Optional::Some(array_new_float32(1.0));
    let a_max = Optional::Some(array_new_float32(5.0));
    let _clipped = clip(&arr, &a_min, &a_max, s).unwrap();
}

#[test]
fn test_concatenate_along_axis() {
    let s = Default::default();
    let arr1 = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let arr2 = arange_f64(5.0, 10.0, 1.0, s).unwrap();

    let mut arrays = CxxVector::new();
    push_array(arrays.pin_mut(), arr1);
    push_array(arrays.pin_mut(), arr2);

    let axis = 0;

    let _concatenated = concatenate_along_axis(&arrays, axis, s).unwrap();
}

#[test]
fn test_concatenate() {
    let s = Default::default();
    let arr1 = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let arr2 = arange_f64(5.0, 10.0, 1.0, s).unwrap();

    let mut arrays = CxxVector::new();
    push_array(arrays.pin_mut(), arr1);
    push_array(arrays.pin_mut(), arr2);

    let _concatenated = concatenate(&arrays, s).unwrap();
}

#[test]
fn test_stack_along_axis() {
    let s = Default::default();
    let arr1 = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let arr2 = arange_f64(5.0, 10.0, 1.0, s).unwrap();

    let mut arrays = CxxVector::new();
    push_array(arrays.pin_mut(), arr1);
    push_array(arrays.pin_mut(), arr2);

    let axis = 0;

    let _stacked = stack_along_axis(&arrays, axis, s).unwrap();
}

#[test]
fn test_stack() {
    let s = Default::default();
    let arr1 = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let arr2 = arange_f64(5.0, 10.0, 1.0, s).unwrap();

    let mut arrays = CxxVector::new();
    push_array(arrays.pin_mut(), arr1);
    push_array(arrays.pin_mut(), arr2);

    let _stacked = stack(&arrays, s).unwrap();
}

#[test]
fn test_repeat_along_axis() {
    let s = Default::default();
    let arr = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let repeats = 2;
    let axis = 0;
    let _repeated = repeat_along_axis(&arr, repeats, axis, s).unwrap();
}

#[test]
fn test_repeat() {
    let s = Default::default();
    let arr = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let repeats = 2;
    let _repeated = repeat(&arr, repeats, s).unwrap();
}