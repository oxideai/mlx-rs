use cxx::CxxVector;
use mlx_sys::{
    array::ffi::{array_from_slice_bool, array_from_slice_int32, array_new_float32}, cxx_vec, dtype::ffi::*, ops::ffi::*, types::{bfloat16::bfloat16_t, complex64::complex64_t, float16::float16_t}, utils::ffi::push_array, Optional
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

#[test]
fn test_tile() {
    let s = Default::default();
    let arr = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let reps = cxx_vec![2, 2];
    let _tiled = tile(&arr, reps, s).unwrap();
}

#[test]
fn test_transpose_axes() {
    let s = Default::default();
    let shape = cxx_vec![1, 2, 3];
    let arr = ones(&shape, s).unwrap();
    let axes = cxx_vec![1, 0, 2];
    let _transposed = transpose_axes(&arr, axes, s).unwrap();
}

#[test]
fn test_swapaxes() {
    let s = Default::default();
    let shape = cxx_vec![1, 2, 3];
    let arr = ones(&shape, s).unwrap();
    let axis1 = 0;
    let axis2 = 1;
    let _swapped = swapaxes(&arr, axis1, axis2, s).unwrap();
}

#[test]
fn test_moveaxis() {
    let s = Default::default();
    let shape = cxx_vec![3, 4, 5];
    let arr = ones(&shape, s).unwrap();
    let _moved = moveaxis(&arr, 0, -1, s).unwrap();
}

#[test]
fn test_pad_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let arr = ones(&shape, s).unwrap();
    let axes = cxx_vec![0, 1];
    let low_pad_size = cxx_vec![1, 1];
    let high_pad_size = cxx_vec![1, 1];
    let pad_value = array_new_float32(0.0);
    let _padded = pad_axes(&arr, &axes, &low_pad_size, &high_pad_size, &pad_value, s).unwrap();
}

#[test]
fn test_pad_each() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let arr = ones(&shape, s).unwrap();
    let pad_width = [[1, 1], [1, 1]];
    let pad_value = array_new_float32(0.0);
    let _padded = pad_each(&arr, &pad_width[..], &pad_value, s).unwrap();
}

#[test]
fn test_pad_same() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let arr = ones(&shape, s).unwrap();
    let pad_width = [1, 1];
    let pad_value = array_new_float32(0.0);
    let _padded = pad_same(&arr, &pad_width, &pad_value, s).unwrap();
}

#[test]
fn test_pad() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let arr = ones(&shape, s).unwrap();
    let pad_width = 1;
    let pad_value = array_new_float32(0.0);
    let _padded = pad(&arr, pad_width, &pad_value, s).unwrap();
}

#[test]
fn test_transpose() {
    let s = Default::default();
    let shape = cxx_vec![3, 2];
    let arr = ones(&shape, s).unwrap();
    let _transposed = transpose(&arr, s).unwrap();
}

#[test]
fn test_broadcast_to() {
    let s = Default::default();
    let a = array_new_float32(1.0);
    let shape = cxx_vec![3, 2];
    let _broadcasted = broadcast_to(&a, &shape, s).unwrap();
}

#[test]
fn test_broadcast_arrays() {
    let s = Default::default();
    let a = array_new_float32(1.0);
    let shape = cxx_vec![3, 3];
    let dtype = dtype_float32();
    let b = ones_dtype(&shape, dtype, s).unwrap();
    let mut inputs = CxxVector::new();
    push_array(inputs.pin_mut(), a);
    push_array(inputs.pin_mut(), b);

    let _broadcasted = broadcast_arrays(&inputs, s).unwrap();
}

#[test]
fn test_equal() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _equal = equal(&a, &b, s).unwrap();
}

#[test]
fn test_not_equal() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _not_equal = not_equal(&a, &b, s).unwrap();
}

#[test]
fn test_greater() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _greater = greater(&a, &b, s).unwrap();
}

#[test]
fn test_greater_equal() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _greater_equal = greater_equal(&a, &b, s).unwrap();
}

#[test]
fn test_less() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _less = less(&a, &b, s).unwrap();
}

#[test]
fn test_less_equal() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _less_equal = less_equal(&a, &b, s).unwrap();
}

#[test]
fn test_array_equal() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _array_equal = array_equal(&a, &b, s).unwrap();
}

#[test]
fn test_isnan() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _isnan = isnan(&a, s).unwrap();
}

#[test]
fn test_isinf() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _isinf = isinf(&a, s).unwrap();
}

#[test]
fn test_isposinf() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _isposinf = isposinf(&a, s).unwrap();
}

#[test]
fn test_isneginf() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _isneginf = isneginf(&a, s).unwrap();
}

#[test]
fn test_where_condition() {
    let s = Default::default();
    let shape = cxx_vec![5];
    let condition = array_from_slice_bool(&[false, true, false, true, false], &shape);
    let x = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let y = arange_f64(5.0, 10.0, 1.0, s).unwrap();
    let _where = where_condition(&condition, &x, &y, s).unwrap();
}

#[test]
fn test_all_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = true;
    let _all = all_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_all() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let _all = all(&a, s).unwrap();
}

#[test]
fn test_allclose() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let rtol = 1e-05;
    let atol = 1e-08;
    let equal_nan = false;
    let _allclose = allclose(&a, &b, rtol, atol, equal_nan, s).unwrap();
}

#[test]
fn test_isclose() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let rtol = 1e-05;
    let atol = 1e-08;
    let equal_nan = false;
    let _isclose = isclose(&a, &b, rtol, atol, equal_nan, s).unwrap();
}

#[test]
fn test_all_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = true;
    let _all_along_axes = all_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_all_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = true;
    let _all_along_axis = all_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_any_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _any = any_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_any() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let _any = any(&a, s).unwrap();
}

#[test]
fn test_any_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = true;
    let _any_along_axes = any_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_any_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = true;
    let _any_along_axis = any_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_sum_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = true;
    let _sum = sum_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_sum() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _sum = sum(&a, s).unwrap();
}

#[test]
fn test_sum_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = true;
    let _sum_along_axes = sum_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_sum_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = true;
    let _sum_along_axis = sum_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_mean_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = true;
    let _mean = mean_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_mean() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _mean = mean(&a, s).unwrap();
}

#[test]
fn test_mean_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = true;
    let _mean_along_axes = mean_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_mean_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = true;
    let _mean_along_axis = mean_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_var_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = true;
    let ddof = 0;
    let _var = var_keepdims(&a, keepdims, ddof, s).unwrap();
}

#[test]
fn test_var() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _var = var(&a, s).unwrap();
}

#[test]
fn test_var_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = true;
    let ddof = 0;
    let _var_along_axes = var_along_axes_keepdims(&a, &axes, keepdims, ddof, s).unwrap();
}

#[test]
fn test_var_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = true;
    let ddof = 0;
    let _var_along_axis = var_along_axis_keepdims(&a, axis, keepdims, ddof, s).unwrap();
}

#[test]
fn test_prod_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _prod = prod_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_prod() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _prod = prod(&a, s).unwrap();
}

#[test]
fn test_prod_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = false;
    let _prod_along_axes = prod_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_prod_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _prod_along_axis = prod_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_max_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _max = max_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_max() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _max = max(&a, s).unwrap();
}

#[test]
fn test_max_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = false;
    let _max_along_axes = max_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_max_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _max_along_axis = max_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_min_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _min = min_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_min() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _min = min(&a, s).unwrap();
}

#[test]
fn test_min_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = false;
    let _min_along_axes = min_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_min_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _min_along_axis = min_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_argmin_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _argmin = argmin_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_argmin() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _argmin = argmin(&a, s).unwrap();
}

#[test]
fn test_argmin_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _argmin_along_axis = argmin_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_argmax_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _argmax = argmax_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_argmax() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _argmax = argmax(&a, s).unwrap();
}

#[test]
fn test_argmax_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _argmax_along_axis = argmax_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_sort() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _sorted = sort(&a, s).unwrap();
}

#[test]
fn test_sort_along_axis() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let axis = 0;
    let _sorted = sort_along_axis(&a, axis, s).unwrap();
}

#[test]
fn test_argsort() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _argsorted = argsort(&a, s).unwrap();
}

#[test]
fn test_argsort_along_axis() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let axis = 0;
    let _argsorted = argsort_along_axis(&a, axis, s).unwrap();
}

#[test]
fn test_partition() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let kth = 3;
    let _partitioned = partition(&a, kth, s).unwrap();
}

#[test]
fn test_partition_along_axis() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let kth = 3;
    let axis = 0;
    let _partitioned = partition_along_axis(&a, kth, axis, s).unwrap();
}

#[test]
fn test_argpartition() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let kth = 3;
    let _argpartitioned = argpartition(&a, kth, s).unwrap();
}

#[test]
fn test_argpartition_along_axis() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let kth = 3;
    let axis = 0;
    let _argpartitioned = argpartition_along_axis(&a, kth, axis, s).unwrap();
}

#[test]
fn test_topk() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let k = 3;
    let _topk = topk(&a, k, s).unwrap();
}

#[test]
fn test_topk_along_axis() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let k = 3;
    let axis = 0;
    let _topk = topk_along_axis(&a, k, axis, s).unwrap();
}

#[test]
fn test_logsumexp_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let keepdims = false;
    let _logsumexp = logsumexp_keepdims(&a, keepdims, s).unwrap();
}

#[test]
fn test_logsumexp() {
    let s = Default::default();
    let shape = cxx_vec![1, 5];
    let a = ones(&shape, s).unwrap();
    let _logsumexp = logsumexp(&a, s).unwrap();
}

#[test]
fn test_logsumexp_along_axes_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let keepdims = false;
    let _logsumexp_along_axes = logsumexp_along_axes_keepdims(&a, &axes, keepdims, s).unwrap();
}

#[test]
fn test_logsumexp_along_axis_keepdims() {
    let s = Default::default();
    let shape = cxx_vec![2, 5];
    let a = ones(&shape, s).unwrap();
    let axis = 0;
    let keepdims = false;
    let _logsumexp_along_axis = logsumexp_along_axis_keepdims(&a, axis, keepdims, s).unwrap();
}

#[test]
fn test_abs() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _abs = abs(&a, s).unwrap();
}

#[test]
fn test_negative() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _negative = negative(&a, s).unwrap();
}

#[test]
fn test_sign() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _sign = sign(&a, s).unwrap();
}

#[test]
fn test_logical_not() {
    let s = Default::default();
    let shape = cxx_vec![5];
    let a = array_from_slice_bool(&[true, false, true, false, true], &shape);
    let _logical_not = logical_not(&a, s).unwrap();
}

#[test]
fn test_logical_and() {
    let s = Default::default();
    let shape = cxx_vec![5];
    let a = array_from_slice_bool(&[true, false, true, false, true], &shape);
    let b = array_from_slice_bool(&[true, true, false, false, true], &shape);
    let _logical_and = logical_and(&a, &b, s).unwrap();
}

#[test]
fn test_logical_or() {
    let s = Default::default();
    let shape = cxx_vec![5];
    let a = array_from_slice_bool(&[true, false, true, false, true], &shape);
    let b = array_from_slice_bool(&[true, true, false, false, true], &shape);
    let _logical_or = logical_or(&a, &b, s).unwrap();
}

#[test]
fn test_reciprocal() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 1.0, s).unwrap();
    let _reciprocal = reciprocal(&a, s).unwrap();
}

#[test]
fn test_add() {
    let s = Default::default();
    let a = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let b = arange_f64(5.0, 10.0, 1.0, s).unwrap();
    let _added = add(&a, &b, s).unwrap();
}

#[test]
fn test_subtract() {
    let s = Default::default();
    let a = arange_f64(5.0, 10.0, 1.0, s).unwrap();
    let b = arange_f64(0.0, 5.0, 1.0, s).unwrap();
    let _subtracted = subtract(&a, &b, s).unwrap();
}

#[test]
fn test_multiply() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _multiplied = multiply(&a, &b, s).unwrap();
}

#[test]
fn test_divide() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _divided = divide(&a, &b, s).unwrap();
}

#[test]
fn test_divmod() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _divmod = divmod(&a, &b, s).unwrap();
}

#[test]
fn test_floor_divide() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _floor_divided = floor_divide(&a, &b, s).unwrap();
}

#[test]
fn test_remainder() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _remainder = remainder(&a, &b, s).unwrap();
}

#[test]
fn test_maximum() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _maximum = maximum(&a, &b, s).unwrap();
}

#[test]
fn test_minimum() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _minimum = minimum(&a, &b, s).unwrap();
}

#[test]
fn test_floor() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _floored = floor(&a, s).unwrap();
}

#[test]
fn test_ceil() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _ceiled = ceil(&a, s).unwrap();
}

#[test]
fn test_square() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _squared = square(&a, s).unwrap();
}

#[test]
fn test_exp() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _exponentiated = exp(&a, s).unwrap();
}

#[test]
fn test_sin() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _sined = sin(&a, s).unwrap();
}

#[test]
fn test_cos() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _cosined = cos(&a, s).unwrap();
}

#[test]
fn test_tan() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _taned = tan(&a, s).unwrap();
}

#[test]
fn test_arcsin() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _arcsined = arcsin(&a, s).unwrap();
}

#[test]
fn test_arccos() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _arccosed = arccos(&a, s).unwrap();
}

#[test]
fn test_arctan() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _arctaned = arctan(&a, s).unwrap();
}

#[test]
fn test_sinh() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _sinhed = sinh(&a, s).unwrap();
}

#[test]
fn test_cosh() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 1.0, s).unwrap();
    let _coshed = cosh(&a, s).unwrap();
}

#[test]
fn test_tanh() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _tanhed = tanh(&a, s).unwrap();
}

#[test]
fn test_arcsinh() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _arcsinhed = arcsinh(&a, s).unwrap();
}

#[test]
fn test_arccosh() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _arccoshed = arccosh(&a, s).unwrap();
}

#[test]
fn test_arctanh() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _arctanhed = arctanh(&a, s).unwrap();
}

#[test]
fn test_log() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _logarithmed = log(&a, s).unwrap();
}

#[test]
fn test_log2() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _log2ed = log2(&a, s).unwrap();
}

#[test]
fn test_log10() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _log10ed = log10(&a, s).unwrap();
}

#[test]
fn test_log1p() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _log1ped = log1p(&a, s).unwrap();
}

#[test]
fn test_logaddexp() {
    let s = Default::default();
    let a = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let b = arange_f64(1.0, 5.0, 0.1, s).unwrap();
    let _logaddexped = logaddexp(&a, &b, s).unwrap();
}

#[test]
fn test_sigmoid() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 0.1, s).unwrap();
    let _sigmoided = sigmoid(&a, s).unwrap();
}

#[test]
fn test_erf() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 0.1, s).unwrap();
    let _erfed = erf(&a, s).unwrap();
}

#[test]
fn test_erfinv() {
    let s = Default::default();
    let a = arange_f64(-1.0, 1.0, 0.1, s).unwrap();
    let _erfinved = erfinv(&a, s).unwrap();
}

#[test]
fn test_stop_gradient() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 0.1, s).unwrap();
    let _stop_gradient = stop_gradient(&a, s).unwrap();
}

#[test]
fn test_round() {
    let s = Default::default();
    let a = arange_f64(-5.0, 5.0, 0.1, s).unwrap();
    let _rounded = round(&a, s).unwrap();
}

#[test]
fn test_matmaul() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let b = ones(&shape, s).unwrap();
    let _matmauled = matmul(&a, &b, s).unwrap();
}

#[test]
fn test_gather_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let axes = cxx_vec![0];
    let slice_sizes = cxx_vec![3, 3];
    let _gathered = gather_along_axes(&a, &indices, &axes, &slice_sizes, s).unwrap();
}

#[test]
fn test_gather_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let axis = 0;
    let slice_sizes = cxx_vec![3, 3];
    let _gathered = gather_along_axis(&a, &indices, axis, &slice_sizes, s).unwrap();
}

#[test]
fn test_take() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let axis = 0;
    let _taken = take(&a, &indices, axis, s).unwrap();
}

#[test]
fn test_take_flattened() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let _taken = take_flattened(&a, &indices, s).unwrap();
}

#[test]
fn test_take_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let indices = ones_dtype(&shape, dtype_uint32(), s).unwrap();
    let axis = 0;
    let _taken = take_along_axis(&a, &indices, axis, s).unwrap();
}

#[test]
fn test_scatter_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = zeros(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let _scattered = scatter_along_axes(&a, &indices, &updates, &axes, s).unwrap();
}

#[test]
fn test_scatter_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = zeros(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axis = 0;
    let _scattered = scatter_along_axis(&a, &indices, &updates, axis, s).unwrap();
}

#[test]
fn test_scatter_add_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = zeros(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let _scattered = scatter_add_along_axes(&a, &indices, &updates, &axes, s).unwrap();
}

#[test]
fn test_scatter_add_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = zeros(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axis = 0;
    let _scattered = scatter_add_along_axis(&a, &indices, &updates, axis, s).unwrap();
}

#[test]
fn test_scatter_prod_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let _scattered = scatter_prod_along_axes(&a, &indices, &updates, &axes, s).unwrap();
}

#[test]
fn test_scatter_prod_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axis = 0;
    let _scattered = scatter_prod_along_axis(&a, &indices, &updates, axis, s).unwrap();
}

#[test]
fn test_scatter_max_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let _scattered = scatter_max_along_axes(&a, &indices, &updates, &axes, s).unwrap();
}

#[test]
fn test_scatter_max_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axis = 0;
    let _scattered = scatter_max_along_axis(&a, &indices, &updates, axis, s).unwrap();
}

#[test]
fn test_scatter_min_along_axes() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let index = array_from_slice_int32(&[0, 1, 2], &shape);
    let mut indices = CxxVector::new();
    push_array(indices.pin_mut(), index);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axes = cxx_vec![0];
    let _scattered = scatter_min_along_axes(&a, &indices, &updates, &axes, s).unwrap();
}

#[test]
fn test_scatter_min_along_axis() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let shape = cxx_vec![3];
    let indices = array_from_slice_int32(&[0, 1, 2], &shape);
    let shape = cxx_vec![3, 3, 3];
    let updates = ones(&shape, s).unwrap();
    let axis = 0;
    let _scattered = scatter_min_along_axis(&a, &indices, &updates, axis, s).unwrap();
}

#[test]
fn test_sqrt() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _sqrted = sqrt(&a, s).unwrap();
}

#[test]
fn test_rsqrt() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _rsqrted = rsqrt(&a, s).unwrap();
}

#[test]
fn test_softmax_along_axes() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axes = cxx_vec![0];
    let _softmaxed = softmax_along_axes(&a, &axes, s).unwrap();
}

#[test]
fn test_softmax() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _softmaxed = softmax(&a, s).unwrap();
}

#[test]
fn test_softmax_along_axis() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axis = 0;
    let _softmaxed = softmax_along_axis(&a, axis, s).unwrap();
}

#[test]
fn test_power() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let _powered = power(&a, &b, s).unwrap();
}

#[test]
fn test_cumsum() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axis = 0;
    let _cumsumed = cumsum(&a, axis, false, false, s).unwrap();
}

#[test]
fn test_cumprod() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axis = 0;
    let _cumproded = cumprod(&a, axis, false, false, s).unwrap();
}

#[test]
fn test_cummax() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axis = 0;
    let _cummaxed = cummax(&a, axis, false, false, s).unwrap();
}

#[test]
fn test_cummin() {
    let s = Default::default();
    let a = arange_f64(1.0, 6.0, 1.0, s).unwrap();
    let axis = 0;
    let _cummined = cummin(&a, axis, false, false, s).unwrap();
}

#[test]
fn test_conv1d() {
    let s = Default::default();
    let shape = cxx_vec![3, 3, 3];
    let input = ones(&shape, s).unwrap();
    let kernel = ones(&shape, s).unwrap();
    let _conv1d = conv1d(&input, &kernel, 1, 0, 1, 1, s).unwrap();
}

#[test]
fn test_conv2d() {
    let s = Default::default();
    let shape = cxx_vec![3, 3, 3, 3];
    let input = ones(&shape, s).unwrap();
    let kernel = ones(&shape, s).unwrap();

    let stride = [1, 1];
    let padding = [0, 0];
    let dilation = [1, 1];
    let groups = 1;

    let _conv2d = conv2d(&input, &kernel, &stride, &padding, &dilation, groups, s).unwrap();
}

#[test]
fn test_quantized_matmul() {
    let s = Default::default();
    let shape = cxx_vec![512, 512];

    // quantize matrices
    let x = ones(&shape, s).unwrap();
    let w = ones(&shape, s).unwrap();

    let group_size = 64;
    let bits = 4;

    let [quantized_w, scales_w, biases_w] = quantize(&w, group_size, bits, s).unwrap();

    let _quantized_matmul = quantized_matmul(&x, &quantized_w, &scales_w, &biases_w, false, group_size, bits, s).unwrap();
}

#[test]
fn test_quantize() {
    let s = Default::default();
    let shape = cxx_vec![512, 512];
    let dtype = dtype_float32();
    let w = ones_dtype(&shape, dtype, s).unwrap();
    let _quantized = quantize(&w, 64, 4, s).unwrap();
}

#[test]
fn test_dequantize() {
    let s = Default::default();
    let shape = cxx_vec![512, 512];
    let a = ones(&shape, s).unwrap();
    let group_size = 64;
    let bits = 4;

    let [quantized_a, scales_a, biases_a] = quantize(&a, group_size, bits, s).unwrap();

    let _dequantized = dequantize(&quantized_a, &scales_a, &biases_a, group_size, bits, s).unwrap();
}

#[test]
fn test_tensordot() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let b = ones(&shape, s).unwrap();
    let dims = 1;
    let _tensordotted = tensordot(&a, &b, dims, s).unwrap();
}

#[test]
fn test_tensordot_list_dims() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let b = ones(&shape, s).unwrap();
    let dims = [cxx_vec![0], cxx_vec![0]];
    let _tensordotted = tensordot_list_dims(&a, &b, &dims, s).unwrap();
}

#[test]
fn test_outer() {
    let s = Default::default();
    let a = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let _outered = outer(&a, &b, s).unwrap();
}

#[test]
fn test_inner() {
    let s = Default::default();
    let a = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let b = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let _innered = inner(&a, &b, s).unwrap();
}

#[test]
fn test_addmm() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let b = ones(&shape, s).unwrap();
    let c = ones(&shape, s).unwrap();
    let alpha = 1.0;
    let beta = 1.0;
    let _addmmed = addmm(a, b, c, &alpha, &beta, s).unwrap();
}

#[test]
fn test_diagonal() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let offset = 0;
    let axis1 = 0;
    let axis2 = 1;
    let _diagonaled = diagonal(&a, offset, axis1, axis2, s).unwrap();
}

#[test]
fn test_diag() {
    let s = Default::default();
    let a = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let k = 0;
    let _diaged = diag(&a, k, s).unwrap();
}

#[test]
fn test_depends() {
    let s = Default::default();
    let mut inputs = CxxVector::new();
    let arr = arange_f64(1.0, 5.0, 1.0, s).unwrap();
    push_array(inputs.pin_mut(), arr);

    let mut dependencies = CxxVector::new();
    let arr = arange_f64(1.0, 5.0, 1.0, s).unwrap();
    push_array(dependencies.pin_mut(), arr);

    let _depended = depends(&inputs, &dependencies).unwrap();
}

#[test]
fn test_atleast_1d() {
    let s = Default::default();
    let a = arange_f64(1.0, 4.0, 1.0, s).unwrap();
    let _atleast_1d = atleast_1d(&a, s).unwrap();
}

#[test]
fn test_atleast_2d() {
    let s = Default::default();
    let shape = cxx_vec![3, 3];
    let a = ones(&shape, s).unwrap();
    let _atleast_2d = atleast_2d(&a, s).unwrap();
}

#[test]
fn test_atleast_3d() {
    let s = Default::default();
    let shape = cxx_vec![3, 3, 3];
    let a = ones(&shape, s).unwrap();
    let _atleast_3d = atleast_3d(&a, s).unwrap();
}