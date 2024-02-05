use mlx_sys::{array::ffi::*, cxx_vec};

#[test]
fn test_array_new_bool() {
    let mut array = array_new_bool(false);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::bool_));

    let item = array.pin_mut().item_bool().unwrap();
    assert_eq!(item, false);
}

#[test]
fn test_array_new_i8() {
    let mut array = array_new_i8(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int8));

    let item = array.pin_mut().item_int8().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_i16() {
    let mut array = array_new_i16(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int16));

    let item = array.pin_mut().item_int16().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_i32() {
    let mut array = array_new_i32(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int32));

    let item = array.pin_mut().item_int32().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_i64() {
    let mut array = array_new_i64(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int64));

    let item = array.pin_mut().item_int64().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_u8() {
    let mut array = array_new_u8(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint8));

    let item = array.pin_mut().item_uint8().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_u16() {
    let mut array = array_new_u16(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint16));

    let item = array.pin_mut().item_uint16().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_u32() {
    let mut array = array_new_u32(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint32));

    let item = array.pin_mut().item_uint32().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_u64() {
    let mut array = array_new_u64(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint64));

    let item = array.pin_mut().item_uint64().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_f32() {
    let mut array = array_new_f32(1.0);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::float32));

    let item = array.pin_mut().item_float32().unwrap();
    assert_eq!(item, 1.0);
}

#[test]
fn test_array_new_f16() {
    let val = mlx_sys::types::float16::float16_t { bits: 0x00 };
    let mut array = array_new_f16(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::float16));

    let item = array.pin_mut().item_float16().unwrap();
    assert_eq!(item.bits, 0x00);
}

#[test]
fn test_array_new_bf16() {
    let val = mlx_sys::types::bfloat16::bfloat16_t { bits: 0x00 };
    let mut array = array_new_bf16(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::bfloat16));

    let item = array.pin_mut().item_bfloat16().unwrap();
    assert_eq!(item.bits, 0x00);
}

#[test]
fn test_array_new_c64() {
    let val = mlx_sys::types::complex64::complex64_t { re: 0.0, im: 0.0 };
    let mut array = array_new_c64(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::complex64));

    let item = array.pin_mut().item_complex64().unwrap();
    assert_eq!(item.re, 0.0);
    assert_eq!(item.im, 0.0);
}

#[test]
fn test_array_from_slice_bool() {
    let shape = cxx_vec![2];
    let data = [true, false];
    let array = array_from_slice_bool(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::bool_));
}

#[test]
fn test_array_from_slice_i8() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int8(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int8));
}

#[test]
fn test_array_from_slice_i16() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int16(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int16));
}

#[test]
fn test_array_from_slice_i32() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int32));
}

#[test]
fn test_array_from_slice_i64() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int64(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::int64));
}

#[test]
fn test_array_from_slice_u8() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint8(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint8));
}

#[test]
fn test_array_from_slice_u16() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint16(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint16));
}

#[test]
fn test_array_from_slice_u32() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint32));
}

#[test]
fn test_array_from_slice_u64() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint64(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::uint64));
}

#[test]
fn test_array_from_slice_f16() {
    let shape = cxx_vec![2];
    let data = [
        mlx_sys::types::float16::float16_t { bits: 0x00 },
        mlx_sys::types::float16::float16_t { bits: 0x00 },
    ];
    let array = array_from_slice_float16(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::float16));
}

#[test]
fn test_array_from_slice_bf16() {
    let shape = cxx_vec![2];
    let data = [
        mlx_sys::types::bfloat16::bfloat16_t { bits: 0x00 },
        mlx_sys::types::bfloat16::bfloat16_t { bits: 0x00 },
    ];
    let array = array_from_slice_bfloat16(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::bfloat16));
}

#[test]
fn test_array_from_slice_f32() {
    let shape = cxx_vec![2];
    let data = [1.0, 2.0];
    let array = array_from_slice_float32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::float32));
}

#[test]
fn test_array_from_slice_c64() {
    let shape = cxx_vec![2];
    let data = [
        mlx_sys::types::complex64::complex64_t { re: 0.0, im: 0.0 },
        mlx_sys::types::complex64::complex64_t { re: 0.0, im: 0.0 },
    ];
    let array = array_from_slice_complex64(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, mlx_sys::dtype::ffi::Val::complex64));
}
