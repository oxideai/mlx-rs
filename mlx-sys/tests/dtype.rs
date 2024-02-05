use mlx_sys::dtype::ffi::*;

#[test]
fn test_dtype_new() {
    let dtype = dtype_new(Val::bool_, 1);
    assert!(matches!(dtype.val, Val::bool_));
}

#[test]
fn test_is_available() {
    let dtype = dtype_bool_();
    assert!(is_available(&dtype));
}

#[test]
fn test_dtype_bool_() {
    let dtype = dtype_bool_();
    assert!(matches!(dtype.val, Val::bool_));
}

#[test]
fn test_dtype_uint8() {
    let dtype = dtype_uint8();
    assert!(matches!(dtype.val, Val::uint8));
}

#[test]
fn test_dtype_uint16() {
    let dtype = dtype_uint16();
    assert!(matches!(dtype.val, Val::uint16));
}

#[test]
fn test_dtype_uint32() {
    let dtype = dtype_uint32();
    assert!(matches!(dtype.val, Val::uint32));
}

#[test]
fn test_dtype_uint64() {
    let dtype = dtype_uint64();
    assert!(matches!(dtype.val, Val::uint64));
}

#[test]
fn test_dtype_int8() {
    let dtype = dtype_int8();
    assert!(matches!(dtype.val, Val::int8));
}

#[test]
fn test_dtype_int16() {
    let dtype = dtype_int16();
    assert!(matches!(dtype.val, Val::int16));
}

#[test]
fn test_dtype_int32() {
    let dtype = dtype_int32();
    assert!(matches!(dtype.val, Val::int32));
}

#[test]
fn test_dtype_int64() {
    let dtype = dtype_int64();
    assert!(matches!(dtype.val, Val::int64));
}

#[test]
fn test_dtype_float16() {
    let dtype = dtype_float16();
    assert!(matches!(dtype.val, Val::float16));
}

#[test]
fn test_dtype_float32() {
    let dtype = dtype_float32();
    assert!(matches!(dtype.val, Val::float32));
}

#[test]
fn test_dtype_bfloat16() {
    let dtype = dtype_bfloat16();
    assert!(matches!(dtype.val, Val::bfloat16));
}

#[test]
fn test_dtype_complex64() {
    let dtype = dtype_complex64();
    assert!(matches!(dtype.val, Val::complex64));
}

#[test]
fn test_promote_types() {
    let t1 = dtype_int32();
    let t2 = dtype_float32();
    let t3 = promote_types(&t1, &t2);
    assert!(matches!(t3.val, Val::float32));
}

#[test]
fn test_size_of() {
    let dtype = dtype_int32();
    assert_eq!(size_of(&dtype), 4);
}

#[test]
fn test_kindof() {
    let dtype = dtype_int32();
    assert!(matches!(kindof(&dtype), Kind::i));
}

#[test]
fn test_is_unsigned() {
    let dtype = dtype_uint32();
    assert!(is_unsigned(&dtype));
}

#[test]
fn test_is_floating_point() {
    let dtype = dtype_float32();
    assert!(is_floating_point(&dtype));
}

#[test]
fn test_is_complex() {
    let dtype = dtype_complex64();
    assert!(is_complex(&dtype));
}

#[test]
fn test_is_integral() {
    let dtype = dtype_int32();
    assert!(is_integral(&dtype));
}

// TODO: test dtype_to_array_protocol
// TODO: test dtype_from_array_protocol
