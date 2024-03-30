use mlx_sys::dtype::ffi::*;

#[test]
fn test_dtype_new() {
    let dtype = dtype_new(Val::Bool, 1);
    assert!(matches!(dtype.val, Val::Bool));
}

#[test]
fn test_dtype_bool_() {
    let dtype = dtype_bool_();
    assert!(matches!(dtype.val, Val::Bool));
}

#[test]
fn test_dtype_uint8() {
    let dtype = dtype_uint8();
    assert!(matches!(dtype.val, Val::Uint8));
}

#[test]
fn test_dtype_uint16() {
    let dtype = dtype_uint16();
    assert!(matches!(dtype.val, Val::Uint16));
}

#[test]
fn test_dtype_uint32() {
    let dtype = dtype_uint32();
    assert!(matches!(dtype.val, Val::Uint32));
}

#[test]
fn test_dtype_uint64() {
    let dtype = dtype_uint64();
    assert!(matches!(dtype.val, Val::Uint64));
}

#[test]
fn test_dtype_int8() {
    let dtype = dtype_int8();
    assert!(matches!(dtype.val, Val::Int8));
}

#[test]
fn test_dtype_int16() {
    let dtype = dtype_int16();
    assert!(matches!(dtype.val, Val::Int16));
}

#[test]
fn test_dtype_int32() {
    let dtype = dtype_int32();
    assert!(matches!(dtype.val, Val::Int32));
}

#[test]
fn test_dtype_int64() {
    let dtype = dtype_int64();
    assert!(matches!(dtype.val, Val::Int64));
}

#[test]
fn test_dtype_float16() {
    let dtype = dtype_float16();
    assert!(matches!(dtype.val, Val::Float16));
}

#[test]
fn test_dtype_float32() {
    let dtype = dtype_float32();
    assert!(matches!(dtype.val, Val::Float32));
}

#[test]
fn test_dtype_bfloat16() {
    let dtype = dtype_bfloat16();
    assert!(matches!(dtype.val, Val::Bfloat16));
}

#[test]
fn test_dtype_complex64() {
    let dtype = dtype_complex64();
    assert!(matches!(dtype.val, Val::Complex64));
}

#[test]
fn test_promote_types() {
    let t1 = dtype_int32();
    let t2 = dtype_float32();
    let t3 = promote_types(&t1, &t2);
    assert!(matches!(t3.val, Val::Float32));
}

#[test]
fn test_size_of() {
    let dtype = dtype_int32();
    assert_eq!(size_of(&dtype), 4);
}

#[test]
fn test_kindof() {
    let dtype = dtype_int32();
    assert!(matches!(kindof(&dtype), Kind::SignedInt));
}

// TODO: test dtype_to_array_protocol
// TODO: test dtype_from_array_protocol
