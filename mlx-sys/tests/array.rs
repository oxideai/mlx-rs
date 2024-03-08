use mlx_sys::{array::ffi::*, cxx_vec, dtype::ffi::Val, ops::ffi::add};

#[test]
fn test_array_new_bool() {
    let array = array_new_bool(false);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Bool));

    let item = array.item_bool().unwrap();
    assert_eq!(item, false);
}

#[test]
fn test_array_new_int8() {
    let array = array_new_int8(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int8));

    let item = array.item_int8().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_int16() {
    let array = array_new_int16(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int16));

    let item = array.item_int16().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_int32() {
    let array = array_new_int32(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int32));

    let item = array.item_int32().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_int64() {
    let array = array_new_int64(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int64));

    let item = array.item_int64().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_uint8() {
    let array = array_new_uint8(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint8));

    let item = array.item_uint8().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_uint16() {
    let array = array_new_uint16(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint16));

    let item = array.item_uint16().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_uint32() {
    let array = array_new_uint32(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint32));

    let item = array.item_uint32().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_uint64() {
    let array = array_new_uint64(1);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint64));

    let item = array.item_uint64().unwrap();
    assert_eq!(item, 1);
}

#[test]
fn test_array_new_float32() {
    let array = array_new_float32(1.0);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Float32));

    let item = array.item_float32().unwrap();
    assert_eq!(item, 1.0);
}

#[test]
fn test_array_new_float16() {
    let val = mlx_sys::types::float16::float16_t { bits: 0x00 };
    let array = array_new_float16(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Float16));

    let item = array.item_float16().unwrap();
    assert_eq!(item.bits, 0x00);
}

#[test]
fn test_array_new_bfloat16() {
    let val = mlx_sys::types::bfloat16::bfloat16_t { bits: 0x00 };
    let array = array_new_bfloat16(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Bfloat16));

    let item = array.item_bfloat16().unwrap();
    assert_eq!(item.bits, 0x00);
}

#[test]
fn test_array_new_complex64() {
    let val = mlx_sys::types::complex64::complex64_t { re: 0.0, im: 0.0 };
    let array = array_new_complex64(val);
    assert!(!array.is_null());
    assert_eq!(array.size(), 1);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Complex64));

    let item = array.item_complex64().unwrap();
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
    assert!(matches!(dtype.val, Val::Bool));
}

#[test]
fn test_array_from_slice_i8() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int8(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int8));
}

#[test]
fn test_array_from_slice_i16() {
    let data = [1, 2];
    let array = array_from_slice_int16(&data[..], &cxx_vec![2]);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int16));
}

#[test]
fn test_array_from_slice_i32() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int32));
}

#[test]
fn test_array_from_slice_i64() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_int64(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Int64));
}

#[test]
fn test_array_from_slice_u8() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint8(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint8));
}

#[test]
fn test_array_from_slice_u16() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint16(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint16));
}

#[test]
fn test_array_from_slice_u32() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint32));
}

#[test]
fn test_array_from_slice_u64() {
    let shape = cxx_vec![2];
    let data = [1, 2];
    let array = array_from_slice_uint64(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Uint64));
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
    assert!(matches!(dtype.val, Val::Float16));
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
    assert!(matches!(dtype.val, Val::Bfloat16));
}

#[test]
fn test_array_from_slice_f32() {
    let shape = cxx_vec![2];
    let data = [1.0, 2.0];
    let array = array_from_slice_float32(&data[..], &shape);
    assert!(!array.is_null());
    assert_eq!(array.size(), 2);

    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Float32));
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
    assert!(matches!(dtype.val, Val::Complex64));
}

#[test]
fn test_array_itemsize() {
    let array = array_new_bool(false);
    assert_eq!(array.itemsize(), 1);
}

#[test]
fn test_array_size() {
    let array = array_new_bool(false);
    assert_eq!(array.size(), 1);
}

#[test]
fn test_array_nbytes() {
    let array = array_new_bool(false);
    assert_eq!(array.nbytes(), 1);
}

#[test]
fn test_array_ndim() {
    let array = array_new_bool(false);
    let _ndim = array.ndim(); // We are just checking linking here
}

#[test]
fn test_array_shape() {
    let array = array_new_bool(false);
    let _shape = array.shape(); // We are just checking linking here
}

#[test]
fn test_array_strides() {
    let array = array_new_bool(false);
    let _strides = array.strides(); // We are just checking linking here
}

#[test]
fn test_array_dtype() {
    let array = array_new_bool(false);
    let dtype = array.dtype();
    assert!(matches!(dtype.val, Val::Bool));
}

#[test]
fn test_array_eval() {
    let a = array_new_float32(1.0);
    let b = array_new_float32(2.0);
    let mut c = add(&a, &b, Default::default()).unwrap();
    c.pin_mut().eval().unwrap();
    assert_eq!(c.item_float32().unwrap(), 3.0);
}

#[test]
fn test_array_id() {
    let array = array_new_bool(false);
    let _id = array.id(); // We are just checking linking here
}

#[test]
fn test_array_primitive_id() {
    let array = array_new_bool(false);
    let _id = array.primitive_id(); // We are just checking linking here
}

#[test]
fn test_array_has_primitive() {
    let array = array_new_bool(true);
    let _has_primitive = array.has_primitive();
}

#[test]
fn test_array_inputs() {
    let array = array_new_bool(true);
    let _inputs = array.inputs();
}

#[test]
fn test_array_siblings() {
    let array = array_new_bool(true);
    let _siblings = array.siblings();
}

#[test]
fn test_array_set_siblings() {
    let mut array = array_new_bool(true);
    let siblings = cxx::CxxVector::new();
    set_array_siblings(array.pin_mut(), siblings, 0);
}

#[test]
fn test_array_outputs() {
    let mut array = array_new_bool(true);
    let _outputs = array_outputs(array.pin_mut());
}

#[test]
fn test_array_detach() {
    let mut array = array_new_bool(true);
    array.pin_mut().detach();
}

#[test]
fn test_array_data_size() {
    let array = array_new_bool(true);
    let _data_size = array.data_size();
}

#[test]
fn test_array_data_mut_bool() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_bool(&[true, false], &shape);
    let ptr = array.pin_mut().data_mut_bool();
    unsafe {
        assert_eq!(*ptr.offset(0) as bool, true);
        assert_eq!(*ptr.offset(1) as bool, false);
    }
}

#[test]
fn test_array_data_mut_uint8() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_uint8(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_uint8();
    unsafe {
        assert_eq!(*ptr.offset(0) as u8, 1);
        assert_eq!(*ptr.offset(1) as u8, 2);
    }
}

#[test]
fn test_array_data_mut_uint16() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_uint16(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_uint16();
    unsafe {
        assert_eq!(*ptr.offset(0) as u16, 1);
        assert_eq!(*ptr.offset(1) as u16, 2);
    }
}

#[test]
fn test_array_data_mut_uint32() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_uint32(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_uint32();
    unsafe {
        assert_eq!(*ptr.offset(0) as u32, 1);
        assert_eq!(*ptr.offset(1) as u32, 2);
    }
}

#[test]
fn test_array_data_mut_uint64() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_uint64(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_uint64();
    unsafe {
        assert_eq!(*ptr.offset(0) as u64, 1);
        assert_eq!(*ptr.offset(1) as u64, 2);
    }
}

#[test]
fn test_array_data_mut_int8() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_int8(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_int8();
    unsafe {
        assert_eq!(*ptr.offset(0) as i8, 1);
        assert_eq!(*ptr.offset(1) as i8, 2);
    }
}

#[test]
fn test_array_data_mut_int16() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_int16(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_int16();
    unsafe {
        assert_eq!(*ptr.offset(0) as i16, 1);
        assert_eq!(*ptr.offset(1) as i16, 2);
    }
}

#[test]
fn test_array_data_mut_int32() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_int32(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_int32();
    unsafe {
        assert_eq!(*ptr.offset(0) as i32, 1);
        assert_eq!(*ptr.offset(1) as i32, 2);
    }
}

#[test]
fn test_array_data_mut_int64() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_int64(&[1, 2], &shape);
    let ptr = array.pin_mut().data_mut_int64();
    unsafe {
        assert_eq!(*ptr.offset(0) as i64, 1);
        assert_eq!(*ptr.offset(1) as i64, 2);
    }
}

#[test]
fn test_array_data_mut_float16() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_float16(
        &[float16_t { bits: 0x3c00 }, float16_t { bits: 0x3c00 }],
        &shape,
    );
    let ptr = array.pin_mut().data_mut_float16();
    unsafe {
        assert_eq!((*ptr.offset(0)).bits, 0x3c00);
        assert_eq!((*ptr.offset(1)).bits, 0x3c00);
    }
}

#[test]
fn test_array_data_mut_bfloat16() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_bfloat16(
        &[bfloat16_t { bits: 0x3c00 }, bfloat16_t { bits: 0x3c00 }],
        &shape,
    );
    let ptr = array.pin_mut().data_mut_bfloat16();
    unsafe {
        assert_eq!((*ptr.offset(0)).bits, 0x3c00);
        assert_eq!((*ptr.offset(1)).bits, 0x3c00);
    }
}

#[test]
fn test_array_data_mut_float32() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_float32(&[1.0, 2.0], &shape);
    let ptr = array.pin_mut().data_mut_float32();
    unsafe {
        assert_eq!(*ptr.offset(0) as f32, 1.0);
        assert_eq!(*ptr.offset(1) as f32, 2.0);
    }
}

#[test]
fn test_array_data_mut_complex64() {
    let shape = cxx_vec![2];
    let mut array = array_from_slice_complex64(
        &[
            complex64_t { re: 1.0, im: 1.0 },
            complex64_t { re: 1.0, im: 1.0 },
        ],
        &shape,
    );
    let ptr = array.pin_mut().data_mut_complex64();
    unsafe {
        assert_eq!((*ptr.offset(0)).re, 1.0);
        assert_eq!((*ptr.offset(0)).im, 1.0);
        assert_eq!((*ptr.offset(1)).re, 1.0);
        assert_eq!((*ptr.offset(1)).im, 1.0);
    }
}

#[test]
fn test_array_data_bool() {
    let shape = cxx_vec![2];
    let array = array_from_slice_bool(&[true, false], &shape);
    let ptr = array.data_bool();
    unsafe {
        assert_eq!(*ptr.offset(0) as bool, true);
        assert_eq!(*ptr.offset(1) as bool, false);
    }
}

#[test]
fn test_array_data_uint8() {
    let shape = cxx_vec![2];
    let array = array_from_slice_uint8(&[1, 2], &shape);
    let ptr = array.data_uint8();
    unsafe {
        assert_eq!(*ptr.offset(0) as u8, 1);
        assert_eq!(*ptr.offset(1) as u8, 2);
    }
}

#[test]
fn test_array_data_uint16() {
    let shape = cxx_vec![2];
    let array = array_from_slice_uint16(&[1, 2], &shape);
    let ptr = array.data_uint16();
    unsafe {
        assert_eq!(*ptr.offset(0) as u16, 1);
        assert_eq!(*ptr.offset(1) as u16, 2);
    }
}

#[test]
fn test_array_data_uint32() {
    let shape = cxx_vec![2];
    let array = array_from_slice_uint32(&[1, 2], &shape);
    let ptr = array.data_uint32();
    unsafe {
        assert_eq!(*ptr.offset(0) as u32, 1);
        assert_eq!(*ptr.offset(1) as u32, 2);
    }
}

#[test]
fn test_array_data_uint64() {
    let shape = cxx_vec![2];
    let array = array_from_slice_uint64(&[1, 2], &shape);
    let ptr = array.data_uint64();
    unsafe {
        assert_eq!(*ptr.offset(0) as u64, 1);
        assert_eq!(*ptr.offset(1) as u64, 2);
    }
}

#[test]
fn test_array_data_int8() {
    let shape = cxx_vec![2];
    let array = array_from_slice_int8(&[1, 2], &shape);
    let ptr = array.data_int8();
    unsafe {
        assert_eq!(*ptr.offset(0) as i8, 1);
        assert_eq!(*ptr.offset(1) as i8, 2);
    }
}

#[test]
fn test_array_data_int16() {
    let shape = cxx_vec![2];
    let array = array_from_slice_int16(&[1, 2], &shape);
    let ptr = array.data_int16();
    unsafe {
        assert_eq!(*ptr.offset(0) as i16, 1);
        assert_eq!(*ptr.offset(1) as i16, 2);
    }
}

#[test]
fn test_array_data_int32() {
    let shape = cxx_vec![2];
    let array = array_from_slice_int32(&[1, 2], &shape);
    let ptr = array.data_int32();
    unsafe {
        assert_eq!(*ptr.offset(0) as i32, 1);
        assert_eq!(*ptr.offset(1) as i32, 2);
    }
}

#[test]
fn test_array_data_int64() {
    let shape = cxx_vec![2];
    let array = array_from_slice_int64(&[1, 2], &shape);
    let ptr = array.data_int64();
    unsafe {
        assert_eq!(*ptr.offset(0) as i64, 1);
        assert_eq!(*ptr.offset(1) as i64, 2);
    }
}

#[test]
fn test_array_data_float16() {
    let shape = cxx_vec![2];
    let array = array_from_slice_float16(
        &[float16_t { bits: 0x3c00 }, float16_t { bits: 0x3c00 }],
        &shape,
    );
    let ptr = array.data_float16();
    unsafe {
        assert_eq!((*ptr.offset(0)).bits, 0x3c00);
        assert_eq!((*ptr.offset(1)).bits, 0x3c00);
    }
}

#[test]
fn test_array_data_bfloat16() {
    let shape = cxx_vec![2];
    let array = array_from_slice_bfloat16(
        &[bfloat16_t { bits: 0x3c00 }, bfloat16_t { bits: 0x3c00 }],
        &shape,
    );
    let ptr = array.data_bfloat16();
    unsafe {
        assert_eq!((*ptr.offset(0)).bits, 0x3c00);
        assert_eq!((*ptr.offset(1)).bits, 0x3c00);
    }
}

#[test]
fn test_array_data_float32() {
    let shape = cxx_vec![2];
    let array = array_from_slice_float32(&[1.0, 2.0], &shape);
    let ptr = array.data_float32();
    unsafe {
        assert_eq!(*ptr.offset(0) as f32, 1.0);
        assert_eq!(*ptr.offset(1) as f32, 2.0);
    }
}

#[test]
fn test_array_data_complex64() {
    let shape = cxx_vec![2];
    let array = array_from_slice_complex64(
        &[
            complex64_t { re: 1.0, im: 1.0 },
            complex64_t { re: 1.0, im: 1.0 },
        ],
        &shape,
    );
    let ptr = array.data_complex64();
    unsafe {
        assert_eq!((*ptr.offset(0)).re, 1.0);
        assert_eq!((*ptr.offset(0)).im, 1.0);
        assert_eq!((*ptr.offset(1)).re, 1.0);
        assert_eq!((*ptr.offset(1)).im, 1.0);
    }
}

#[test]
fn test_array_is_evaled() {
    let array = array_new_bool(true);
    let _is_evaled = array.is_evaled();
}

#[test]
fn test_array_set_tracer() {
    let mut array = array_new_bool(true);
    array.pin_mut().set_tracer(true);
}

#[test]
fn test_array_is_tracer() {
    let array = array_new_bool(true);
    let _is_tracer = array.is_tracer();
}

#[test]
fn test_array_overwrite_descriptor() {
    let mut a = array_new_bool(true);
    let b = array_new_bool(true);
    a.pin_mut().overwrite_descriptor(&b);
}
