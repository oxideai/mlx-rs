
#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/mlx_cxx.hpp");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx::core"]
        type float16_t = crate::types::float16::float16_t;

        #[namespace = "mlx::core"]
        type bfloat16_t = crate::types::bfloat16::bfloat16_t;

        #[namespace = "mlx::core"]
        type complex64_t = crate::types::complex64::complex64_t;

        #[namespace = "mlx::core"]
        type array;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_bool(value: bool) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i8(value: i8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i16(value: i16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i32(value: i32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_i64(value: i64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u8(value: u8) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u16(value: u16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u32(value: u32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_u64(value: u64) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_f32(value: f32) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_f16(value: float16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_bf16(value: bfloat16_t) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "new_unique"]
        fn array_new_c64(value: complex64_t) -> UniquePtr<array>;

        #[namespace = "mlx::core"]
        fn itemsize(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn size(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn nbytes(self: &array) -> usize;

        #[namespace = "mlx::core"]
        fn ndim(self: &array) -> usize;

        #[namespace = "mlx::core"]
        #[rust_name = "shape"]
        fn shape(self: &array) -> &CxxVector<i32>;

        #[namespace = "mlx::core"]
        #[rust_name = "shape_of_dim"]
        fn shape(self: &array, dim: i32) -> i32;

        #[namespace = "mlx::core"]
        fn strides(self: &array) -> &CxxVector<usize>;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        fn dtype(self: &array) -> Dtype;

        #[namespace = "mlx::core"]
        fn eval(self: Pin<&mut array>);
        
        #[namespace = "mlx_cxx"]
        fn array_item_bool(arr: Pin<&mut array>) -> bool;

        #[namespace = "mlx_cxx"]
        fn array_item_uint8(arr: Pin<&mut array>) -> u8;

        #[namespace = "mlx_cxx"]
        fn array_item_uint16(arr: Pin<&mut array>) -> u16;

        #[namespace = "mlx_cxx"]
        fn array_item_uint32(arr: Pin<&mut array>) -> u32;

        #[namespace = "mlx_cxx"]
        fn array_item_uint64(arr: Pin<&mut array>) -> u64;

        #[namespace = "mlx_cxx"]
        fn array_item_int8(arr: Pin<&mut array>) -> i8;

        #[namespace = "mlx_cxx"]
        fn array_item_int16(arr: Pin<&mut array>) -> i16;

        #[namespace = "mlx_cxx"]
        fn array_item_int32(arr: Pin<&mut array>) -> i32;

        #[namespace = "mlx_cxx"]
        fn array_item_int64(arr: Pin<&mut array>) -> i64;

        #[namespace = "mlx_cxx"]
        fn array_item_float16(arr: Pin<&mut array>) -> float16_t;
        
        #[namespace = "mlx_cxx"]
        fn array_item_bfloat16(arr: Pin<&mut array>) -> bfloat16_t;

        #[namespace = "mlx_cxx"]
        fn array_item_float32(arr: Pin<&mut array>) -> f32;

        #[namespace = "mlx_cxx"]
        fn array_item_complex64(arr: Pin<&mut array>) -> complex64_t;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_bool(slice: &[bool], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint8(slice: &[u8], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint16(slice: &[u16], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint32(slice: &[u32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_uint64(slice: &[u64], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int8(slice: &[i8], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int16(slice: &[i16], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int32(slice: &[i32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_int64(slice: &[i64], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_float16(slice: &[float16_t], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_bfloat16(slice: &[bfloat16_t], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_float32(slice: &[f32], shape: &CxxVector<i32>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_from_slice_complex64(slice: &[complex64_t], shape: &CxxVector<i32>) -> UniquePtr<array>;

        // TODO: how to get data from cxx to rust? The method `data()` tho is public but is intended
        // for use by the backend implementation
    }

    impl CxxVector<array> {} // Explicit instantiation
}


#[cfg(test)]
mod tests {
    use cxx::CxxVector;

    use crate::cxx_vec;

    use super::*;

    #[test]
    fn test_array_new_bool() {
        let mut array = ffi::array_new_bool(true);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));

        let item = ffi::array_item_bool(array.pin_mut());
        assert_eq!(item, true);
    }

    #[test]
    fn test_array_new_i8() {
        let mut array = ffi::array_new_i8(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int8));

        let item = ffi::array_item_int8(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_i16() {
        let mut array = ffi::array_new_i16(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int16));

        let item = ffi::array_item_int16(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_i32() {
        let mut array = ffi::array_new_i32(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int32));

        let item = ffi::array_item_int32(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_i64() {
        let mut array = ffi::array_new_i64(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int64));

        let item = ffi::array_item_int64(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_u8() {
        let mut array = ffi::array_new_u8(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint8));

        let item = ffi::array_item_uint8(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_u16() {
        let mut array = ffi::array_new_u16(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint16));

        let item = ffi::array_item_uint16(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_u32() {
        let mut array = ffi::array_new_u32(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint32));

        let item = ffi::array_item_uint32(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_u64() {
        let mut array = ffi::array_new_u64(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint64));

        let item = ffi::array_item_uint64(array.pin_mut());
        assert_eq!(item, 1);
    }

    #[test]
    fn test_array_new_f32() {
        let mut array = ffi::array_new_f32(1.0);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::float32));

        let item = ffi::array_item_float32(array.pin_mut());
        assert_eq!(item, 1.0);
    }

    #[test]
    fn test_array_new_f16() {
        let mut array = ffi::array_new_f16(ffi::float16_t { bits: 0x3c00 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::float16));

        let item = ffi::array_item_float16(array.pin_mut());
        assert_eq!(item.bits, 0x3c00);
    }

    #[test]
    fn test_array_new_bf16() {
        let mut array = ffi::array_new_bf16(ffi::bfloat16_t { bits: 0x3c00 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::bfloat16));

        let item = ffi::array_item_bfloat16(array.pin_mut());
        assert_eq!(item.bits, 0x3c00);
    }

    #[test]
    fn test_array_new_c64() {
        let mut array = ffi::array_new_c64(ffi::complex64_t { re: 1.0, im: 1.0 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::complex64
        ));

        let item = ffi::array_item_complex64(array.pin_mut());
        assert_eq!(item.re, 1.0);
        assert_eq!(item.im, 1.0);
    }

    #[test]
    fn test_array_from_slice_bool() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_bool(&[true, false], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));
    }

    #[test]
    fn test_array_from_slice_uint8() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_uint8(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint8));
    }

    #[test]
    fn test_array_from_slice_uint16() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_uint16(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint16));
    }

    #[test]
    fn test_array_from_slice_uint32() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_uint32(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint32));
    }

    #[test]
    fn test_array_from_slice_uint64() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_uint64(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint64));
    }

    #[test]
    fn test_array_from_slice_int8() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_int8(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int8));
    }

    #[test]
    fn test_array_from_slice_int16() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_int16(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int16));
    }

    #[test]
    fn test_array_from_slice_int32() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_int32(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int32));
    }

    #[test]
    fn test_array_from_slice_int64() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_int64(&[1, 2], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int64));
    }

    #[test]
    fn test_array_from_slice_float16() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_float16(&[ffi::float16_t { bits: 0x3c00 }, ffi::float16_t { bits: 0x3c00 }], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::float16
        ));
    }

    #[test]
    fn test_array_from_slice_bfloat16() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_bfloat16(&[ffi::bfloat16_t { bits: 0x3c00 }, ffi::bfloat16_t { bits: 0x3c00 }], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::bfloat16
        ));
    }

    #[test]
    fn test_array_from_slice_float32() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_float32(&[1.0, 2.0], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::float32
        ));
    }

    #[test]
    fn test_array_from_slice_complex64() {
        let shape = cxx_vec![2];
        let array = ffi::array_from_slice_complex64(&[ffi::complex64_t { re: 1.0, im: 1.0 }, ffi::complex64_t { re: 1.0, im: 1.0 }], &shape);
        assert!(!array.is_null());
        assert_eq!(array.size(), 2);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::complex64
        ));
    }
}