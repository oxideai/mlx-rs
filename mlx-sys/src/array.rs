
#[cxx::bridge]
mod ffi {
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

        // TODO: create array from vec and iterator

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
        fn eval(self: Pin<&mut array>, retain_graph: bool);
        
        #[namespace = "mlx_cxx"]
        fn array_item_bool(arr: &array, retain_graph: bool) -> bool;

        #[namespace = "mlx_cxx"]
        fn array_item_uint8(arr: &array, retain_graph: bool) -> u8;

        #[namespace = "mlx_cxx"]
        fn array_item_uint16(arr: &array, retain_graph: bool) -> u16;

        #[namespace = "mlx_cxx"]
        fn array_item_uint32(arr: &array, retain_graph: bool) -> u32;

        #[namespace = "mlx_cxx"]
        fn array_item_uint64(arr: &array, retain_graph: bool) -> u64;

        #[namespace = "mlx_cxx"]
        fn array_item_int8(arr: &array, retain_graph: bool) -> i8;

        #[namespace = "mlx_cxx"]
        fn array_item_int16(arr: &array, retain_graph: bool) -> i16;

        #[namespace = "mlx_cxx"]
        fn array_item_int32(arr: &array, retain_graph: bool) -> i32;

        #[namespace = "mlx_cxx"]
        fn array_item_int64(arr: &array, retain_graph: bool) -> i64;

        #[namespace = "mlx_cxx"]
        fn array_item_float16(arr: &array, retain_graph: bool) -> float16_t;
        
        #[namespace = "mlx_cxx"]
        fn array_item_bfloat16(arr: &array, retain_graph: bool) -> bfloat16_t;

        #[namespace = "mlx_cxx"]
        fn array_item_float32(arr: &array, retain_graph: bool) -> f32;

        #[namespace = "mlx_cxx"]
        fn array_item_complex64(arr: &array, retain_graph: bool) -> complex64_t;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_new_bool() {
        let array = ffi::array_new_bool(true);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::bool_));
    }

    #[test]
    fn test_array_new_i8() {
        let array = ffi::array_new_i8(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int8));
    }

    #[test]
    fn test_array_new_i16() {
        let array = ffi::array_new_i16(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int16));
    }

    #[test]
    fn test_array_new_i32() {
        let array = ffi::array_new_i32(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int32));
    }

    #[test]
    fn test_array_new_i64() {
        let array = ffi::array_new_i64(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::int64));
    }

    #[test]
    fn test_array_new_u8() {
        let array = ffi::array_new_u8(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint8));
    }

    #[test]
    fn test_array_new_u16() {
        let array = ffi::array_new_u16(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint16));
    }

    #[test]
    fn test_array_new_u32() {
        let array = ffi::array_new_u32(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint32));
    }

    #[test]
    fn test_array_new_u64() {
        let array = ffi::array_new_u64(1);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::uint64));
    }

    #[test]
    fn test_array_new_f32() {
        let array = ffi::array_new_f32(1.0);
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::float32));
    }

    #[test]
    fn test_array_new_f16() {
        let array = ffi::array_new_f16(ffi::float16_t { bits: 0x3c00 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::float16));
    }

    #[test]
    fn test_array_new_bf16() {
        let array = ffi::array_new_bf16(ffi::bfloat16_t { bits: 0x3c00 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(dtype.val, crate::dtype::ffi::Val::bfloat16));
    }

    #[test]
    fn test_array_new_c64() {
        let array = ffi::array_new_c64(ffi::complex64_t { re: 1.0, im: 1.0 });
        assert!(!array.is_null());
        assert_eq!(array.size(), 1);

        let dtype = array.dtype();
        assert!(matches!(
            dtype.val,
            crate::dtype::ffi::Val::complex64
        ));
    }
}