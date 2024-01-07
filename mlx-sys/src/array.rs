
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/mlx_cxx.hpp");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx_cxx"]
        type f16 = crate::types::float16::ffi::f16;

        #[namespace = "mlx_cxx"]
        type bf16 = crate::types::bfloat16::ffi::bf16;

        #[namespace = "mlx_cxx"]
        type c64 = crate::types::complex64::ffi::c64;

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
        fn array_new_f16(value: f16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_new_bf16(value: bf16) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        fn array_new_c64(value: c64) -> UniquePtr<array>;

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

        // TODO: 
        // - Dtype

        // // extern function with generic parameters is not supported yet
        // #[namespace = "mlx_cxx"]
        // #[cxx_name = "new_unique"]
        // fn new_scalar_array<T>(value: T) -> UniquePtr<array>;
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
    }

    #[test]
    fn test_array_new_i8() {
        let array = ffi::array_new_i8(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_i16() {
        let array = ffi::array_new_i16(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_i32() {
        let array = ffi::array_new_i32(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_i64() {
        let array = ffi::array_new_i64(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_u8() {
        let array = ffi::array_new_u8(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_u16() {
        let array = ffi::array_new_u16(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_u32() {
        let array = ffi::array_new_u32(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_u64() {
        let array = ffi::array_new_u64(1);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_f32() {
        let array = ffi::array_new_f32(1.0);
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_f16() {
        let array = ffi::array_new_f16(ffi::f16 { bits: 0x3c00 });
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_bf16() {
        let array = ffi::array_new_bf16(ffi::bf16 { bits: 0x3c00 });
        assert!(!array.is_null());
    }

    #[test]
    fn test_array_new_c64() {
        let array = ffi::array_new_c64(ffi::c64 { re: 1.0, im: 1.0 });
        assert!(!array.is_null());
    }
}