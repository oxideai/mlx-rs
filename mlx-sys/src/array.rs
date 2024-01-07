
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/mlx_cxx.hpp");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx_cxx"]
        type f16 = crate::types::float16::ffi::f16;

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

        // TODO: 
        // - float16
        // - bfloat16
        // - complex64

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
}