#[cxx::bridge]
pub mod ffi {
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DtypeVal"]
    #[derive(Clone)]
    #[repr(i32)]
    enum Val {
        bool_,
        uint8,
        uint16,
        uint32,
        uint64,
        int8,
        int16,
        int32,
        int64,
        float16,
        float32,
        bfloat16,
        complex64,
    }

    #[namespace = "mlx_cxx"]
    #[cxx_name = "DtypeKind"]
    #[derive(Clone)]
    #[repr(i32)]
    enum Kind {
        b, /* bool */
        u, /* unsigned int */
        i, /* signed int */
        f, /* float */
        c, /* complex */
        V, /* void - used for brain float */
    }

    #[namespace = "mlx::core"]
    struct Dtype {
        val: Val,
        size: u8,
    }

    extern "C++" {
        include!("mlx-cxx/dtype.hpp");

        #[namespace = "mlx_cxx"]
        #[cxx_name = "DtypeVal"]
        type Val;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "DtypeKind"]
        type Kind;

    }

    unsafe extern "C++" {
        include!("mlx/dtype.h");
        include!("mlx-cxx/dtype.hpp");

        #[namespace = "mlx::core"]
        type Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_new(val: Val, size: u8) -> Dtype;

        #[namespace = "mlx::core"]
        fn is_available(dtype: &Dtype) -> bool;

        #[namespace = "mlx_cxx"]
        fn dtype_bool_() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_uint8() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_uint16() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_uint32() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_uint64() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_int8() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_int16() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_int32() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_int64() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_float16() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_float32() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_bfloat16() -> Dtype;

        #[namespace = "mlx_cxx"]
        fn dtype_complex64() -> Dtype;

        #[namespace = "mlx::core"]
        fn promote_types(t1: &Dtype, t2: &Dtype) -> Dtype;

        #[namespace = "mlx::core"]
        fn size_of(t: &Dtype) -> u8;

        #[namespace = "mlx::core"]
        fn kindof(t: &Dtype) -> Kind;

        #[namespace = "mlx::core"]
        fn is_unsigned(t: &Dtype) -> bool;

        #[namespace = "mlx::core"]
        fn is_floating_point(t: &Dtype) -> bool;

        #[namespace = "mlx::core"]
        fn is_complex(t: &Dtype) -> bool;

        #[namespace = "mlx::core"]
        fn is_integral(t: &Dtype) -> bool;

        #[namespace = "mlx_cxx"]
        fn dtype_to_array_protocol(t: &Dtype) -> UniquePtr<CxxString>;

        #[namespace = "mlx::core"]
        fn dtype_from_array_protocol(s: &CxxString) -> Result<Dtype>;
    }
}

impl Default for ffi::Dtype {
    fn default() -> Self {
        ffi::dtype_float32()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_dtype_new() {
//         let dtype = ffi::dtype_new(ffi::Val::bool_, 1);
//         assert!(matches!(dtype.val, ffi::Val::bool_));
//     }

//     #[test]
//     fn test_is_available() {
//         let dtype = ffi::dtype_bool_();
//         assert!(ffi::is_available(&dtype));
//     }

//     #[test]
//     fn test_dtype_bool_() {
//         let dtype = ffi::dtype_bool_();
//         assert!(matches!(dtype.val, ffi::Val::bool_));
//     }

//     #[test]
//     fn test_dtype_uint8() {
//         let dtype = ffi::dtype_uint8();
//         assert!(matches!(dtype.val, ffi::Val::uint8));
//     }

//     #[test]
//     fn test_dtype_uint16() {
//         let dtype = ffi::dtype_uint16();
//         assert!(matches!(dtype.val, ffi::Val::uint16));
//     }

//     #[test]
//     fn test_dtype_uint32() {
//         let dtype = ffi::dtype_uint32();
//         assert!(matches!(dtype.val, ffi::Val::uint32));
//     }

//     #[test]
//     fn test_dtype_uint64() {
//         let dtype = ffi::dtype_uint64();
//         assert!(matches!(dtype.val, ffi::Val::uint64));
//     }

//     #[test]
//     fn test_dtype_int8() {
//         let dtype = ffi::dtype_int8();
//         assert!(matches!(dtype.val, ffi::Val::int8));
//     }

//     #[test]
//     fn test_dtype_int16() {
//         let dtype = ffi::dtype_int16();
//         assert!(matches!(dtype.val, ffi::Val::int16));
//     }

//     #[test]
//     fn test_dtype_int32() {
//         let dtype = ffi::dtype_int32();
//         assert!(matches!(dtype.val, ffi::Val::int32));
//     }

//     #[test]
//     fn test_dtype_int64() {
//         let dtype = ffi::dtype_int64();
//         assert!(matches!(dtype.val, ffi::Val::int64));
//     }

//     #[test]
//     fn test_dtype_float16() {
//         let dtype = ffi::dtype_float16();
//         assert!(matches!(dtype.val, ffi::Val::float16));
//     }

//     #[test]
//     fn test_dtype_float32() {
//         let dtype = ffi::dtype_float32();
//         assert!(matches!(dtype.val, ffi::Val::float32));
//     }

//     #[test]
//     fn test_dtype_bfloat16() {
//         let dtype = ffi::dtype_bfloat16();
//         assert!(matches!(dtype.val, ffi::Val::bfloat16));
//     }

//     #[test]
//     fn test_dtype_complex64() {
//         let dtype = ffi::dtype_complex64();
//         assert!(matches!(dtype.val, ffi::Val::complex64));
//     }

//     #[test]
//     fn test_promote_types() {
//         let t1 = ffi::dtype_int32();
//         let t2 = ffi::dtype_float32();
//         let t3 = ffi::promote_types(&t1, &t2);
//         assert!(matches!(t3.val, ffi::Val::float32));
//     }

//     #[test]
//     fn test_size_of() {
//         let dtype = ffi::dtype_int32();
//         assert_eq!(ffi::size_of(&dtype), 4);
//     }

//     #[test]
//     fn test_kindof() {
//         let dtype = ffi::dtype_int32();
//         assert!(matches!(ffi::kindof(&dtype), ffi::Kind::i));
//     }

//     #[test]
//     fn test_is_unsigned() {
//         let dtype = ffi::dtype_uint32();
//         assert!(ffi::is_unsigned(&dtype));
//     }

//     #[test]
//     fn test_is_floating_point() {
//         let dtype = ffi::dtype_float32();
//         assert!(ffi::is_floating_point(&dtype));
//     }

//     #[test]
//     fn test_is_complex() {
//         let dtype = ffi::dtype_complex64();
//         assert!(ffi::is_complex(&dtype));
//     }

//     #[test]
//     fn test_is_integral() {
//         let dtype = ffi::dtype_int32();
//         assert!(ffi::is_integral(&dtype));
//     }

//     // TODO: test dtype_to_array_protocol
//     // TODO: test dtype_from_array_protocol
// }
