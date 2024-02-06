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
