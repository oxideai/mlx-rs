use crate::array::ffi::{bfloat16_t, complex64_t, float16_t};

#[cxx::bridge]
pub mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DtypeVal"]
    #[repr(i32)]
    enum Val {
        #[cxx_name = "bool_"]
        Bool,

        #[cxx_name = "uint8"]
        Uint8,

        #[cxx_name = "uint16"]
        Uint16,

        #[cxx_name = "uint32"]
        Uint32,

        #[cxx_name = "uint64"]
        Uint64,

        #[cxx_name = "int8"]
        Int8,

        #[cxx_name = "int16"]
        Int16,

        #[cxx_name = "int32"]
        Int32,

        #[cxx_name = "int64"]
        Int64,

        #[cxx_name = "float16"]
        Float16,

        #[cxx_name = "float32"]
        Float32,

        #[cxx_name = "bfloat16"]
        Bfloat16,

        #[cxx_name = "complex64"]
        Complex64,
    }

    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DtypeKind"]
    #[repr(i32)]
    enum Kind {
        #[cxx_name = "b"]
        Bool,

        #[cxx_name = "u"]
        UnsignedInt,

        #[cxx_name = "i"]
        SignedInt,

        #[cxx_name = "f"]
        Float,

        #[cxx_name = "c"]
        Complex,

        #[cxx_name = "V"]
        Void,
    }

    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DtypeCategory"]
    #[repr(i32)]
    enum Category {
        // complexfloating,
        #[cxx_name = "complexfloating"]
        ComplexFloating,

        // floating,
        #[cxx_name = "floating"]
        Floating,
        
        // inexact,
        #[cxx_name = "inexact"]
        Inexact,
        
        // signedinteger,
        #[cxx_name = "signedinteger"]
        SignedInteger,

        // unsignedinteger,
        #[cxx_name = "unsignedinteger"]
        UnsignedInteger,

        // integer,
        #[cxx_name = "integer"]
        Integer,

        // number,
        #[cxx_name = "number"]
        Number,

        // generic
        #[cxx_name = "generic"]
        Generic,
    }

    #[derive(Clone, Copy)]
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

        #[namespace = "mlx_cxx"]
        #[cxx_name = "DtypeCategory"]
        type Category;
    }

    unsafe extern "C++" {
        include!("mlx/dtype.h");
        include!("mlx-cxx/dtype.hpp");
        include!("mlx-cxx/utils.hpp");

        #[namespace = "mlx::core"]
        type Dtype;

        #[namespace = "std"]
        #[cxx_name = "string_view"]
        type StringView<'a> = crate::utils::StringView<'a>;

        #[namespace = "mlx_cxx"]
        fn dtype_new(val: Val, size: u8) -> Dtype;

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

        #[namespace = "mlx_cxx"]
        fn dtype_category_complexfloating() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_floating() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_inexact() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_signedinteger() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_unsignedinteger() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_integer() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_number() -> Category;

        #[namespace = "mlx_cxx"]
        fn dtype_category_generic() -> Category;

        #[namespace = "mlx::core"]
        fn promote_types(t1: &Dtype, t2: &Dtype) -> Dtype;

        #[namespace = "mlx::core"]
        fn size_of(t: &Dtype) -> u8;

        #[namespace = "mlx::core"]
        fn kindof(t: &Dtype) -> Kind;

        #[namespace = "mlx_cxx"]
        fn dtype_to_array_protocol(t: &Dtype) -> UniquePtr<CxxString>;

        #[namespace = "mlx::core"]
        fn dtype_from_array_protocol(s: StringView) -> Result<Dtype>;

        #[namespace = "mlx::core"]
        fn issubdtype(a: &Dtype, b: &Dtype) -> bool;

        #[namespace = "mlx::core"]
        #[cxx_name = "issubdtype"]
        fn issubdtype_cat_dtype(a: &Category, b: &Dtype) -> bool;

        #[namespace = "mlx::core"]
        #[cxx_name = "issubdtype"]
        fn issubdtype_dtype_cat(a: &Dtype, b: &Category) -> bool;   

        #[namespace = "mlx::core"]
        #[cxx_name = "issubdtype"]
        fn issubdtype_cat_cat(a: &Category, b: &Category) -> bool;
    }
}

impl Default for ffi::Dtype {
    fn default() -> Self {
        ffi::dtype_float32()
    }
}

pub trait DataType {
    fn dtype() -> ffi::Dtype;

    fn dtype_of_val(&self) -> ffi::Dtype {
        Self::dtype()
    }
}

impl DataType for bool {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_bool_()
    }
}

impl DataType for u8 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_uint8()
    }
}

impl DataType for u16 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_uint16()
    }
}

impl DataType for u32 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_uint32()
    }
}

impl DataType for u64 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_uint64()
    }
}

impl DataType for i8 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_int8()
    }
}

impl DataType for i16 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_int16()
    }
}

impl DataType for i32 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_int32()
    }
}

impl DataType for i64 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_int64()
    }
}

impl DataType for float16_t {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_float16()
    }
}

impl DataType for f32 {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_float32()
    }
}

impl DataType for bfloat16_t {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_bfloat16()
    }
}

impl DataType for complex64_t {
    fn dtype() -> ffi::Dtype {
        ffi::dtype_complex64()
    }
}