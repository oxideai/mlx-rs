use crate::array::wrapper::Array;
use crate::sealed::Sealed;
use num_complex::{Complex, Complex32};

/// Array element type
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, num_enum::IntoPrimitive, num_enum::TryFromPrimitive,
)]
#[repr(u32)]
pub enum Kind {
    Bool = mlx_sys::mlx_array_dtype__MLX_BOOL,
    Uint8 = mlx_sys::mlx_array_dtype__MLX_UINT8,
    Uint16 = mlx_sys::mlx_array_dtype__MLX_UINT16,
    Uint32 = mlx_sys::mlx_array_dtype__MLX_UINT32,
    Uint64 = mlx_sys::mlx_array_dtype__MLX_UINT64,
    Int8 = mlx_sys::mlx_array_dtype__MLX_INT8,
    Int16 = mlx_sys::mlx_array_dtype__MLX_INT16,
    Int32 = mlx_sys::mlx_array_dtype__MLX_INT32,
    Int64 = mlx_sys::mlx_array_dtype__MLX_INT64,
    Float16 = mlx_sys::mlx_array_dtype__MLX_FLOAT16,
    Float32 = mlx_sys::mlx_array_dtype__MLX_FLOAT32,
    Bfloat16 = mlx_sys::mlx_array_dtype__MLX_BFLOAT16,
    Complex64 = mlx_sys::mlx_array_dtype__MLX_COMPLEX64,
}

/// Kinds for tensor elements
///
/// # Safety
/// The specified Kind must be for a type that has the same length as Self.
pub unsafe trait Element: Clone {
    const KIND: Kind;
    const ZERO: Self;

    fn array_item(array: &Array) -> Self;
    fn array_data(array: &Array) -> *const Self;
}

impl Sealed for bool {}
unsafe impl Element for bool {
    const KIND: Kind = Kind::Bool;
    const ZERO: Self = false;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_bool(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_bool(array.c_array) }
    }
}

impl Sealed for u8 {}
unsafe impl Element for u8 {
    const KIND: Kind = Kind::Uint8;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_uint8(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_uint8(array.c_array) }
    }
}

impl Sealed for u16 {}
unsafe impl Element for u16 {
    const KIND: Kind = Kind::Uint16;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_uint16(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_uint16(array.c_array) }
    }
}

impl Sealed for u32 {}
unsafe impl Element for u32 {
    const KIND: Kind = Kind::Uint32;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_uint32(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_uint32(array.c_array) }
    }
}

impl Sealed for u64 {}
unsafe impl Element for u64 {
    const KIND: Kind = Kind::Uint64;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_uint64(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_uint64(array.c_array) }
    }
}

impl Sealed for i8 {}
unsafe impl Element for i8 {
    const KIND: Kind = Kind::Int8;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_int8(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_int8(array.c_array) }
    }
}

impl Sealed for i16 {}
unsafe impl Element for i16 {
    const KIND: Kind = Kind::Int16;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_int16(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_int16(array.c_array) }
    }
}

impl Sealed for i32 {}
unsafe impl Element for i32 {
    const KIND: Kind = Kind::Int32;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_int32(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_int32(array.c_array) }
    }
}

impl Sealed for i64 {}
unsafe impl Element for i64 {
    const KIND: Kind = Kind::Int64;
    const ZERO: Self = 0;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_int64(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_int64(array.c_array) }
    }
}

impl Sealed for f32 {}
unsafe impl Element for f32 {
    const KIND: Kind = Kind::Float32;
    const ZERO: Self = 0.;

    fn array_item(array: &Array) -> Self {
        unsafe { mlx_sys::mlx_array_item_float32(array.c_array) }
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_float32(array.c_array) }
    }
}

impl Sealed for Complex32 {}
unsafe impl Element for Complex32 {
    const KIND: Kind = Kind::Complex64;
    const ZERO: Self = Complex::new(0., 0.);

    fn array_item(array: &Array) -> Self {
        bindgen_complex_to_complex(unsafe { mlx_sys::mlx_array_item_complex64(array.c_array) })
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_complex64(array.c_array) as *const Self }
    }
}

impl Sealed for half::f16 {}
unsafe impl Element for half::f16 {
    const KIND: Kind = Kind::Float16;
    const ZERO: Self = half::f16::ZERO;

    fn array_item(array: &Array) -> Self {
        Self::from_bits(unsafe { mlx_sys::mlx_array_item_float16(array.c_array).0 })
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_float16(array.c_array) as *const Self }
    }
}

impl Sealed for half::bf16 {}
unsafe impl Element for half::bf16 {
    const KIND: Kind = Kind::Bfloat16;
    const ZERO: Self = half::bf16::ZERO;

    fn array_item(array: &Array) -> Self {
        Self::from_bits(unsafe { mlx_sys::mlx_array_item_bfloat16(array.c_array) })
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_bfloat16(array.c_array) as *const Self }
    }
}

#[inline]
fn bindgen_complex_to_complex<T>(item: mlx_sys::__BindgenComplex<T>) -> Complex<T> {
    Complex {
        re: item.re,
        im: item.im,
    }
}
