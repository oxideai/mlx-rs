use crate::error::Result;
use crate::sealed::Sealed;
use crate::{complex64, Array, Dtype};
use half::{bf16, f16};
use mlx_sys::{__BindgenComplex, bfloat16_t, float16_t};

/// A marker trait for array elements.
pub trait ArrayElement: Sized + Sealed {
    const DTYPE: Dtype;

    fn array_item(array: &Array) -> Result<Self>;

    fn array_data(array: &Array) -> *const Self;
}

macro_rules! impl_array_element {
    ($type:ty, $dtype:expr, $ctype:ident) => {
        paste::paste! {
            impl Sealed for $type {}
            impl ArrayElement for $type {
                const DTYPE: Dtype = $dtype;

                fn array_item(array: &Array) -> Result<Self> {
                    let mut val = <$type as Default>::default();
                    check_status!{
                        unsafe { mlx_sys::[<mlx_array_item_ $ctype >](&mut val as *mut _, array.c_array) },
                        {}
                    }
                    Ok(val)
                }

                fn array_data(array: &Array) -> *const Self {
                    unsafe { mlx_sys::[<mlx_array_data_ $ctype >](array.c_array) }
                }

            }
        }
    };
}

impl_array_element!(bool, Dtype::Bool, bool);
impl_array_element!(u8, Dtype::Uint8, uint8);
impl_array_element!(u16, Dtype::Uint16, uint16);
impl_array_element!(u32, Dtype::Uint32, uint32);
impl_array_element!(u64, Dtype::Uint64, uint64);
impl_array_element!(i8, Dtype::Int8, int8);
impl_array_element!(i16, Dtype::Int16, int16);
impl_array_element!(i32, Dtype::Int32, int32);
impl_array_element!(i64, Dtype::Int64, int64);
impl_array_element!(f32, Dtype::Float32, float32);

impl Sealed for f16 {}
impl ArrayElement for f16 {
    const DTYPE: Dtype = Dtype::Float16;

    fn array_item(array: &Array) -> Result<Self> {
        let mut val = float16_t::default();
        check_status! {
            unsafe { mlx_sys::mlx_array_item_float16(&mut val as *mut _, array.c_array) },
            {}
        }
        Ok(f16::from_bits(val.0))
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_float16(array.c_array) as *const Self }
    }
}

impl Sealed for bf16 {}
impl ArrayElement for bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;

    fn array_item(array: &Array) -> Result<Self> {
        let mut bits = bfloat16_t::default();
        check_status! {
            unsafe { mlx_sys::mlx_array_item_bfloat16(&mut bits as *mut _, array.c_array) },
            {}
        }
        Ok(bf16::from_bits(bits))
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_bfloat16(array.c_array) as *const Self }
    }
}

impl Sealed for complex64 {}
impl ArrayElement for complex64 {
    const DTYPE: Dtype = Dtype::Complex64;

    fn array_item(array: &Array) -> Result<Self> {
        let mut bindgen_complex64 = __BindgenComplex::<f32>::default();
        check_status! {
            unsafe { mlx_sys::mlx_array_item_complex64(&mut bindgen_complex64 as *mut _, array.c_array) },
            {}
        }
        Ok(Self {
            re: bindgen_complex64.re,
            im: bindgen_complex64.im,
        })
    }

    fn array_data(array: &Array) -> *const Self {
        // complex64 has the same memory layout as __BindgenComplex<f32>
        unsafe { mlx_sys::mlx_array_data_complex64(array.c_array) as *const Self }
    }
}
