use cxx::{ExternType, type_id};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct float16_t {
    pub bits: u16,
}

unsafe impl ExternType for float16_t {
    type Id = type_id!("mlx::core::float16_t");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge(namespace = "mlx::core")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("mlx/types/half_types.h");

        type float16_t = crate::types::float16::float16_t;
    }
}

#[cfg(feature = "half")]
impl From<half::f16> for float16_t {
    fn from(value: half::f16) -> Self {
        ffi::f16 { bits: value.to_bits() }
    }
}

#[cfg(feature = "half")]
impl From<float16_t> for half::f16 {
    fn from(value: ffi::f16) -> Self {
        half::f16::from_bits(value.bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}