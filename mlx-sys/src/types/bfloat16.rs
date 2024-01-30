use cxx::{type_id, ExternType};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct bfloat16_t {
    pub bits: u16,
}

unsafe impl ExternType for bfloat16_t {
    type Id = type_id!("mlx::core::bfloat16_t");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge(namespace = "mlx::core")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/types/half_types.h");

        type bfloat16_t = crate::types::bfloat16::bfloat16_t;
    }
}

#[cfg(feature = "half")]
impl From<half::bf16> for bfloat16_t {
    fn from(value: half::bf16) -> Self {
        ffi::bfloat16_t {
            bits: value.to_bits(),
        }
    }
}

#[cfg(feature = "half")]
impl From<bfloat16_t> for half::bf16 {
    fn from(value: ffi::bfloat16_t) -> Self {
        half::bf16::from_bits(value.bits)
    }
}
