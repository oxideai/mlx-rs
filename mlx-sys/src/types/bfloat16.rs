use cxx::{ExternType, type_id};

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

#[cfg(test)]
mod tests {
    use super::*;
}