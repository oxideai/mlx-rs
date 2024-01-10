use cxx::{ExternType, type_id};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct complex64_t {
    pub re: f32,
    pub im: f32,
}

unsafe impl ExternType for complex64_t {
    type Id = type_id!("mlx::core::complex64_t");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge(namespace = "mlx::core")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/types/complex.h");

        type complex64_t = crate::types::complex64::complex64_t;
    }
}