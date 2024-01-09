use cxx::{ExternType, type_id};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct float16_t {
    pub bits: u16, // Wrapping half::f16 is not supported
}

unsafe impl ExternType for float16_t {
    type Id = type_id!("float16_t");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");

        type float16_t = super::float16_t;

        #[namespace = "mlx_cxx"]
        fn test_f16_to_bits(value: float16_t) -> u16;
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        fn f16_to_bits(value: float16_t) -> u16;
    }
}

fn f16_to_bits(value: ffi::float16_t) -> u16 {
    value.bits
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

    // TODO: remove later
    #[test]
    fn test_f16_to_bits() {
        let value = ffi::float16_t { bits: 0x3c00 };
        let bits = ffi::test_f16_to_bits(value);
        assert_eq!(bits, 0x3c00);
    }
}