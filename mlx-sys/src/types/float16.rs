#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    pub struct float16_t {
        pub bits: u16, // Wrapping half::f16 is not supported
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        fn f16_to_bits(value: float16_t) -> u16;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");

        #[namespace = "mlx_cxx"]
        fn test_f16_to_bits(value: float16_t) -> u16;
    }
}

fn f16_to_bits(value: ffi::float16_t) -> u16 {
    value.bits
}

#[cfg(feature = "half")]
impl From<half::f16> for ffi::float16_t {
    fn from(value: half::f16) -> Self {
        ffi::f16 { bits: value.to_bits() }
    }
}

#[cfg(feature = "half")]
impl From<ffi::float16_t> for half::f16 {
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