#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    pub struct bfloat16_t {
        pub bits: u16, // Wrapping half::bf16 is not supported
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        fn bf16_to_bits(value: bfloat16_t) -> u16;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");

        #[namespace = "mlx_cxx"]
        fn test_bf16_to_bits(value: bfloat16_t) -> u16;
    }
}

fn bf16_to_bits(value: ffi::bfloat16_t) -> u16 {
    value.bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_bits() {
        let value = ffi::bfloat16_t { bits: 0x3c00 };
        let bits = ffi::test_bf16_to_bits(value);
        assert_eq!(bits, 0x3c00);
    }
}