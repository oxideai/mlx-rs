#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    pub struct f16 {
        pub bits: u16, // Wrapping half::f16 is not supported
    }

    extern "Rust" {
        fn f16_to_bits(value: f16) -> u16;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");

        #[namespace = "mlx_cxx"]
        fn cxx_f16_to_bits(value: f16) -> u16;
    }

    // unsafe extern "C++" {
    //     include!("mlx/types/fp16.h");
    //     include!("mlx/types/half_types.h");

    //     include!("mlx-cxx/types.hpp");

    //     #[namespace = "mlx::core"]
    //     type float16_t;
    // }
}

fn f16_to_bits(value: ffi::f16) -> u16 {
    value.bits
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: remove later
    #[test]
    fn test_cxx_f16_to_bits() {
        let value = ffi::f16 { bits: 0x3c00 };
        let bits = unsafe { ffi::cxx_f16_to_bits(value) };
        assert_eq!(bits, 0x3c00);
    }
}