#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    pub struct complex64_t {
        pub re: f32,
        pub im: f32,
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        fn real(self: &complex64_t) -> f32;

        #[namespace = "mlx_cxx"]
        fn imag(self: &complex64_t) -> f32;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");
    }
}

impl ffi::complex64_t {
    pub fn new(re: f32, im: f32) -> Self {
        ffi::complex64_t { re, im }
    }

    pub fn real(&self) -> f32 {
        self.re
    }

    pub fn imag(&self) -> f32 {
        self.im
    }
}