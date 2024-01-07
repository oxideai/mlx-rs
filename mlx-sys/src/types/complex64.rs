#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    pub struct c64 {
        pub re: f32,
        pub im: f32,
    }

    extern "Rust" {
        #[namespace = "mlx_cxx"]
        fn real(self: &c64) -> f32;

        #[namespace = "mlx_cxx"]
        fn imag(self: &c64) -> f32;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/types.hpp");
    }
}

impl ffi::c64 {
    pub fn new(re: f32, im: f32) -> Self {
        ffi::c64 { re, im }
    }

    pub fn real(&self) -> f32 {
        self.re
    }

    pub fn imag(&self) -> f32 {
        self.im
    }
}