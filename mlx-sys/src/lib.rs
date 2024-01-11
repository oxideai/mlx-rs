#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/stream.h");
        include!("mlx/device.h");

        #[namespace = "mlx::core"]
        type Stream;

        #[namespace = "mlx::core"]
        type Device;
    }
}

pub mod types;
pub mod array;
pub mod dtype;
pub mod macros;
pub mod backend;
pub mod device;
pub mod stream;
pub mod fft;