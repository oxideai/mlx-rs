#[repr(C, u8)]
pub enum StreamOrDevice {
    Default,
    Stream(ffi::Stream),
    Device(ffi::Device),
}

impl Default for StreamOrDevice {
    fn default() -> Self {
        Self::Default
    }
}

unsafe impl cxx::ExternType for StreamOrDevice {
    type Id = cxx::type_id!("mlx_cxx::StreamOrDevice");
    type Kind = cxx::kind::Trivial;
}

#[repr(C, u8)]
pub enum Optional<T> {
    None,
    Some(T),
}

impl<T> Default for Optional<T> {
    fn default() -> Self {
        Self::None
    }
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/stream.h");
        include!("mlx/device.h");

        #[namespace = "mlx::core"]
        type Stream = crate::stream::ffi::Stream;

        #[namespace = "mlx::core"]
        type Device = crate::device::ffi::Device;
    }

    unsafe extern "C++" {
        include!("mlx-cxx/mlx_cxx.hpp");

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::StreamOrDevice;
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
pub mod utils;
pub mod linalg;
pub mod ops;
pub mod io;