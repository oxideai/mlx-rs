#[cxx::bridge]
pub mod ffi {
    #[derive(Clone)]
    #[namespace = "mlx::core"]
    pub struct Stream {
        pub index: i32,
        pub device: Device,
    }

    unsafe extern "C++" {
        include!("mlx/stream.h");
        include!("mlx/device.h");

        #[namespace = "mlx::core"]
        type Device = crate::device::ffi::Device;

        #[namespace = "mlx::core"]
        type Stream;

        #[namespace = "mlx::core"]
        fn default_stream(d: Device) -> Result<Stream>;

        #[namespace = "mlx::core"]
        fn set_default_stream(s: Stream) -> Result<()>;

        #[namespace = "mlx::core"]
        fn new_stream(d: Device) -> Result<Stream>;
    }
}
