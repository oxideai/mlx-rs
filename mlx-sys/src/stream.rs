#[cxx::bridge]
pub mod ffi {
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
        fn default_stream(d: Device) -> Stream;

        #[namespace = "mlx::core"]
        fn set_default_device(s: Stream);

        #[namespace = "mlx::core"]
        fn new_stream(d: Device) -> Stream;
    }
}