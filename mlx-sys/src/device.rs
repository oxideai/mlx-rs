#[cxx::bridge]
pub mod ffi {
    #[derive(Clone, Copy)]
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DeviceDeviceType"]
    #[repr(i32)]
    pub enum DeviceType {
        #[cxx_name = "cpu"]
        Cpu,
        #[cxx_name = "gpu"]
        Gpu,
    }

    #[derive(Clone, Copy)]
    #[namespace = "mlx::core"]
    pub struct Device {
        pub device_type: DeviceType,
        pub index: i32,
    }

    unsafe extern "C++" {
        include!("mlx/device.h");
        include!("mlx-cxx/device.hpp");

        #[namespace = "mlx_cxx"]
        #[cxx_name = "DeviceDeviceType"]
        type DeviceType;

        #[namespace = "mlx::core"]
        type Device;

        #[namespace = "mlx_cxx"]
        fn new_device(device_type: DeviceType, index: i32) -> Device;

        #[namespace = "mlx::core"]
        fn default_device() -> &'static Device;

        #[namespace = "mlx::core"]
        fn set_default_device(device: &Device) -> Result<()>;
    }
}
