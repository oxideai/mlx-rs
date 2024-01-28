#[cxx::bridge]
pub mod ffi {
    #[namespace = "mlx_cxx"]
    #[cxx_name = "DeviceDeviceType"]
    #[repr(i32)]
    pub enum DeviceType {
        cpu,
        gpu,
    }

    #[namespace = "mlx::core"]
    pub struct Device {
        pub r#type: DeviceType,
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
        fn set_default_device(device: &Device);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_default_deveice() {
        let device = super::ffi::default_device();
        println!("{:?}", device.index);
    }

    #[test]
    fn test_new_device() {
        let device = super::ffi::new_device(super::ffi::DeviceType::gpu, 0);
        println!("{:?}", device.index);
    }

    #[test]
    fn test_default_device() {
        let _device = super::ffi::default_device();
    }

    #[test]
    fn test_set_default_device() {
        let device = super::ffi::new_device(super::ffi::DeviceType::gpu, 0);
        super::ffi::set_default_device(&device);
    }
}