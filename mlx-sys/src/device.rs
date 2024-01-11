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
}