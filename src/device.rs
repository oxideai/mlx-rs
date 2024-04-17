use crate::utils::mlx_describe;

///Type of device.
pub enum DeviceType {
    Cpu,
    Gpu,
}

/// Representation of a Device in MLX.
#[derive(Debug)]
pub struct Device {
    pub(crate) c_device: mlx_sys::mlx_device,
}

impl Device {
    pub fn new(device_type: DeviceType, index: i32) -> Device {
        let c_device_type: u32 = match device_type {
            DeviceType::Cpu => mlx_sys::mlx_device_type__MLX_CPU,
            DeviceType::Gpu => mlx_sys::mlx_device_type__MLX_GPU,
        };

        let ctx = unsafe { mlx_sys::mlx_device_new(c_device_type, index) };
        Device { c_device: ctx }
    }

    pub fn cpu() -> Device {
        Device::new(DeviceType::Cpu, 0)
    }

    pub fn gpu() -> Device {
        Device::new(DeviceType::Gpu, 0)
    }

    /// Set the default device.
    ///
    /// Example:
    /// ```rust
    /// use mlx::device::{Device, DeviceType};
    /// Device::set_default(&Device::new(DeviceType::Cpu, 1));
    /// ```
    ///
    /// By default, this is `gpu()`.
    pub fn set_default(device: &Device) {
        unsafe { mlx_sys::mlx_set_default_device(device.c_device) };
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_device as *mut std::ffi::c_void) };
    }
}

impl Default for Device {
    fn default() -> Self {
        let ctx = unsafe { mlx_sys::mlx_default_device() };
        Self { c_device: ctx }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = mlx_describe(self.c_device as *mut std::os::raw::c_void);
        let description = description.unwrap_or_else(|| "Device".to_string());

        write!(f, "{}", description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt() {
        let device = Device::default();
        let description = format!("{}", device);
        assert_eq!(description, "Device(gpu, 0)");
    }
}
