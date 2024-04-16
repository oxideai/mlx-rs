use crate::utils::mlx_describe;

///Type of device.
pub enum DeviceType {
    Cpu,
    Gpu,
}

/// Representation of a Device in MLX.
#[derive(Debug)]
pub struct Device {
    ctx: mlx_sys::mlx_device,
}

impl Device {
    pub fn new_default() -> Device {
        let ctx = unsafe { mlx_sys::mlx_default_device() };
        Device { ctx }
    }

    pub fn new(device_type: DeviceType, index: i32) -> Device {
        let c_device_type: u32 = match device_type {
            DeviceType::Cpu => mlx_sys::mlx_device_type__MLX_CPU,
            DeviceType::Gpu => mlx_sys::mlx_device_type__MLX_GPU,
        };

        let ctx = unsafe { mlx_sys::mlx_device_new(c_device_type, index) };
        Device { ctx }
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
    /// Device::set_default(Device::new(DeviceType::Cpu, 1));
    /// ```
    ///
    /// By default, this is `gpu()`.
    pub fn set_default(&self) {
        unsafe { mlx_sys::mlx_set_default_device(self.ctx) };
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.ctx as *mut std::ffi::c_void) };
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = unsafe { mlx_describe(self.ctx as *mut std::os::raw::c_void) };
        let description = description.unwrap_or_else(|| "Device".to_string());

        write!(f, "{}", description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt() {
        let device = Device::new_default();
        let description = format!("{}", device);
        assert_eq!(description, "Device(gpu, 0)");
    }
}
