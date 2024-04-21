use mlx_sys::mlx_retain;

use crate::utils::mlx_describe;

///Type of device.
#[derive(num_enum::IntoPrimitive, Debug, Clone, Copy)]
#[repr(u32)]
pub enum DeviceType {
    Cpu = mlx_sys::mlx_device_type__MLX_CPU,
    Gpu = mlx_sys::mlx_device_type__MLX_GPU,
}

/// Representation of a Device in MLX.
pub struct Device {
    pub(crate) c_device: mlx_sys::mlx_device,
}

impl Device {
    pub fn new(device_type: DeviceType, index: i32) -> Device {
        let ctx = unsafe { mlx_sys::mlx_device_new(device_type.into(), index) };
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
    /// use mlx::{Device, DeviceType};
    /// Device::set_default(&Device::new(DeviceType::Cpu, 1));
    /// ```
    ///
    /// By default, this is `gpu()`.
    pub fn set_default(device: &Device) {
        unsafe { mlx_sys::mlx_set_default_device(device.c_device) };
    }
}

/// The `Device` is a simple struct on the c++ side
///
/// ```cpp
/// struct Device {
///   enum class DeviceType {
///     cpu,
///     gpu,
///   };
///
///   // ... other methods
///
///   DeviceType type;
///   int index;
/// };
/// ```
///
/// There is no function that mutates the device, so we can implement `Clone` for it.
impl Clone for Device {
    fn clone(&self) -> Self {
        unsafe {
            // Increment the reference count.
            mlx_retain(self.c_device as *mut std::ffi::c_void);
            Self {
                c_device: self.c_device.clone(),
            }
        }
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

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = mlx_describe(self.c_device as *mut std::os::raw::c_void);
        let description = description.unwrap_or_else(|| "Device".to_string());

        write!(f, "{}", description)
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
        println!("{:?}", device);
        assert_eq!(description, "Device(gpu, 0)");
    }
}
