use mlx_sys::device::ffi::*;

#[test]
fn test_default_device() {
    let default_device = default_device();
    let device_type = default_device.device_type;

    #[cfg(feature = "metal")]
    assert!(matches!(device_type, DeviceType::Gpu));

    #[cfg(not(feature = "metal"))]
    assert!(matches!(device_type, DeviceType::Cpu));
}

#[test]
fn test_new_device() {
    #[cfg(feature = "metal")]
    let _device = new_device(DeviceType::Gpu, 0);

    #[cfg(not(feature = "metal"))]
    let _device = new_device(DeviceType::Cpu, 0);
}

#[test]
fn test_set_default_device() {
    #[cfg(feature = "metal")]
    let device = new_device(DeviceType::Gpu, 0);

    #[cfg(not(feature = "metal"))]
    let device = new_device(DeviceType::Cpu, 0);

    let result = set_default_device(&device);
    assert!(result.is_ok());
}
