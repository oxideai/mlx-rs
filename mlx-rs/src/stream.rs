use std::ffi::CStr;

use crate::{
    device::Device,
    error::Result,
    utils::{guard::Guarded, SUCCESS},
};

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// If omitted it will use the [Default::default()], which will be [Device::gpu()] unless
/// set otherwise.
#[derive(PartialEq)]
pub struct StreamOrDevice {
    pub(crate) stream: Stream,
}

impl StreamOrDevice {
    /// Create a new [`StreamOrDevice`] with a [`Stream`].
    pub fn new(stream: Stream) -> StreamOrDevice {
        StreamOrDevice { stream }
    }

    /// Create a new [`StreamOrDevice`] with a [`Device`].
    pub fn new_with_device(device: &Device) -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::new_with_device(device),
        }
    }

    /// Current default CPU stream.
    pub fn cpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::cpu(),
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::gpu(),
        }
    }
}

impl Default for StreamOrDevice {
    /// The default stream on the default device.
    ///
    /// This will be [Device::gpu()] unless [Device::set_default()]
    /// sets it otherwise.
    fn default() -> Self {
        Self {
            stream: Stream::new(),
        }
    }
}

impl AsRef<Stream> for StreamOrDevice {
    fn as_ref(&self) -> &Stream {
        &self.stream
    }
}

impl std::fmt::Debug for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

impl std::fmt::Display for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically, this is used via the `stream:` parameter on a method with a [StreamOrDevice]:
pub struct Stream {
    pub(crate) c_stream: mlx_sys::mlx_stream,
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Stream {
    /// Create a new stream on the default device. Panics if fails.
    pub fn new() -> Stream {
        unsafe {
            let mut dev = mlx_sys::mlx_device_new();
            // SAFETY: mlx_get_default_device internally never throws an error
            mlx_sys::mlx_get_default_device(&mut dev as *mut _);

            let mut c_stream = mlx_sys::mlx_stream_new();
            // SAFETY: mlx_get_default_stream internally never throws if dev is valid
            mlx_sys::mlx_get_default_stream(&mut c_stream as *mut _, dev);

            mlx_sys::mlx_device_free(dev);
            Stream { c_stream }
        }
    }

    /// Try to get the default stream on the given device.
    pub fn try_default_on_device(device: &Device) -> Result<Stream> {
        Stream::try_from_op(|res| unsafe { mlx_sys::mlx_get_default_stream(res, device.c_device) })
    }

    /// Create a new stream on the given device
    pub fn new_with_device(device: &Device) -> Stream {
        unsafe {
            let c_stream = mlx_sys::mlx_stream_new_device(device.c_device);
            Stream { c_stream }
        }
    }

    /// Get the underlying C pointer.
    pub fn as_ptr(&self) -> mlx_sys::mlx_stream {
        self.c_stream
    }

    /// Current default CPU stream.
    pub fn cpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_cpu_stream_new();
            Stream { c_stream }
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_gpu_stream_new();
            Stream { c_stream }
        }
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let result = match mlx_sys::mlx_stream_tostring(&mut mlx_str as *mut _, self.c_stream) {
                SUCCESS => {
                    let ptr = mlx_sys::mlx_string_data(mlx_str);
                    let c_str = CStr::from_ptr(ptr);
                    write!(f, "{}", c_str.to_string_lossy())
                }
                _ => Err(std::fmt::Error),
            };
            mlx_sys::mlx_string_free(mlx_str);
            result
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_stream_free(self.c_stream) };
    }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlx_sys::mlx_stream_equal(self.c_stream, other.c_stream) }
    }
}
