use std::ffi::CStr;

use crate::{device::Device, error::Exception, utils::SUCCESS};

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
    pub fn new(stream: Stream) -> StreamOrDevice {
        StreamOrDevice { stream }
    }

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

    /// The underlying stream's pointer.
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_stream {
        self.stream.c_stream
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

impl Stream {
    fn new_with_mlx_mlx_stream(stream: mlx_sys::mlx_stream) -> Stream {
        Stream { c_stream: stream }
    }

    /// Create a new stream on the default device. Panics if fails.
    pub fn new() -> Stream {
        // TODO: is there a better way to handle this so that we don't panic?
        Self::try_new().unwrap()
    }

    /// Try to create a new stream on the default device
    pub fn try_new() -> Result<Stream, Exception> {
        unsafe {
            let mut dev = mlx_sys::mlx_device_new();
            check_status!{
                mlx_sys::mlx_get_default_device(&mut dev as *mut _),
                mlx_sys::mlx_device_free(dev)
            };
            let mut c_stream = mlx_sys::mlx_stream_new();
            check_status!{
                mlx_sys::mlx_get_default_stream(&mut c_stream as *mut _, dev),
                {
                    let _ = mlx_sys::mlx_stream_free(c_stream);
                    mlx_sys::mlx_device_free(dev)
                }
            };
            check_status!{
                mlx_sys::mlx_device_free(dev),
                mlx_sys::mlx_stream_free(c_stream)
            };
            Ok(Stream { c_stream })
        }
    }

    pub fn try_default_on_device(device: &Device) -> Result<Stream, Exception> {
        unsafe {
            let mut c_stream = mlx_sys::mlx_stream_new();
            check_status!{
                mlx_sys::mlx_get_default_stream(&mut c_stream as *mut _, device.c_device),
                mlx_sys::mlx_stream_free(c_stream)
            };
            Ok(Stream { c_stream })
        }
    }

    pub fn new_with_device(device: &Device) -> Stream {
        unsafe {
            let c_stream = mlx_sys::mlx_stream_new_device(device.c_device);
            Stream { c_stream }
        }
    }

    pub fn as_raw(&self) -> mlx_sys::mlx_stream {
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
                },
                _ => Err(std::fmt::Error::default())
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
