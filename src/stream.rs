use crate::device::Device;
use crate::utils::mlx_describe;

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// If omitted it will use the [default()], which will be [Device::gpu()] unless
/// set otherwise.
#[derive(Clone)]
pub struct StreamOrDevice {
    pub(crate) stream: Stream,
}

impl StreamOrDevice {
    pub fn new(stream: Stream) -> StreamOrDevice {
        StreamOrDevice { stream }
    }

    pub fn new_with_device(device: &Device) -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::default_stream_on_device(device),
        }
    }

    /// The `[Stream::default_stream()] on the [Device::cpu()]
    pub fn cpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::default_stream_on_device(&Device::cpu()),
        }
    }

    /// The `[Stream::default_stream()] on the [Device::gpu()]
    pub fn gpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::default_stream_on_device(&Device::gpu()),
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

    pub fn new() -> Stream {
        let default_device = unsafe { mlx_sys::mlx_default_device() };
        let c_stream = unsafe { mlx_sys::mlx_default_stream(default_device) };
        unsafe { mlx_sys::mlx_free(default_device as *mut std::ffi::c_void) };
        Stream { c_stream }
    }

    pub fn new_with_device(index: i32, device: &Device) -> Stream {
        let c_stream = unsafe { mlx_sys::mlx_stream_new(index, device.c_device) };
        Stream { c_stream }
    }

    pub fn default_stream_on_device(device: &Device) -> Stream {
        let default_stream = unsafe { mlx_sys::mlx_default_stream(device.c_device) };
        Stream::new_with_mlx_mlx_stream(default_stream)
    }
}

/// The `Stream` is a simple struct on the c++ side
///
/// ```cpp
/// struct Stream {
///     int index;
///     Device device;
///
///     // ... constructor
/// };
/// ```
///
/// There is no function that mutates the stream, so we can implement `Clone` for it.
impl Clone for Stream {
    fn clone(&self) -> Self {
        unsafe {
            // Increment the reference count.
            mlx_sys::mlx_retain(self.c_stream as *mut std::ffi::c_void);
            Stream {
                c_stream: self.c_stream,
            }
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_stream as *mut std::ffi::c_void) };
    }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = mlx_describe(self.c_stream as *mut std::os::raw::c_void);
        let description = description.unwrap_or_else(|| "Stream".to_string());
        write!(f, "{}", description)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = mlx_describe(self.c_stream as *mut std::os::raw::c_void);
        let description = description.unwrap_or_else(|| "Stream".to_string());
        write!(f, "{}", description)
    }
}
