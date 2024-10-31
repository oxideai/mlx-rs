use crate::error::IOError;
use crate::utils::{mlx_map_array_values, mlx_map_string_values};
use crate::{Array, Stream};
use mlx_internal_macros::default_device;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;

/// Load array from a binary file in `.npy` format.
///
/// # Params
/// - path: path of file to load
/// - stream: stream or device to evaluate on
#[default_device]
pub fn load_array_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Array, IOError> {
    if !path.is_file() {
        return Err(IOError::NotFile);
    }

    // Convert path to string, handling invalid UTF-8
    let path_str = path.to_str().ok_or_else(|| IOError::InvalidUtf8)?;

    // Create C string for filename, handling null bytes
    let filename = CString::new(path_str).map_err(|_| IOError::NullBytes)?;

    // Create MLX string
    let filename = unsafe { mlx_sys::mlx_string_new(filename.as_ptr()) };

    let result = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| IOError::NoExtension)?
    {
        "npy" => unsafe {
            let load_result = (|| unsafe {
                let c_array = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_load(filename, stream.as_ref().as_ptr())
                };
                Ok(Array::from_ptr(c_array))
            })();

            match load_result {
                Ok(array) => Ok(array),
                Err(e) => Err(IOError::from(e)),
            }
        },
        _ => Err(IOError::UnsupportedFormat),
    };

    unsafe {
        mlx_sys::mlx_free(filename as *mut c_void);
    }
    result
}

/// Load dictionary of ``MLXArray`` from a `safetensors` file.
///
/// # Params
/// - path: path of file to load
/// - stream: stream or device to evaluate on
///
#[default_device]
pub fn load_arrays_device(
    path: &Path,
    stream: impl AsRef<Stream>,
) -> Result<HashMap<String, Array>, IOError> {
    if !path.is_file() {
        return Err(IOError::NotFile);
    }

    // Convert path to string, handling invalid UTF-8
    let path_str = path.to_str().ok_or_else(|| IOError::InvalidUtf8)?;

    // Create C string for filename, handling null bytes
    let filename = CString::new(path_str).map_err(|_| IOError::NullBytes)?;

    // Create MLX string
    let filename = unsafe { mlx_sys::mlx_string_new(filename.as_ptr()) };

    let result = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| IOError::NoExtension)?
    {
        "safetensors" => unsafe {
            let load_result = (|| unsafe {
                let mlx_safetensors = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_load_safetensors(filename, stream.as_ref().as_ptr())
                };

                let mlx_arrays = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_safetensors_data(mlx_safetensors)
                };

                mlx_sys::mlx_free(mlx_safetensors as *mut c_void);
                let map = mlx_map_array_values(mlx_arrays);

                mlx_sys::mlx_free(mlx_arrays as *mut c_void);
                Ok(map)
            })();

            match load_result {
                Ok(map) => Ok(map),
                Err(e) => Err(IOError::from(e)),
            }
        },
        _ => Err(IOError::UnsupportedFormat),
    };

    unsafe {
        mlx_sys::mlx_free(filename as *mut c_void);
    }
    result
}

/// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
///
/// # Params
/// - path: path of file to load
/// - stream: stream or device to evaluate on
#[default_device]
pub fn load_arrays_with_metadata_device(
    path: &Path,
    stream: impl AsRef<Stream>,
) -> Result<(HashMap<String, Array>, HashMap<String, String>), IOError> {
    if !path.is_file() {
        return Err(IOError::NotFile);
    }

    // Convert path to string, handling invalid UTF-8
    let path_str = path.to_str().ok_or_else(|| IOError::InvalidUtf8)?;

    // Create C string for filename, handling null bytes
    let filename = CString::new(path_str).map_err(|_| IOError::NullBytes)?;

    // Create MLX string
    let filename = unsafe { mlx_sys::mlx_string_new(filename.as_ptr()) };

    let result = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| IOError::NoExtension)?
    {
        "safetensors" => unsafe {
            let load_result = (|| unsafe {
                let mlx_safetensors = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_load_safetensors(filename, stream.as_ref().as_ptr())
                };

                let mlx_arrays = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_safetensors_data(mlx_safetensors)
                };

                let mlx_metadata = try_catch_c_ptr_expr! {
                    mlx_sys::mlx_safetensors_metadata(mlx_safetensors)
                };

                mlx_sys::mlx_free(mlx_safetensors as *mut c_void);
                let map = mlx_map_array_values(mlx_arrays);
                let metadata = mlx_map_string_values(mlx_metadata);

                mlx_sys::mlx_free(mlx_arrays as *mut c_void);
                mlx_sys::mlx_free(mlx_metadata as *mut c_void);
                Ok((map, metadata))
            })();

            match load_result {
                Ok(map) => Ok(map),
                Err(e) => Err(IOError::from(e)),
            }
        },
        _ => Err(IOError::UnsupportedFormat),
    };

    unsafe {
        mlx_sys::mlx_free(filename as *mut c_void);
    }
    result
}
