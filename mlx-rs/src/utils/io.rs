use crate::error::{Exception, IoError};
use crate::utils::SUCCESS;
use crate::{Array, Stream};
use mlx_sys::FILE;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::{null_mut, NonNull};

pub(crate) struct FilePtr(NonNull<FILE>);

impl Drop for FilePtr {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::fclose(self.0.as_ptr());
        }
    }
}

impl FilePtr {
    pub(crate) fn as_ptr(&self) -> *mut FILE {
        self.0.as_ptr()
    }

    pub(crate) fn open(path: &Path, mode: &str) -> Result<Self, IoError> {
        let path = CString::new(path.to_str().ok_or(IoError::InvalidUtf8)?)?;
        let mode = CString::new(mode)?;

        unsafe {
            NonNull::new(mlx_sys::fopen(path.as_ptr(), mode.as_ptr()))
                .map(Self)
                .ok_or(IoError::UnableToOpenFile)
        }
    }
}

pub(crate) struct SafeTensors {
    c_metadata: mlx_sys::mlx_map_string_to_string,
    c_data: mlx_sys::mlx_map_string_to_array,
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_map_string_to_string_free(self.c_metadata);
            mlx_sys::mlx_map_string_to_array_free(self.c_data);
        }
    }
}

impl SafeTensors {
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IoError> {
        if !path.is_file() {
            return Err(IoError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IoError::UnsupportedFormat)?;

        if extension != "safetensors" {
            return Err(IoError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IoError::InvalidUtf8)?;
        let filepath = CString::new(path_str)?;

        unsafe {
            let mut c_metadata = mlx_sys::mlx_map_string_to_string_new();
            let mut c_data = mlx_sys::mlx_map_string_to_array_new();
            check_status! {
                mlx_sys::mlx_load_safetensors(&mut c_data as *mut _, &mut c_metadata as *mut _, filepath.as_ptr(), stream.as_ref().as_ptr()),
                {
                    mlx_sys::mlx_map_string_to_string_free(c_metadata);
                    mlx_sys::mlx_map_string_to_array_free(c_data);
                }
            };

            Ok(Self { c_metadata, c_data })
        }
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        if !crate::error::is_mlx_error_handler_set() {
            crate::error::setup_mlx_error_handler();
        }
        let mut map = HashMap::new();
        unsafe {
            let iterator = mlx_sys::mlx_map_string_to_array_iterator_new(self.c_data);

            loop {
                let mut key_ptr: *const ::std::os::raw::c_char = null_mut();
                let mut value = mlx_sys::mlx_array_new();
                let status = mlx_sys::mlx_map_string_to_array_iterator_next(
                    &mut key_ptr as *mut *const _,
                    &mut value,
                    iterator,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                        let array = Array::from_ptr(value);
                        map.insert(key, array);
                    }
                    1 => {
                        mlx_sys::mlx_array_free(value);
                        return Err(crate::error::get_and_clear_last_mlx_error()
                            .expect("A non-success status was returned, but no error was set."));
                    }
                    2 => {
                        mlx_sys::mlx_array_free(value);
                        break;
                    }
                    _ => unreachable!(),
                }
            }

            mlx_sys::mlx_map_string_to_array_iterator_free(iterator);
        }

        Ok(map)
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, String>, Exception> {
        if !crate::error::is_mlx_error_handler_set() {
            crate::error::setup_mlx_error_handler();
        }

        let mut map = HashMap::new();
        unsafe {
            let iterator = mlx_sys::mlx_map_string_to_string_iterator_new(self.c_metadata);

            let mut key: *const ::std::os::raw::c_char = null_mut();
            let mut value: *const ::std::os::raw::c_char = null_mut();
            loop {
                let status = mlx_sys::mlx_map_string_to_string_iterator_next(
                    &mut key as *mut *const _,
                    &mut value as *mut *const _,
                    iterator,
                );

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let value = CStr::from_ptr(value).to_string_lossy().into_owned();
                        map.insert(key, value);
                    }
                    1 => {
                        return Err(crate::error::get_and_clear_last_mlx_error()
                            .expect("A non-success status was returned, but no error was set."))
                    }
                    2 => break,
                    _ => unreachable!(),
                }
            }
        }

        Ok(map)
    }
}
