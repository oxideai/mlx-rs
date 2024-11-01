use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use mlx_sys::FILE;
use crate::Array;
use crate::error::IOError;
use crate::utils::MlxString;

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

    pub(crate) fn open(path: &Path, mode: &str) -> Result<Self, IOError> {
        let path = CString::new(path.to_str().ok_or_else(|| IOError::InvalidUtf8)?)
            .map_err(|_| IOError::NullBytes)?;
        let mode = CString::new(mode).map_err(|_| IOError::NullBytes)?;

        unsafe {
            NonNull::new(mlx_sys::fopen(path.as_ptr(), mode.as_ptr()))
                .map(Self)
                .ok_or(IOError::UnableToOpenFile)
        }
    }
}

pub(crate) struct SafeTensors {
    c_safetensors: mlx_sys::mlx_safetensors,
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_safetensors as *mut std::os::raw::c_void) }
    }
}

impl SafeTensors {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_safetensors {
        self.c_safetensors
    }

    pub(crate) unsafe fn from_ptr(c_safetensors: mlx_sys::mlx_safetensors) -> Self {
        Self { c_safetensors }
    }
}

pub(crate) struct StringToArrayMap {
    c_map: mlx_sys::mlx_map_string_to_array,
}

impl Drop for StringToArrayMap {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_map as *mut c_void) }
    }
}

impl StringToArrayMap {
    pub(crate) fn from_ptr(c_map: mlx_sys::mlx_map_string_to_array) -> Self {
        Self { c_map }
    }

    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_map_string_to_array {
        self.c_map
    }

    pub(crate) fn as_hash_map(&self) -> HashMap<String, Array> {
        let mut result = HashMap::new();

        unsafe {
            let iterator = mlx_sys::mlx_map_string_to_array_iterate(self.as_ptr());

            while !mlx_sys::mlx_map_string_to_array_iterator_end(iterator) {
                let key =
                    MlxString::from_ptr(mlx_sys::mlx_map_string_to_array_iterator_key(iterator));
                let key = CStr::from_ptr(mlx_sys::mlx_string_data(key.as_ptr()))
                    .to_string_lossy()
                    .into_owned();

                let mlx_array_ctx = mlx_sys::mlx_map_string_to_array_iterator_value(iterator);
                let array = Array::from_ptr(mlx_array_ctx);
                result.insert(key, array);

                mlx_sys::mlx_map_string_to_array_iterator_next(iterator);
            }

            mlx_sys::free(iterator as *mut c_void);
        }

        result
    }
}

impl TryFrom<&HashMap<String, Array>> for StringToArrayMap {
    type Error = IOError;

    fn try_from(hashmap: &HashMap<String, Array>) -> Result<Self, Self::Error> {
        let ptr = unsafe {
            let mlx_map = mlx_sys::mlx_map_string_to_array_new();

            for (key, array) in hashmap {
                let mlx_key = MlxString::try_from(key.as_str()).unwrap();
                mlx_sys::mlx_map_string_to_array_insert(mlx_map, mlx_key.as_ptr(), array.as_ptr());
            }

            mlx_map
        };

        if ptr.is_null() {
            return Err(IOError::AllocationError);
        }

        Ok(Self::from_ptr(ptr))
    }
}

pub(crate) struct StringToStringMap {
    c_map: mlx_sys::mlx_map_string_to_string,
}

impl Drop for StringToStringMap {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_free(self.c_map as *mut c_void) }
    }
}

impl StringToStringMap {
    pub(crate) fn from_ptr(c_map: mlx_sys::mlx_map_string_to_string) -> Self {
        Self { c_map }
    }

    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_map_string_to_string {
        self.c_map
    }

    pub(crate) fn as_hash_map(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();

        unsafe {
            let iterator = mlx_sys::mlx_map_string_to_string_iterate(self.as_ptr());

            while !mlx_sys::mlx_map_string_to_string_iterator_end(iterator) {
                let key =
                    MlxString::from_ptr(mlx_sys::mlx_map_string_to_string_iterator_key(iterator));
                let key = CStr::from_ptr(mlx_sys::mlx_string_data(key.as_ptr()))
                    .to_string_lossy()
                    .into_owned();

                let value =
                    MlxString::from_ptr(mlx_sys::mlx_map_string_to_string_iterator_value(iterator));
                let value = CStr::from_ptr(mlx_sys::mlx_string_data(value.as_ptr()))
                    .to_string_lossy()
                    .into_owned();

                result.insert(key, value);

                mlx_sys::mlx_map_string_to_string_iterator_next(iterator);
            }

            mlx_sys::mlx_free(iterator as *mut c_void);
        }

        result
    }
}

impl TryFrom<&HashMap<String, String>> for StringToStringMap {
    type Error = IOError;

    fn try_from(hashmap: &HashMap<String, String>) -> Result<Self, Self::Error> {
        let ptr = unsafe {
            let mlx_map = mlx_sys::mlx_map_string_to_string_new();

            for (key, value) in hashmap {
                let mlx_key = MlxString::try_from(key.as_str()).unwrap();
                let mlx_value = MlxString::try_from(value.as_str()).unwrap();
                mlx_sys::mlx_map_string_to_string_insert(
                    mlx_map,
                    mlx_key.as_ptr(),
                    mlx_value.as_ptr(),
                );
            }

            mlx_map
        };

        if ptr.is_null() {
            return Err(IOError::AllocationError);
        }

        Ok(Self::from_ptr(ptr))
    }
}
