use crate::error::{Exception, IOError};
use crate::utils::MlxString;
use crate::{Array, Stream, StreamOrDevice};
use mlx_internal_macros::default_device;
use mlx_sys::FILE;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;

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
        let path = CString::new(path.to_str().ok_or(IOError::InvalidUtf8)?)
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
    #[default_device]
    pub(crate) fn load_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Self, IOError> {
        if !path.is_file() {
            return Err(IOError::NotFile);
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or(IOError::UnsupportedFormat)?;

        if extension != "safetensors" {
            return Err(IOError::UnsupportedFormat);
        }

        let path_str = path.to_str().ok_or(IOError::InvalidUtf8)?;
        let mlx_path = MlxString::try_from(path_str).map_err(|_| IOError::NullBytes)?;

        let load_result = (|| unsafe {
            let c_safetensors = try_catch_c_ptr_expr! {
                mlx_sys::mlx_load_safetensors(mlx_path.as_ptr(), stream.as_ref().as_ptr())
            };

            Ok(Self { c_safetensors })
        })();

        match load_result {
            Ok(map) => Ok(map),
            Err(e) => Err(IOError::from(e)),
        }
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        let arrays = unsafe {
            StringToArrayMap::from_ptr(try_catch_c_ptr_expr! {
                mlx_sys::mlx_safetensors_data(self.c_safetensors)
            })
        };

        Ok(arrays.as_hash_map())
    }

    pub(crate) fn metadata(&self) -> Result<HashMap<String, String>, Exception> {
        let metadata = unsafe {
            StringToStringMap::from_ptr(try_catch_c_ptr_expr! {
                mlx_sys::mlx_safetensors_metadata(self.c_safetensors)
            })
        };

        Ok(metadata.as_hash_map())
    }
}

// Trait to define required MLX operations for a type
pub(crate) trait MlxMapValue: Sized {
    type MapType;
    type CType;
    type Iterator;

    fn mlx_map_new() -> Self::MapType;
    fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator;
    fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool;
    fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string;
    fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType;
    fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool;
    fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool;
    fn from_mlx_ptr(ptr: Self::CType) -> Self;
    fn as_mlx_ptr(&self) -> Self::CType;
}

// Implementation for Array
impl MlxMapValue for Array {
    type MapType = mlx_sys::mlx_map_string_to_array;
    type CType = mlx_sys::mlx_array;
    type Iterator = mlx_sys::mlx_map_string_to_array_iterator;

    fn mlx_map_new() -> Self::MapType {
        unsafe { mlx_sys::mlx_map_string_to_array_new() }
    }

    fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator {
        unsafe { mlx_sys::mlx_map_string_to_array_iterate(map) }
    }

    fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_array_iterator_end(iterator) }
    }

    fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string {
        unsafe { mlx_sys::mlx_map_string_to_array_iterator_key(iterator) }
    }

    fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType {
        unsafe { mlx_sys::mlx_map_string_to_array_iterator_value(iterator) }
    }

    fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_array_iterator_next(iterator) }
    }

    fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_array_insert(map, key, value) }
    }

    fn from_mlx_ptr(ptr: Self::CType) -> Self {
        unsafe { Array::from_ptr(ptr) }
    }

    fn as_mlx_ptr(&self) -> Self::CType {
        self.as_ptr()
    }
}

// Implementation for String
impl MlxMapValue for String {
    type MapType = mlx_sys::mlx_map_string_to_string;
    type CType = mlx_sys::mlx_string;
    type Iterator = mlx_sys::mlx_map_string_to_string_iterator;

    fn mlx_map_new() -> Self::MapType {
        unsafe { mlx_sys::mlx_map_string_to_string_new() }
    }

    fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator {
        unsafe { mlx_sys::mlx_map_string_to_string_iterate(map) }
    }

    fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_string_iterator_end(iterator) }
    }

    fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string {
        unsafe { mlx_sys::mlx_map_string_to_string_iterator_key(iterator) }
    }

    fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType {
        unsafe { mlx_sys::mlx_map_string_to_string_iterator_value(iterator) }
    }

    fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_string_iterator_next(iterator) }
    }

    fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool {
        unsafe { mlx_sys::mlx_map_string_to_string_insert(map, key, value) }
    }

    fn from_mlx_ptr(ptr: Self::CType) -> Self {
        unsafe {
            CStr::from_ptr(mlx_sys::mlx_string_data(ptr))
                .to_string_lossy()
                .into_owned()
        }
    }

    fn as_mlx_ptr(&self) -> Self::CType {
        MlxString::try_from(self.as_str()).unwrap().as_ptr()
    }
}

// Generic map structure
pub(crate) struct StringToMap<T: MlxMapValue> {
    c_map: T::MapType,
    _phantom: PhantomData<T>,
}

impl<T: MlxMapValue> Drop for StringToMap<T> {
    fn drop(&mut self) {
        unsafe {
            let ptr = &self.c_map as *const T::MapType as *mut c_void;
            mlx_sys::mlx_free(ptr)
        }
    }
}

impl<T: MlxMapValue> StringToMap<T> {
    pub(crate) fn from_ptr(c_map: T::MapType) -> Self {
        Self {
            c_map,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> T::MapType
    where
        T::MapType: Copy,
    {
        self.c_map
    }

    pub(crate) fn as_hash_map(&self) -> HashMap<String, T>
    where
        T::MapType: Copy,
        T::Iterator: Copy,
    {
        let mut result = HashMap::new();

        unsafe {
            let iterator = T::mlx_map_iterate(self.as_ptr());

            while !T::mlx_map_iterator_end(iterator) {
                let key_ptr = T::mlx_map_iterator_key(iterator);
                let key = CStr::from_ptr(mlx_sys::mlx_string_data(key_ptr))
                    .to_string_lossy()
                    .into_owned();

                let value = T::from_mlx_ptr(T::mlx_map_iterator_value(iterator));
                result.insert(key, value);

                T::mlx_map_iterator_next(iterator);
            }

            let ptr = &iterator as *const T::Iterator as *mut c_void;
            mlx_sys::mlx_free(ptr);
        }

        result
    }
}

impl<T: MlxMapValue> TryFrom<&HashMap<String, T>> for StringToMap<T>
where
    T::MapType: Copy,
{
    type Error = IOError;

    fn try_from(hashmap: &HashMap<String, T>) -> Result<Self, Self::Error> {
        let mlx_map = T::mlx_map_new();

        for (key, value) in hashmap {
            let mlx_key = MlxString::try_from(key.as_str()).unwrap();
            let success = T::mlx_map_insert(mlx_map, mlx_key.as_ptr(), value.as_mlx_ptr());
            if !success {
                let ptr = &mlx_map as *const T::MapType as *mut c_void;
                unsafe {
                    mlx_sys::mlx_free(ptr);
                }

                return Err(IOError::AllocationError);
            }
        }

        Ok(Self::from_ptr(mlx_map))
    }
}

pub(crate) type StringToArrayMap = StringToMap<Array>;
pub(crate) type StringToStringMap = StringToMap<String>;
