use crate::error::{Exception, IoError};
use crate::utils::{MlxString, SUCCESS};
use crate::{Array, Stream};
use mlx_sys::FILE;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
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

        let load_result = (|| unsafe {
            let mut c_metadata = mlx_sys::mlx_map_string_to_string_new();
            let mut c_data = mlx_sys::mlx_map_string_to_array_new();
            check_status! {
                mlx_sys::mlx_load_safetensors(&mut c_data as *mut _, &mut c_metadata as *mut _, filepath.as_ptr(), stream.as_ref().as_ptr()),
                {
                    mlx_sys::mlx_map_string_to_string_free(c_metadata);
                    mlx_sys::mlx_map_string_to_array_free(c_data);
                }
            };

            Ok(Self {
                c_metadata,
                c_data,
            })
        })();

        match load_result {
            Ok(map) => Ok(map),
            Err(e) => Err(IoError::from(e)),
        }
    }

    pub(crate) fn data(&self) -> Result<HashMap<String, Array>, Exception> {
        if !crate::error::is_mlx_error_handler_set() {
            crate::error::setup_mlx_error_handler();
        }
        let mut map = HashMap::new();
        unsafe {
            let iterator = mlx_sys::mlx_map_string_to_array_iterator_new(self.c_data);

            let mut key: *const ::std::os::raw::c_char = null_mut();
            let mut value = mlx_sys::mlx_array_new();
            loop {
                let status = mlx_sys::mlx_map_string_to_array_iterator_next(&mut key as *mut *const _, &mut value, iterator);

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let array = Array::from_ptr(value);
                        map.insert(key, array);
                    },
                    1 => {
                        mlx_sys::mlx_array_free(value);
                        return Err(crate::error::get_and_clear_last_mlx_error()
                            .expect("A non-success status was returned, but no error was set."));
                    },
                    2 => {
                        mlx_sys::mlx_array_free(value);
                        break;
                    },
                    _ => unreachable!()
                }
            }
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
                let status = mlx_sys::mlx_map_string_to_string_iterator_next(&mut key as *mut *const _, &mut value as *mut *const _, iterator);

                match status {
                    SUCCESS => {
                        let key = CStr::from_ptr(key).to_string_lossy().into_owned();
                        let value = CStr::from_ptr(value).to_string_lossy().into_owned();
                        map.insert(key, value);
                    },
                    1 => return Err(crate::error::get_and_clear_last_mlx_error()
                        .expect("A non-success status was returned, but no error was set.")),
                    2 => break,
                    _ => unreachable!()
                }
            }
        }

        Ok(map)
    }
}

// // Trait to define required MLX operations for a type
// pub(crate) trait MlxMapValue: Sized {
//     type MapType;
//     type CType;
//     type Iterator;

//     fn mlx_map_new() -> Self::MapType;
//     fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator;
//     fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool;
//     fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string;
//     fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType;
//     fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool;
//     fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool;
//     fn from_mlx_ptr(ptr: Self::CType) -> Self;
//     fn as_mlx_ptr(&self) -> Self::CType;
// }

// // Implementation for Array
// impl MlxMapValue for Array {
//     type MapType = mlx_sys::mlx_map_string_to_array;
//     type CType = mlx_sys::mlx_array;
//     type Iterator = mlx_sys::mlx_map_string_to_array_iterator;

//     fn mlx_map_new() -> Self::MapType {
//         unsafe { mlx_sys::mlx_map_string_to_array_new() }
//     }

//     fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator {
//         unsafe { mlx_sys::mlx_map_string_to_array_iterate(map) }
//     }

//     fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_array_iterator_end(iterator) }
//     }

//     fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string {
//         unsafe { mlx_sys::mlx_map_string_to_array_iterator_key(iterator) }
//     }

//     fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType {
//         unsafe { mlx_sys::mlx_map_string_to_array_iterator_value(iterator) }
//     }

//     fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_array_iterator_next(iterator) }
//     }

//     fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_array_insert(map, key, value) }
//     }

//     fn from_mlx_ptr(ptr: Self::CType) -> Self {
//         unsafe { Array::from_ptr(ptr) }
//     }

//     fn as_mlx_ptr(&self) -> Self::CType {
//         self.as_ptr()
//     }
// }

// // Implementation for String
// impl MlxMapValue for String {
//     type MapType = mlx_sys::mlx_map_string_to_string;
//     type CType = mlx_sys::mlx_string;
//     type Iterator = mlx_sys::mlx_map_string_to_string_iterator;

//     fn mlx_map_new() -> Self::MapType {
//         unsafe { mlx_sys::mlx_map_string_to_string_new() }
//     }

//     fn mlx_map_iterate(map: Self::MapType) -> Self::Iterator {
//         unsafe { mlx_sys::mlx_map_string_to_string_iterate(map) }
//     }

//     fn mlx_map_iterator_end(iterator: Self::Iterator) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_string_iterator_end(iterator) }
//     }

//     fn mlx_map_iterator_key(iterator: Self::Iterator) -> mlx_sys::mlx_string {
//         unsafe { mlx_sys::mlx_map_string_to_string_iterator_key(iterator) }
//     }

//     fn mlx_map_iterator_value(iterator: Self::Iterator) -> Self::CType {
//         unsafe { mlx_sys::mlx_map_string_to_string_iterator_value(iterator) }
//     }

//     fn mlx_map_iterator_next(iterator: Self::Iterator) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_string_iterator_next(iterator) }
//     }

//     fn mlx_map_insert(map: Self::MapType, key: mlx_sys::mlx_string, value: Self::CType) -> bool {
//         unsafe { mlx_sys::mlx_map_string_to_string_insert(map, key, value) }
//     }

//     fn from_mlx_ptr(ptr: Self::CType) -> Self {
//         unsafe {
//             CStr::from_ptr(mlx_sys::mlx_string_data(ptr))
//                 .to_string_lossy()
//                 .into_owned()
//         }
//     }

//     fn as_mlx_ptr(&self) -> Self::CType {
//         MlxString::try_from(self.as_str()).unwrap().as_ptr()
//     }
// }

// // Generic map structure
// pub(crate) struct StringToMap<T: MlxMapValue> {
//     c_map: T::MapType,
//     _phantom: PhantomData<T>,
// }

// impl<T: MlxMapValue> Drop for StringToMap<T> {
//     fn drop(&mut self) {
//         unsafe {
//             let ptr = &self.c_map as *const T::MapType as *mut c_void;
//             mlx_sys::mlx_free(ptr)
//         }
//     }
// }

// impl<T: MlxMapValue> StringToMap<T> {
//     pub(crate) fn from_ptr(c_map: T::MapType) -> Self {
//         Self {
//             c_map,
//             _phantom: PhantomData,
//         }
//     }

//     pub(crate) fn as_ptr(&self) -> T::MapType
//     where
//         T::MapType: Copy,
//     {
//         self.c_map
//     }

//     pub(crate) fn as_hash_map(&self) -> HashMap<String, T>
//     where
//         T::MapType: Copy,
//         T::Iterator: Copy,
//     {
//         let mut result = HashMap::new();

//         unsafe {
//             let iterator = T::mlx_map_iterate(self.as_ptr());

//             while !T::mlx_map_iterator_end(iterator) {
//                 let key_ptr = T::mlx_map_iterator_key(iterator);
//                 let key = CStr::from_ptr(mlx_sys::mlx_string_data(key_ptr))
//                     .to_string_lossy()
//                     .into_owned();

//                 let value = T::from_mlx_ptr(T::mlx_map_iterator_value(iterator));
//                 result.insert(key, value);

//                 T::mlx_map_iterator_next(iterator);
//             }

//             let ptr = &iterator as *const T::Iterator as *mut c_void;
//             mlx_sys::mlx_free(ptr);
//         }

//         result
//     }
// }

// impl<T: MlxMapValue> TryFrom<&HashMap<String, T>> for StringToMap<T>
// where
//     T::MapType: Copy,
// {
//     type Error = IoError;

//     fn try_from(hashmap: &HashMap<String, T>) -> Result<Self, Self::Error> {
//         let mlx_map = T::mlx_map_new();

//         for (key, value) in hashmap {
//             let mlx_key = MlxString::try_from(key.as_str()).unwrap();
//             let success = T::mlx_map_insert(mlx_map, mlx_key.as_ptr(), value.as_mlx_ptr());
//             if !success {
//                 let ptr = &mlx_map as *const T::MapType as *mut c_void;
//                 unsafe { mlx_sys::mlx_free(ptr) };

//                 return Err(IoError::AllocationError);
//             }
//         }

//         Ok(Self::from_ptr(mlx_map))
//     }
// }

// pub(crate) type StringToArrayMap = StringToMap<Array>;
// pub(crate) type StringToStringMap = StringToMap<String>;
