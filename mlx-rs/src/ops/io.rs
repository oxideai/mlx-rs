use crate::error::IOError;
use crate::utils::{
    mlx_map_array_values, mlx_map_string_values, new_mlx_array_map, new_mlx_string_map, MlxString,
};
use crate::{Array, Stream, StreamOrDevice};
use mlx_internal_macros::default_device;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;

fn prepare_file_path(path: &Path) -> Result<MlxString, IOError> {
    if !path.is_file() {
        return Err(IOError::NotFile);
    }

    let path_str = path.to_str().ok_or_else(|| IOError::InvalidUtf8)?;
    let path = MlxString::try_from(path_str).map_err(|_| IOError::NullBytes)?;

    Ok(path)
}

fn check_file_extension(path: &Path, expected: &str) -> Result<(), IOError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == expected => Ok(()),
        _ => Err(IOError::UnsupportedFormat),
    }
}

/// Load array from a binary file in `.npy` format.
///
/// # Params
/// - path: path of file to load
/// - stream: stream or device to evaluate on
#[default_device]
pub fn load_array_device(path: &Path, stream: impl AsRef<Stream>) -> Result<Array, IOError> {
    let mlx_path = prepare_file_path(path)?;
    check_file_extension(path, "npy")?;

    let load_result = (|| unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_load(mlx_path.as_ptr(), stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    })();

    match load_result {
        Ok(array) => Ok(array),
        Err(e) => Err(IOError::from(e)),
    }
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
    let mlx_path = prepare_file_path(path)?;
    check_file_extension(path, "safetensors")?;

    let load_result = (|| unsafe {
        let mlx_safetensors = try_catch_c_ptr_expr! {
            mlx_sys::mlx_load_safetensors(mlx_path.as_ptr(), stream.as_ref().as_ptr())
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
    let mlx_path = prepare_file_path(path)?;
    check_file_extension(path, "safetensors")?;

    let load_result = (|| unsafe {
        let mlx_safetensors = try_catch_c_ptr_expr! {
            mlx_sys::mlx_load_safetensors(mlx_path.as_ptr(), stream.as_ref().as_ptr())
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
}

/// Save array to a binary file in `.npy`format.
///
/// # Params
/// - a: array to save
/// - url: URL of file to load
pub fn save_array(a: &Array, path: &Path) -> Result<(), IOError> {
    check_file_extension(path, "npy")?;

    let path = CString::new(path.to_str().ok_or_else(|| IOError::InvalidUtf8)?)
        .map_err(|_| IOError::NullBytes)?;
    let mode = CString::new("w").map_err(|_| IOError::NullBytes)?;

    let file_ptr = unsafe { mlx_sys::fopen(path.as_ptr(), mode.as_ptr()) };
    if file_ptr.is_null() {
        return Err(IOError::UnableToOpenFile);
    }

    unsafe {
        mlx_sys::mlx_save_file(file_ptr, a.c_array);
        mlx_sys::fclose(file_ptr);
    }

    Ok(())
}

/// Save dictionary of arrays in `safetensors` format.
///
/// # Params
/// - a: array to save
/// - metadata: metadata to save
/// - url: URL of file to load
/// - stream: stream or device to evaluate on
pub fn save_arrays<'a>(
    arrays: &HashMap<String, Array>,
    metadata: impl Into<Option<&'a HashMap<String, String>>>,
    path: &Path,
) -> Result<(), IOError> {
    check_file_extension(path, "safetensors")?;

    let mlx_arrays = new_mlx_array_map(arrays);

    // Create an owned HashMap that lives for the duration of the function
    let default_metadata = HashMap::new();
    let metadata_ref = metadata.into().unwrap_or(&default_metadata);
    let mlx_metadata = new_mlx_string_map(metadata_ref);

    let path = CString::new(path.to_str().ok_or_else(|| IOError::InvalidUtf8)?)
        .map_err(|_| IOError::NullBytes)?;
    let mode = CString::new("w").map_err(|_| IOError::NullBytes)?;

    let file_ptr = unsafe { mlx_sys::fopen(path.as_ptr(), mode.as_ptr()) };
    if file_ptr.is_null() {
        return Err(IOError::UnableToOpenFile);
    }

    unsafe {
        mlx_sys::mlx_save_safetensors_file(file_ptr, mlx_arrays, mlx_metadata);
        mlx_sys::free(mlx_arrays as *mut c_void);
        mlx_sys::free(mlx_metadata as *mut c_void);
        mlx_sys::fclose(file_ptr);
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::Array;
    use crate::ops::{load_array, load_arrays, save_array, save_arrays};

    #[test]
    fn test_save_arrays() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.safetensors");

        let mut arrays = std::collections::HashMap::new();
        arrays.insert("foo".to_string(), Array::ones::<i32>(&[1, 2]).unwrap());
        arrays.insert("bar".to_string(), Array::zeros::<i32>(&[2, 1]).unwrap());

        save_arrays(&arrays, None, &path).unwrap();

        let loaded_arrays = load_arrays(&path).unwrap();

        // compare values
        let mut loaded_keys: Vec<_> = loaded_arrays.keys().cloned().collect();
        let mut original_keys: Vec<_> = arrays.keys().cloned().collect();
        loaded_keys.sort();
        original_keys.sort();
        assert_eq!(loaded_keys, original_keys);

        for key in loaded_keys {
            let loaded_array = loaded_arrays.get(&key).unwrap();
            let original_array = arrays.get(&key).unwrap();
            assert!(loaded_array.all_close(original_array, None, None, None).unwrap().item::<bool>());
        }
    }

    #[test]
    fn test_save_array() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.npy");

        let a = Array::ones::<i32>(&[2, 4]).unwrap();
        save_array(&a, &path).unwrap();

        let b = load_array(&path).unwrap();
        assert!(a.all_close(&b, None, None, None).unwrap().item::<bool>());
    }
}
