use crate::error::IOError;
use crate::utils::{
    io::{FilePtr, SafeTensors, StringToArrayMap, StringToStringMap},
    MlxString,
};
use crate::{Array, Stream, StreamOrDevice};
use mlx_internal_macros::default_device;
use std::collections::HashMap;
use std::path::Path;

fn prepare_file_path(path: &Path) -> Result<MlxString, IOError> {
    if !path.is_file() {
        return Err(IOError::NotFile);
    }

    let path_str = path.to_str().ok_or(IOError::InvalidUtf8)?;
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
///
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
///
/// - path: path of file to load
/// - stream: stream or device to evaluate on
///
#[default_device]
pub fn load_arrays_device(
    path: &Path,
    stream: impl AsRef<Stream>,
) -> Result<HashMap<String, Array>, IOError> {
    let safetensors = SafeTensors::load_device(path, stream)?;
    let data = safetensors.data()?;
    Ok(data)
}

/// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
///
/// # Params
///
/// - path: path of file to load
/// - stream: stream or device to evaluate on
#[allow(clippy::type_complexity)]
#[default_device]
pub fn load_arrays_with_metadata_device(
    path: &Path,
    stream: impl AsRef<Stream>,
) -> Result<(HashMap<String, Array>, HashMap<String, String>), IOError> {
    let safetensors = SafeTensors::load_device(path, stream)?;
    let data = safetensors.data()?;
    let metadata = safetensors.metadata()?;

    Ok((data, metadata))
}

/// Save array to a binary file in `.npy`format.
///
/// # Params
///
/// - array: array to save
/// - url: URL of file to load
pub fn save_array(array: &Array, path: &Path) -> Result<(), IOError> {
    check_file_extension(path, "npy")?;
    let file_ptr = FilePtr::open(path, "w")?;

    unsafe { mlx_sys::mlx_save_file(file_ptr.as_ptr(), a.as_ptr()) };

    Ok(())
}

/// Save dictionary of arrays in `safetensors` format.
///
/// # Params
///
/// - arrays: arrays to save
/// - metadata: metadata to save
/// - path: path of file to save
/// - stream: stream or device to evaluate on
pub fn save_arrays<'a>(
    arrays: &HashMap<String, Array>,
    metadata: impl Into<Option<&'a HashMap<String, String>>>,
    path: &Path,
) -> Result<(), IOError> {
    check_file_extension(path, "safetensors")?;

    let arrays = StringToArrayMap::try_from(arrays)?;

    let default_metadata = HashMap::new();
    let metadata_ref = metadata.into().unwrap_or(&default_metadata);
    let metadata = StringToStringMap::try_from(metadata_ref)?;

    let file_ptr = FilePtr::open(path, "w")?;

    unsafe {
        mlx_sys::mlx_save_safetensors_file(file_ptr.as_ptr(), arrays.as_ptr(), metadata.as_ptr());
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ops::{load_array, load_arrays, save_array, save_arrays};
    use crate::Array;

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
            assert!(loaded_array
                .all_close(original_array, None, None, None)
                .unwrap()
                .item::<bool>());
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
