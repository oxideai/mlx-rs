/// Helper method to get a string representation of an mlx object.
pub(crate) fn mlx_describe(ptr: *mut ::std::os::raw::c_void) -> Option<String> {
    let mlx_description = unsafe { mlx_sys::mlx_tostring(ptr) };
    let c_str = unsafe { mlx_sys::mlx_string_data(mlx_description) };

    let description = if c_str.is_null() {
        None
    } else {
        Some(unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        })
    };

    unsafe { mlx_sys::mlx_free(mlx_description as *mut std::ffi::c_void) };

    description
}


pub(crate) fn resolve_index_unchecked(index: i32, len: usize) -> usize {
    if index.is_negative() {
        (len as i32 + index) as usize
    } else {
        index as usize
    }
}

/// `mlx` differs from `numpy` in the way it handles out of bounds indices. It's behavior is more
/// like `jax`. See the related issue here https://github.com/ml-explore/mlx/issues/206.
/// 
/// The issue says it would use the last element if the index is out of bounds. But testing with
/// python seems more like undefined behavior. Here we will use the last element if the index is
/// is out of bounds.
pub(crate) fn resolve_index(index: i32, len: usize) -> usize {
    let abs_index = index.abs() as usize;

    if index.is_negative() {
        if abs_index <= len {
            len - abs_index
        } else {
            len - 1
        }
    } else {
        if abs_index < len {
            abs_index
        } else {
            len - 1
        }
    }
}