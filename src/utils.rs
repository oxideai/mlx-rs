use crate::array::Array;

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

/// Helper method to create a mlx_closure from a Rust closure.
pub(crate) fn new_mlx_closure<F>(closure: F) -> mlx_sys::mlx_closure
where
    F: 'static + Fn(&[Array]) -> Vec<Array>,
{
    // Box the closure to keep it on the heap
    let closure_box = Box::new(closure);

    // Create a raw pointer from the Box, transferring ownership to C
    let payload = Box::into_raw(Box::new(closure_box)) as *mut std::ffi::c_void;

    unsafe { mlx_sys::mlx_closure_new_with_payload(Some(trampoline), payload, Some(free_closure)) }
}

/// Function to create a new (+1 reference) mlx_vector_array from a vector of Array
fn new_mlx_vector_array(arrays: Vec<Array>) -> mlx_sys::mlx_vector_array {
    unsafe {
        let result = mlx_sys::mlx_vector_array_new();
        let ctx_ptrs: Vec<mlx_sys::mlx_array> = arrays.iter().map(|array| array.ctx()).collect();
        mlx_sys::mlx_vector_array_add_arrays(result, ctx_ptrs.as_ptr(), arrays.len());
        result
    }
}

fn mlx_vector_array_values(vector_array: mlx_sys::mlx_vector_array) -> Vec<Array> {
    unsafe {
        let size = mlx_sys::mlx_vector_array_size(vector_array);
        (0..size)
            .map(|index| {
                // ctx is a +1 reference, the array takes ownership
                let ctx = mlx_sys::mlx_vector_array_get(vector_array, index);
                Array::new(ctx)
            })
            .collect()
    }
}

extern "C" fn trampoline(
    vector_array: mlx_sys::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> mlx_sys::mlx_vector_array {
    unsafe {
        let closure: &mut Box<dyn Fn(&[Array]) -> Vec<Array>> = &mut *(payload as *mut _);
        let arrays = mlx_vector_array_values(vector_array);
        let result = closure(&arrays);
        new_mlx_vector_array(result)
    }
}

extern "C" fn free_closure(payload: *mut std::ffi::c_void) {
    unsafe {
        let _dropped_box: Box<Box<dyn Fn(&[Array]) -> Vec<Array>>> =
            Box::from_raw(payload as *mut _);
    }
}
