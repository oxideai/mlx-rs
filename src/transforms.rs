use crate::{
    error::{
        get_and_clear_last_mlx_error, is_mlx_error_handler_set, setup_mlx_error_handler, Exception,
    },
    utils::VectorArray,
    Array,
};

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a mut Array>) -> Result<(), Exception> {
    if !is_mlx_error_handler_set() {
        setup_mlx_error_handler();
    }

    let vec = VectorArray::from_iter(outputs.into_iter());

    unsafe {
        mlx_sys::mlx_eval(vec.as_ptr());
    }

    get_and_clear_last_mlx_error().map_or(Ok(()), Err)
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a mut Array>) -> Result<(), Exception> {
    if !is_mlx_error_handler_set() {
        setup_mlx_error_handler();
    }

    let vec = VectorArray::from_iter(outputs.into_iter());

    unsafe {
        mlx_sys::mlx_async_eval(vec.as_ptr());
    }

    get_and_clear_last_mlx_error().map_or(Ok(()), Err)
}
