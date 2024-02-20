use mlx_sys::{ops::ffi::*, dtype::ffi::*};

#[test]
fn test_arange_start_stop_dtype() {
    let dtype = dtype_float32();
    let _arr = arange_start_stop_dtype(0.0, 10.0, dtype, Default::default()).unwrap();
}