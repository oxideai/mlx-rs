use mlx_sys::array::ffi::*;

#[test]
fn test_array_new_bool() {
    let mut array = array_new_bool(false);
}

#[test]
fn test_array_new_f32() {
    let mut array = array_new_f32(1.0);
}