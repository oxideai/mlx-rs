use cxx::CxxVector;

#[test]
fn test_push_array() {
    let mut vec = CxxVector::new();
    let arr1 = mlx_sys::array::ffi::array_new_bool(true);
    mlx_sys::utils::ffi::push_array(vec.pin_mut(), arr1);
}