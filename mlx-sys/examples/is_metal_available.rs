fn main() {
    let mut is_available = false;
    let status = unsafe { mlx_sys::mlx_metal_is_available(&mut is_available as *mut bool) };
    assert_eq!(status, 0);
    println!("{:?}", is_available);
}
