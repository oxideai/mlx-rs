fn main() {
    let is_available = unsafe { mlx_sys::mlx_metal_is_available() };
    println!("{:?}", is_available);
}
