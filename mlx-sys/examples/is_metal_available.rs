fn main() {
    let is_available = mlx_sys::backend::metal::ffi::is_available();
    println!("{:?}", is_available);
}
