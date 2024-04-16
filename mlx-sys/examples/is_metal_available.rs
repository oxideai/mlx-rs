fn main() {
    let is_available = mlx_sys::is_metal_available();
    println!("{:?}", is_available);
}
