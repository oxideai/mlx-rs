fn main() {
    let arr = mlx_sys::array::ffi::array_new_f32(1.0);
    let dsize = arr.data_size();
    println!("{:?}", dsize);
}