use autocxx::WithinUniquePtr;

fn main() {
    let s = mlx_sys::ext::hello();
    println!("{}", s);

    let a = mlx_sys::ext::array::new_scalar_array_bool(true).within_unique_ptr();
}