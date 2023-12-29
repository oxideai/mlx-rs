#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");

        fn say_something();

        #[namespace = "mlx::core::random"]
        fn seed(seed: u64);
    }
}

fn main() {
    ffi::say_something();
}