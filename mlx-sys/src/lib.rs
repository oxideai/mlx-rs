#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");

        #[namespace = "mlx::core::random"]
        fn seed(seed: u64);
    }
}