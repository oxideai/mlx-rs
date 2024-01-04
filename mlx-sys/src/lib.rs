#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");

        #[namespace = "mlx::core"]
        type array;

        #[namespace = "mlx::core::random"]
        fn seed(seed: u64);

        #[namespace = "mlx::core::random"]
        fn key(seed: u64) -> array;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        unsafe {
            let a = crate::ffi::key(0);
        }
    }

}