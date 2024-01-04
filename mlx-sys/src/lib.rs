#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/mlx.h");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx::core"]
        type array;

        #[namespace = "mlx::core::random"]
        fn seed(seed: u64);

        #[namespace = "mlx_cxx"]
        fn hello();
    }
}

#[cfg(test)]
mod tests {
    use crate::ffi;

    #[test]
    fn it_works() {
        ffi::hello();
    }
}