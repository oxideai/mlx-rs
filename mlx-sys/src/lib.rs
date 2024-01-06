#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {

    }
}

#[cxx::bridge]
mod array {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/array.hpp");

        #[namespace = "mlx::core"]
        type array;

        #[namespace = "mlx_cxx"]
        fn hello();

        #[namespace = "mlx_cxx"]
        fn array_new_bool(value: bool) -> UniquePtr<array>;
    }

}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_works() {
        array::hello();
    }

    #[test]
    fn test_array_new_bool() {
        let array = array::array_new_bool(true);
        assert!(!array.is_null());
    }
}