#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/types/bf16.h");
        include!("mlx/types/complex.h");
        include!("mlx/types/fp16.h");
        include!("mlx/types/half_types.h");

        // include!("mlx-cxx/complext.hpp");
    }

    extern "Rust" {

    }
}