//! TODO: Are these supposed to be exposed? The header is not included in mlx/mlx.h

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/primitives.h");

        #[namespace = "mlx::core"]
        type Primitive;
    }
}