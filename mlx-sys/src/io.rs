//! TODO: maybe leave IO entirely to rust?

#[cxx::bridge(namespace = "mlx::core::io")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/io/load.h");

        type Reader;

        type Writer;

        type FileReader;

        type FileWriter;
    }
}
