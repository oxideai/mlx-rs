#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {

    }
}

pub mod types;
pub mod array;
pub mod dtype;
pub mod macros;
pub mod backend;
pub mod device;
pub mod stream;