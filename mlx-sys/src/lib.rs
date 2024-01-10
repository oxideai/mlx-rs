#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {

    }
}

pub(crate) mod types;
pub(crate) mod array;
pub(crate) mod dtype;
pub(crate) mod macros;