#[repr(C, u8)]
pub enum Optional<T> {
    None,
    Some(T),
}

impl<T> Default for Optional<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> From<Option<T>> for Optional<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(t) => Self::Some(t),
            None => Self::None,
        }
    }
}

// TODO: remove unused ffi mod
#[cxx::bridge]
pub mod ffi {

}

pub mod array;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod fft;
pub mod linalg;
pub mod ops;
pub mod random;
pub mod stream;
pub mod transforms;
pub mod types;
pub mod utils;
pub mod compat;
pub mod compile;
pub mod fast;

pub mod macros;
