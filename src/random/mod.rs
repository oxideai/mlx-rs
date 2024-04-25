use crate::{Array, StreamOrDevice};

mod state;

pub fn seed(seed: u64) {}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``random::*``
/// functions take an optional key -- this will let you control the
/// random number generation.
pub fn key(seed: u64) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_random_key(seed)) }
}

/// Split a PRNG key into two keys and return a tuple.
pub fn split(key: &Array, stream: StreamOrDevice) -> (Array, Array) {
    let keys = unsafe {
        Array::from_ptr(mlx_sys::mlx_random_split_equal_parts(
            key.as_ptr(),
            2,
            stream.as_ptr(),
        ))
    };

    // TODO: index 0 and 1 into keys
}