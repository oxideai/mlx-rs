mod array;
mod factory;

use crate::{Array, StreamOrDevice};
pub use array::*;
use mlx_macros::default_device;

/// Element-wise addition.
///
/// Add two arrays with <doc:broadcasting>.
#[default_device]
pub fn add_device(a: &Array, b: &Array, stream: StreamOrDevice) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_add(a.c_array, b.c_array, stream.as_ptr())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_device() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = add(&a, &b);
        c.eval();

        let c_data: &[f32] = c.as_slice().unwrap();
        assert_eq!(c_data, &[5.0, 7.0, 9.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice().unwrap();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice().unwrap();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }
}
