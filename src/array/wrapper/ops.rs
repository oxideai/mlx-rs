use crate::{
    array::{kind::Kind, wrapper::Array},
    stream::StreamOrDevice,
};

impl Array {
    pub fn zeros(shape: &[usize], kind: Kind, stream: StreamOrDevice) -> Array {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let ctx = stream.as_ptr();

        unsafe {
            Array::from_ptr(mlx_sys::mlx_zeros(
                shape.as_ptr(),
                shape.len(),
                kind.into(),
                ctx,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let mut array = Array::zeros(&[2, 3], Kind::Float32, StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Kind::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[0.0; 6]);
    }
}
