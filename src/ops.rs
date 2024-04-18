use crate::{array::Array, stream::StreamOrDevice};

impl Array {
    pub fn zeros(shape: &[usize], dtype: crate::dtype::Dtype, stream: StreamOrDevice) -> Array {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let ctx = stream.as_ptr();

        unsafe { Array::from_ptr( mlx_sys::mlx_zeros(shape.as_ptr(), shape.len(), dtype.into(), ctx)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;

    #[test]
    fn test_zeros() {
        let mut array = Array::zeros(&[2, 3], Dtype::Float32, StreamOrDevice::default());
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Float32);

        array.eval();
        let data: &[f32] = array.as_slice().unwrap();
        assert_eq!(data, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}