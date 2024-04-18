use crate::array::shape::Shape;
use crate::array::{kind, wrapper, MLXArray};
use crate::stream::StreamOrDevice;

impl<E: kind::Element, const D: usize> MLXArray<E, D> {
    pub fn zeros<S: Into<Shape<D>>>(shape: S, stream: StreamOrDevice) -> Self {
        let shape = shape.into();
        let tensor = wrapper::Array::zeros(&shape.dims, E::KIND, stream);

        Self {
            tensor,
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamOrDevice;

    #[test]
    fn test_zeros() {
        let mut array: MLXArray<f32, 2> = MLXArray::zeros([2, 3], StreamOrDevice::default());
        array.eval();
        let data = array.as_slice().unwrap();

        assert_eq!(data, &[0.0; 6]);
    }
}
