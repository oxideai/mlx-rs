use crate::array::shape::Shape;

mod kind;
pub mod ops;
mod shape;
mod wrapper;

pub struct MLXArray<E: kind::Element, const D: usize> {
    pub tensor: wrapper::Array,
    phantom: std::marker::PhantomData<E>,
}

impl<E: kind::Element, const D: usize> MLXArray<E, D> {
    pub fn eval(&mut self) {
        self.tensor.eval();
    }

    pub fn shape(&self) -> Shape<D> {
        Shape::from(self.tensor.shape())
    }

    pub fn as_slice(&self) -> Option<&[E]> {
        self.tensor.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamOrDevice;

    #[test]
    fn test_shape() {
        let array: MLXArray<f32, 2> = MLXArray::zeros([2, 3], StreamOrDevice::default());
        assert_eq!(array.shape().dims, [2, 3]);
    }
}
