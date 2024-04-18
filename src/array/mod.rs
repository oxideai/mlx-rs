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

    pub fn as_slice(&self) -> Option<&[E]> {
        self.tensor.as_slice()
    }
}
