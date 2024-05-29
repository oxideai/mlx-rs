#[macro_export]
macro_rules! as_type {
    ($a:expr, dtype=$dtype:ty) => {
        $a.as_type::<$dtype>()
    };
    ($a:expr, dtype=$dtype:ty, stream=$stream:expr) => {
        $a.as_type_device::<$dtype>($stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_as_type() {
        let stream = StreamOrDevice::default();
        let a = Array::from_slice(&[1, 2, 3], &[3]);

        // We are just testing that the macro compiles
        let _ = as_type!(a, dtype = f32);
        let _ = as_type!(a, dtype = f32, stream = stream);
    }
}
