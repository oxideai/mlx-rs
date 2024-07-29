/// See [`crate::Array::take`] for more information.
#[macro_export]
macro_rules! take {
    ($a:expr, $indices:expr, $axis:expr) => {
        $a.take($indices.as_ref(), $axis)
    };
    ($a:expr, $indices:expr, $axis:expr, stream=$stream:expr) => {
        $a.take_device($indices.as_ref(), $axis, $stream)
    };
}

/// See [`crate::Array::take_all`] for more information.
#[macro_export]
macro_rules! take_all {
    ($a:expr, $indices:expr) => {
        $a.take_all($indices.as_ref())
    };
    ($a:expr, $indices:expr, stream=$stream:expr) => {
        $a.take_all_device($indices.as_ref(), $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn test_take() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let indices = Array::from_slice(&[0, 2, 4], &[3]);
        let _result = take!(a, indices, 0);

        let stream = crate::StreamOrDevice::default();
        let _result = take!(a, indices, 0, stream = &stream);
    }

    #[test]
    fn test_take_all() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let indices = Array::from_slice(&[0, 2, 4], &[3]);
        let _result = take_all!(a, indices);

        let stream = crate::StreamOrDevice::default();
        let _result = take_all!(a, indices, stream = &stream);
    }
}
