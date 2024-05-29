/// See [`crate::Array::take`] for more information.
#[macro_export]
macro_rules! take {
    ($a:expr, $indices:expr) => {
        $a.take($indices.as_ref(), None)
    };
    ($a:expr, $indices:expr, $stream:expr) => {
        $a.take_device($indices.as_ref(), $stream)
    };
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn test_take() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let indices = Array::from_slice(&[0, 2, 4], &[3]);
        let _result = take!(a, indices);
    }
}