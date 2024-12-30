/// Asserts that two arrays are equal.
/// 
/// It checks that the two arrays have the same shape and that all elements are
/// sufficiently close.
#[macro_export]
macro_rules! assert_array_eq {
    ($value:expr, $expected:expr) => {
        assert_array_eq!($value, $expected, None);
    };
    ($value:expr, $expected:expr, $atol:expr) => {
        assert_eq!($value.shape(), $expected.shape(), "Shapes are not equal");
        let assert = $value.all_close(&$expected, $atol, $atol, None);
        assert!(
            assert.unwrap().item::<bool>(),
            "Values are not sufficiently close"
        );
    };
}
