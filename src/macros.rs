//! All internal macros

/// See `assertEqual` in the swift binding tests
macro_rules! assert_array_all_close {
    ($a:tt, $b:tt) => {
        let _b: Array = $b.into();
        let mut assert = $a.all_close(&_b, None, None, None);
        assert!(assert.item::<bool>());
    };
}
