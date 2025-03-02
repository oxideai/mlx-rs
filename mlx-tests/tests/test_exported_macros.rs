//! This contains the tests for some of the exported macros.
//! 
//! This is mainly a sanity check to ensure that the exported macros are working as expected.

use mlx_rs::{abs, array};

#[test]
fn test_ops_arithmetic_abs() {
    let data = array!([1i32, 2, -3, -4, -5]);
    let result = abs!(data).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5]);
}