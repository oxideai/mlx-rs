#![allow(unused_variables)]

use mlx_internal_macros::{default_device, generate_macro};
use mlx_rs::{Stream, StreamOrDevice};

/// Test macro generation.
#[generate_macro]
#[default_device]
fn foo_device(
    a: i32,                                 // Mandatory argument
    b: i32,                                 // Mandatory argument
    #[optional] c: Option<i32>,             // Optional argument
    #[optional] d: impl Into<Option<i32>>,  // Optional argument but impl Trait
    #[optional] stream: impl AsRef<Stream>, // stream always optional and placed at the end
) -> i32 {
    a + b + c.unwrap_or(0) + d.into().unwrap_or(0)
}

#[test]
fn test_foo() {
    assert_eq!(foo!(1, 2), 3);
    assert_eq!(foo!(1, 2, c = Some(3)), 6);
    assert_eq!(foo!(1, 2, d = Some(4)), 7);
    assert_eq!(foo!(1, 2, c = Some(3), d = Some(4)), 10);

    let stream = Stream::new();

    assert_eq!(foo!(1, 2, stream = &stream), 3);
    assert_eq!(foo!(1, 2, c = Some(3), stream = &stream), 6);
    assert_eq!(foo!(1, 2, d = Some(4), stream = &stream), 7);
    assert_eq!(foo!(1, 2, c = Some(3), d = Some(4), stream = &stream), 10);
}
