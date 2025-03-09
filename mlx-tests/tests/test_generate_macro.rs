#![allow(unused_variables)]

use mlx_internal_macros::{default_device, generate_macro};
use mlx_rs::{Stream, StreamOrDevice};

// Test generate_macro for functions with no generic type arguments.
#[generate_macro(customize(root = "$crate"))]
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

// Test generate_macro for functions with generic type arguments.
#[generate_macro(customize(
    root = "$crate",
    default_dtype = i32,
))]
#[default_device]
fn bar_device<T: Into<i32>>(
    a: T,                                  // Mandatory argument
    b: T,                                  // Mandatory argument
    #[optional] c: Option<T>,              // Optional argument
    #[optional] d: impl Into<Option<T>>,   // Optional argument but impl Trait
    #[optional] stream: impl AsRef<Stream>, // stream always optional and placed at the end
) -> i32 {
    let a = a.into();
    let b = b.into();
    let c = c.map(Into::into);
    let d = d.into().map(Into::into);
    a + b + c.unwrap_or(0) + d.unwrap_or(0)
}

#[test]
fn test_bar() {
    // Without specifying dtype, the default is i32.

    let result = bar!(1, 2);
    assert_eq!(result, 3);
    
    let result = bar!(1, 2, c = Some(3));
    assert_eq!(result, 6);
    
    let result = bar!(1, 2, d = Some(4));
    assert_eq!(result, 7);
    
    let result = bar!(1, 2, c = Some(3), d = Some(4));
    assert_eq!(result, 10);

    // With dtype specified as i16.

    let result = bar!(1, 2, dtype=i16);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), dtype=i16);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), dtype=i16);
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4), dtype=i16);
    assert_eq!(result, 10);

    // With stream specified.

    let stream = Stream::new();

    let result = bar!(1, 2, stream = &stream);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), stream = &stream);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), stream = &stream);
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4), stream = &stream);
    assert_eq!(result, 10);

    // With dtype and stream specified.

    let result = bar!(1, 2, dtype=i16, stream = &stream);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), dtype=i16, stream = &stream);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), dtype=i16, stream = &stream);
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4), dtype=i16, stream = &stream);
    assert_eq!(result, 10);
}

// Test named mandatory arguments.
#[generate_macro(customize(root = "$crate"))]
#[default_device]
fn baz_device(
    #[optional] a: Option<i32>,                         // Optinal argument
    #[named] b: i32,                                    // Mandatory argument
    #[optional] c: Option<i32>,                         // Optional argument
    #[optional] stream: impl AsRef<Stream>,             // stream always optional and placed at the end
) -> i32 {
    a.unwrap_or(0) + b + c.unwrap_or(0)
}

#[test]
fn test_baz() {
    assert_eq!(baz!(b = 1), 1);
    assert_eq!(baz!(a = Some(2), b = 1), 3);
    assert_eq!(baz!(b = 1, c = Some(3)), 4);
    assert_eq!(baz!(a = Some(2), b = 1, c = Some(3)), 6);

    let stream = Stream::new();

    assert_eq!(baz!(b = 1, stream = &stream), 1);
    assert_eq!(baz!(a = Some(2), b = 1, stream = &stream), 3);
    assert_eq!(baz!(b = 1, c = Some(3), stream = &stream), 4);
    assert_eq!(baz!(a = Some(2), b = 1, c = Some(3), stream = &stream), 6);
}