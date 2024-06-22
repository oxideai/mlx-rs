//! Macros for creating arrays.

/// A helper macro to create an array with up to 3 dimensions.
///
/// Please note that this macro will always create non-scalar arrays. If you need to create a scalar
/// array, you should use the corresponding function like [`crate::Array::from_float`].
///
/// # Examples
///
/// ```rust
/// use mlx_rs::array;
///
/// // Create a 1D array
/// let a1 = array![1, 2, 3];
///
/// // Create a 2D array
/// let a2 = array![
///     [1, 2, 3],
///     [4, 5, 6]
/// ];
///
/// // Create a 3D array
/// let a3 = array![
///     [
///         [1, 2, 3],
///         [4, 5, 6]
///     ],
///     [
///         [7, 8, 9],
///         [10, 11, 12]
///     ]
/// ];
/// ```
#[macro_export]
macro_rules! array {
    // Empty array default to f32
    () => {
        $crate::Array::from_slice::<f32>(&[], &[0])
    };
    ($([$([$($x:expr),*]),*]),*) => {
        {
            let arr = [$([$([$($x,)*],)*],)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
    ($([$($x:expr),*]),*) => {
        {
            let arr = [$([$($x,)*],)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
    ($($x:expr),*) => {
        {
            let arr = [$($x,)*];
            <$crate::Array as $crate::FromNested<_>>::from_nested(arr)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::ops::indexing::IndexOp;

    #[test]
    fn test_array_1d() {
        let arr = array![1, 2, 3];

        assert!(arr.ndim() == 1);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.index(0).item::<i32>(), 1);
        assert_eq!(arr.index(1).item::<i32>(), 2);
        assert_eq!(arr.index(2).item::<i32>(), 3);
    }

    #[test]
    fn test_array_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];

        assert!(a.ndim() == 2);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.index((0, 0)).item::<i32>(), 1);
        assert_eq!(a.index((0, 1)).item::<i32>(), 2);
        assert_eq!(a.index((0, 2)).item::<i32>(), 3);
        assert_eq!(a.index((1, 0)).item::<i32>(), 4);
        assert_eq!(a.index((1, 1)).item::<i32>(), 5);
        assert_eq!(a.index((1, 2)).item::<i32>(), 6);
    }

    #[test]
    fn test_array_3d() {
        let a = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];

        assert!(a.ndim() == 3);
        assert_eq!(a.shape(), &[2, 2, 3]);
        assert_eq!(a.index((0, 0, 0)).item::<i32>(), 1);
        assert_eq!(a.index((0, 0, 1)).item::<i32>(), 2);
        assert_eq!(a.index((0, 0, 2)).item::<i32>(), 3);
        assert_eq!(a.index((0, 1, 0)).item::<i32>(), 4);
        assert_eq!(a.index((0, 1, 1)).item::<i32>(), 5);
        assert_eq!(a.index((0, 1, 2)).item::<i32>(), 6);
        assert_eq!(a.index((1, 0, 0)).item::<i32>(), 7);
        assert_eq!(a.index((1, 0, 1)).item::<i32>(), 8);
        assert_eq!(a.index((1, 0, 2)).item::<i32>(), 9);
        assert_eq!(a.index((1, 1, 0)).item::<i32>(), 10);
        assert_eq!(a.index((1, 1, 1)).item::<i32>(), 11);
        assert_eq!(a.index((1, 1, 2)).item::<i32>(), 12);
    }
}
