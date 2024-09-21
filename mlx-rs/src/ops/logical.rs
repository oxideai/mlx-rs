use crate::array::Array;
use crate::error::Exception;
use crate::stream::StreamOrDevice;
use crate::utils::{axes_or_default_to_all, IntoOption};
use crate::Stream;
use mlx_macros::default_device;

impl Array {
    /// Element-wise equality returning an error if the arrays are not broadcastable.
    ///
    /// Equality comparison on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.eq(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    #[default_device]
    pub fn eq_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_equal(
                    self.as_ptr(),
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise less than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Less than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.le(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    #[default_device]
    pub fn le_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_less_equal(
                    self.as_ptr(),
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise greater than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Greater than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ge(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    #[default_device]
    pub fn ge_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_greater_equal(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise not equal returning an error if the arrays are not broadcastable.
    ///
    /// Not equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ne(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    #[default_device]
    pub fn ne_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_not_equal(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise less than returning an error if the arrays are not broadcastable.
    ///
    /// Less than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.lt(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    #[default_device]
    pub fn lt_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_less(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise greater than returning an error if the arrays are not broadcastable.
    ///
    /// Greater than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.gt(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    #[default_device]
    pub fn gt_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_greater(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise logical and returning an error if the arrays are not broadcastable.
    ///
    /// Logical and on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_and(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, false, false]
    /// ```
    #[default_device]
    pub fn logical_and_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_logical_and(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise logical or returning an error if the arrays are not broadcastable.
    ///
    /// Logical or on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_or(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    #[default_device]
    pub fn logical_or_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_logical_or(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Unary element-wise logical not.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a: Array = false.into();
    /// let mut b = a.logical_not_device(StreamOrDevice::default());
    ///
    /// let b_data: &[bool] = b.as_slice();
    /// // b_data == [true]
    /// ```
    #[default_device]
    pub fn logical_not_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_logical_not(
                self.c_array,
                stream.as_ref().as_ptr(),
            ))
        }
    }

    /// Approximate comparison of two arrays returning an error if the inputs aren't valid.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```text
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - rtol: relative tolerance = defaults to 1e-5 when None
    /// - atol: absolute tolerance - defaults to 1e-8 when None
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    ///
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx_rs::array;
    /// let a = array!([0., 1., 2., 3.]).sqrt();
    /// let b = array!([0., 1., 2., 3.]).power(array!(0.5)).unwrap();
    /// let mut c = a.all_close(&b, None, None, None).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true]
    /// ```
    #[default_device]
    pub fn all_close_device(
        &self,
        other: impl AsRef<Array>,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_allclose(
                    self.c_array,
                    other.as_ref().as_ptr(),
                    rtol.into().unwrap_or(1e-5),
                    atol.into().unwrap_or(1e-8),
                    equal_nan.into().unwrap_or(false),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Returns a boolean array where two arrays are element-wise equal within a tolerance returning an error if the arrays are not broadcastable.
    ///
    /// Infinite values are considered equal if they have the same sign, NaN values are not equal unless
    /// `equalNAN` is `true`.
    ///
    /// Two values are considered close if:
    ///
    /// ```text
    /// abs(a - b) <= (atol + rtol * abs(b))
    /// ```
    ///
    /// Unlike [self.array_eq] this function supports [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    #[default_device]
    pub fn is_close_device(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_isclose(
                    self.c_array,
                    other.c_array,
                    rtol.into().unwrap_or(1e-5),
                    atol.into().unwrap_or(1e-8),
                    equal_nan.into().unwrap_or(false),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `true` iff the arrays have
    /// the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
    ///
    /// let c = a.array_eq(&b, None);
    /// // c == [true]
    /// ```
    #[default_device]
    pub fn array_eq_device(
        &self,
        other: impl AsRef<Array>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_array_equal(
                self.c_array,
                other.as_ref().as_ptr(),
                equal_nan.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        }
    }

    /// An `or` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: if `true` keep reduced axis as singleton dimension -- defaults to false if not provided
    ///
    ///  # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    ///
    /// // will produce a scalar Array with true -- some of the values are non-zero
    /// let all = array.any(None, None).unwrap();
    ///
    /// // produces an Array([true, true, true, true]) -- all rows have non-zeros
    /// let all_rows = array.any(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn any_device<'a>(
        &self,
        axes: impl IntoOption<&'a [i32]>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_any(
                    self.c_array,
                    axes.as_ptr(),
                    axes.len(),
                    keep_dims.into().unwrap_or(false),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }
}

/// See [`Array::any`]
#[default_device]
pub fn any_device<'a>(
    array: &Array,
    axes: impl IntoOption<&'a [i32]>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    array.any_device(axes, keep_dims, stream)
}

/// See [`Array::logical_and`]
#[default_device]
pub fn logical_and_device(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array, Exception> {
    a.as_ref().logical_and_device(b, StreamOrDevice::default())
}

/// See [`Array::logical_or`]
#[default_device]
pub fn logical_or_device(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array, Exception> {
    a.as_ref().logical_or_device(b, StreamOrDevice::default())
}

/// See [`Array::logical_not`]
#[default_device]
pub fn logical_not_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    a.logical_not_device(stream)
}

/// See [`Array::all_close`]
#[default_device]
pub fn all_close_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    rtol: impl Into<Option<f64>>,
    atol: impl Into<Option<f64>>,
    equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref()
        .all_close_device(b, rtol, atol, equal_nan, stream)
}

/// See [`Array::is_close`]
#[default_device]
pub fn is_close_device(
    a: &Array,
    b: &Array,
    rtol: impl Into<Option<f64>>,
    atol: impl Into<Option<f64>>,
    equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.is_close_device(b, rtol, atol, equal_nan, stream)
}

/// See [`Array::array_eq`]
#[default_device]
pub fn array_eq_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    a.as_ref().array_eq_device(b, equal_nan, stream)
}

/// See [`Array::eq`]
#[default_device]
pub fn eq_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().eq_device(b, stream)
}

/// See [`Array::le`]
#[default_device]
pub fn le_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().le_device(b, stream)
}

/// See [`Array::ge`]
#[default_device]
pub fn ge_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().ge_device(b, stream)
}

/// See [`Array::ne`]
#[default_device]
pub fn ne_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().ne_device(b, stream)
}

/// See [`Array::lt`]
#[default_device]
pub fn lt_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().lt_device(b, stream)
}

/// See [`Array::gt`]
#[default_device]
pub fn gt_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.as_ref().gt_device(b, stream)
}

// TODO: check if the functions below could throw an exception.

/// Return a boolean array indicating which elements are NaN.
#[default_device]
pub fn is_nan_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_isnan(array.c_array, stream.as_ref().as_ptr())) }
}

/// Return a boolean array indicating which elements are +/- inifnity.
#[default_device]
pub fn is_inf_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_isinf(array.c_array, stream.as_ref().as_ptr())) }
}

/// Return a boolean array indicating which elements are positive infinity.
#[default_device]
pub fn is_pos_inf_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_isposinf(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

/// Return a boolean array indicating which elements are negative infinity.
#[default_device]
pub fn is_neg_inf_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_isneginf(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

/// Select from `a` or `b` according to `condition` returning an error if the arrays are not
/// broadcastable.
///
/// The condition and input arrays must be the same shape or
/// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting)
/// with each another.
///
/// # Params
///
/// - condition: condition array
/// - a: input selected from where condition is non-zero or `true`
/// - b: input selected from where condition is zero or `false`
#[default_device]
pub fn r#where_device(
    condition: &Array,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_where(
                condition.c_array,
                a.as_ref().as_ptr(),
                b.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

/// Alias for [`r#where`]
#[default_device]
pub fn which_device(
    condition: &Array,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_where(
                condition.c_array,
                a.as_ref().as_ptr(),
                b.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[cfg(test)]
mod tests {
    use crate::{array, Dtype};

    use super::*;

    #[test]
    fn test_eq() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.eq(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_eq_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.eq(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_le() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.le(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_le_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.le(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ge() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.ge(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_ge_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.ge(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ne() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.ne(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_ne_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.ne(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_lt() {
        let a = Array::from_slice(&[1, 0, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.lt(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, true, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 0, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_lt_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.lt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_gt() {
        let a = Array::from_slice(&[1, 4, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.gt(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, true, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 4, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_gt_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.gt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_and() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let c = a.logical_and(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, false, false]);

        // check a and b are not modified
        let a_data: &[bool] = a.as_slice();
        assert_eq!(a_data, [true, false, true]);

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true, true, false]);
    }

    #[test]
    fn test_logical_and_invalid_broadcast() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false, true], &[4]);
        let c = a.logical_and(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_or() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let c = a.logical_or(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[bool] = a.as_slice();
        assert_eq!(a_data, [true, false, true]);

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true, true, false]);
    }

    #[test]
    fn test_logical_or_invalid_broadcast() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false, true], &[4]);
        let c = a.logical_or(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_all_close() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4])
            .power(array!(0.5))
            .unwrap();
        let c = a.all_close(&b, 1e-5, None, None).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_all_close_invalid_broadcast() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3., 4.], &[5]);
        let c = a.all_close(&b, 1e-5, None, None);
        assert!(c.is_err());
    }

    #[test]
    fn test_is_close_false() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let c = a.is_close(&b, None, None, false).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);
    }

    #[test]
    fn test_is_close_true() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let c = a.is_close(&b, 0.1, 0.2, true).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);
    }

    #[test]
    fn test_is_close_invalid_broadcast() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3, 4.4], &[4]);
        let c = a.is_close(&b, None, None, false);
        assert!(c.is_err());
    }

    #[test]
    fn test_array_eq() {
        let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let c = a.array_eq(&b, None);

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_any() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.any(&[0][..], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(results, &[true, true, true, true]);
    }

    #[test]
    fn test_any_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.any(&[][..], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_any_out_of_bounds() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[12]);
        let result = array.any(&[1][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_duplicate_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let result = array.any(&[0, 0][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_which() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);
        let c = which(&condition, &a, &b).unwrap();

        let c_data: &[i32] = c.as_slice();
        assert_eq!(c_data, [1, 5, 3]);
    }

    #[test]
    fn test_which_invalid_broadcast() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6, 7], &[4]);
        let c = which(&condition, &a, &b);
        assert!(c.is_err());
    }

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_unary_logical_not() {
        let x = array!(false);
        assert!(logical_not(&x).item::<bool>());

        let x = array!(1.0);
        let y = logical_not(&x);
        assert_eq!(y.dtype(), Dtype::Bool);
        assert!(!y.item::<bool>());

        let x = array!(0);
        let y = logical_not(&x);
        assert_eq!(y.dtype(), Dtype::Bool);
        assert!(y.item::<bool>());
    }

    #[test]
    fn test_unary_logical_and() {
        let x = array!(true);
        let y = array!(true);
        assert!(logical_and(&x, &y).unwrap().item::<bool>());

        let x = array!(1.0);
        let y = array!(1.0);
        let z = logical_and(&x, &y).unwrap();
        assert_eq!(z.dtype(), Dtype::Bool);
        assert!(z.item::<bool>());

        let x = array!(0);
        let y = array!(1.0);
        let z = logical_and(&x, &y).unwrap();
        assert_eq!(z.dtype(), Dtype::Bool);
        assert!(!z.item::<bool>());
    }

    #[test]
    fn test_unary_logical_or() {
        let a = array!(false);
        let b = array!(false);
        assert!(!logical_or(&a, &b).unwrap().item::<bool>());

        let a = array!(1.0);
        let b = array!(1.0);
        let c = logical_or(&a, &b).unwrap();
        assert_eq!(c.dtype(), Dtype::Bool);
        assert!(c.item::<bool>());

        let a = array!(0);
        let b = array!(1.0);
        let c = logical_or(&a, &b).unwrap();
        assert_eq!(c.dtype(), Dtype::Bool);
        assert!(c.item::<bool>());
    }
}
