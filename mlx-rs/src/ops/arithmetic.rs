use crate::array::Array;
use crate::error::Result;
use crate::sealed::Sealed;

use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, ScalarOrArray, VectorArray};
use crate::Stream;
use mlx_internal_macros::{default_device, generate_macro};
use smallvec::SmallVec;

impl Array {
    /// Element-wise absolute value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
    /// let mut result = array.abs().unwrap();
    ///
    /// let data: &[i32] = result.as_slice();
    /// // data == [1, 2, 3, 4, 5]
    /// ```
    #[default_device]
    pub fn abs_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_abs(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise addition returning an error if arrays are not broadcastable.
    ///
    /// Add two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to add
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.add(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [5.0, 7.0, 9.0]
    /// ```
    #[default_device]
    pub fn add_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_add(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise subtraction returning an error if arrays are not broadcastable.
    ///
    /// Subtract two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to subtract
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.subtract(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [-3.0, -3.0, -3.0]
    /// ```
    #[default_device]
    pub fn subtract_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_subtract(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Unary element-wise negation. Returns an error if the array is of type bool.
    ///
    /// Negate the values in the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.negative().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [-1.0, -2.0, -3.0]
    /// ```
    #[default_device]
    pub fn negative_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_negative(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise multiplication returning an error if arrays are not broadcastable.
    ///
    /// Multiply two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.multiply(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [4.0, 10.0, 18.0]
    /// ```
    #[default_device]
    pub fn multiply_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_multiply(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Replace NaN and Inf values with finite numbers.
    ///
    /// # Params
    /// - nan: value to replace NaN with
    /// - posInf: value to replace positive inifinites with.  If not specified will use
    ///     the largest finite value for the given dtype.
    /// - negInf: value to replace negative inifinites with.  If not specified will use
    ///     the negative of the largest finite value for the given dtype.
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn nan_to_num_device(
        &self,
        nan: impl IntoOption<f32>,
        pos_inf: impl IntoOption<f32>,
        neg_inf: impl IntoOption<f32>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let pos_inf = pos_inf.into_option();
        let neg_inf = neg_inf.into_option();

        let pos_inf = mlx_sys::mlx_optional_float {
            value: pos_inf.unwrap_or(0.0),
            has_value: pos_inf.is_some(),
        };
        let neg_inf = mlx_sys::mlx_optional_float {
            value: neg_inf.unwrap_or(0.0),
            has_value: neg_inf.is_some(),
        };

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_nan_to_num(
                res,
                self.as_ptr(),
                nan.into_option().unwrap_or(0.),
                pos_inf,
                neg_inf,
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise division returning an error if arrays are not broadcastable.
    ///
    /// Divide two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to divide
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.divide(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    #[default_device]
    pub fn divide_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_divide(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise power operation returning an error if arrays are not broadcastable if they have different shapes.
    ///
    /// Raise the elements of the array to the power of the elements of another array.
    ///
    /// # Params
    ///
    /// - other: array to raise to the power of
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);
    /// let mut c = a.power(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 8.0, 81.0]
    /// ```
    #[default_device]
    pub fn power_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_power(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise remainder of division returning an error if arrays are not broadcastable.
    ///
    /// Computes the remainder of dividing `lhs` with `rhs` with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to divide
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
    /// let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);
    /// let mut c = a.remainder(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 3.0, 2.0]
    /// ```
    #[default_device]
    pub fn remainder_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_remainder(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise square root
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
    /// let mut b = a.sqrt().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn sqrt_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sqrt(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise cosine
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.cos().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.54030234, -0.41614687]
    /// ```
    #[default_device]
    pub fn cos_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_cos(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise exponential.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.exp().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.7182817, 7.389056]
    /// ```
    #[default_device]
    pub fn exp_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_exp(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise floor returning an error if the array is of type complex64.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
    /// let mut b = a.floor().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn floor_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_floor(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise integer division returning an error if arrays are not broadcastable.
    ///
    /// Divide two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// If either array is a floating point type then it is equivalent to calling [`Array::floor()`]
    /// after `/`.
    ///
    /// # Params
    ///
    /// - other: array to divide
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.floor_divide(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    #[default_device]
    pub fn floor_divide_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_floor_divide(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Return a boolean array indicating which elements are NaN.
    ///
    /// # Params
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn is_nan_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isnan(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Return a boolean array indicating which elements are infinity.
    ///
    /// # Params
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn is_inf_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isinf(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Return a boolean array indicating which elements are finite.
    ///
    /// # Params
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn is_finite_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isfinite(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Return a boolean array indicating which elements are negative infinity.
    ///
    /// # Params
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn is_neg_inf_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isneginf(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Return a boolean array indicating which elements are positive infinity.
    ///
    /// # Params
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn is_pos_inf_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isposinf(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise natural logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 0.6931472, 1.0986123]
    /// ```
    #[default_device]
    pub fn log_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_log(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise base-2 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
    /// let mut b = a.log2().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn log2_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_log2(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise base-10 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
    /// let mut b = a.log10().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn log10_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_log10(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise natural log of one plus the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log1p().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.6931472, 1.0986123, 1.3862944]
    /// ```
    #[default_device]
    pub fn log1p_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_log1p(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Matrix multiplication returning an error if inputs are not valid.
    ///
    /// Perform the (possibly batched) matrix multiplication of two arrays. This function supports
    /// broadcasting for arrays with more than two dimensions.
    ///
    /// - If the first array is 1-D then a 1 is prepended to its shape to make it
    ///   a matrix. Similarly, if the second array is 1-D then a 1 is appended to its
    ///   shape to make it a matrix. In either case the singleton dimension is removed
    ///   from the result.
    /// - A batched matrix multiplication is performed if the arrays have more than
    ///   2 dimensions.  The matrix dimensions for the matrix product are the last
    ///   two dimensions of each input.
    /// - All but the last two dimensions of each input are broadcast with one another using
    ///   standard [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to multiply
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
    /// let b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);
    ///
    /// // produces a [2, 3] result
    /// let mut c = a.matmul(&b);
    /// ```
    #[default_device]
    pub fn matmul_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_matmul(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise reciprocal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
    /// let mut b = a.reciprocal().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.5, 0.25]
    /// ```
    #[default_device]
    pub fn reciprocal_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_reciprocal(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Round to the given number of decimals.
    ///
    /// # Params
    ///
    /// - decimals: number of decimals to round to - default is 0 if not provided
    #[default_device]
    pub fn round_device(
        &self,
        decimals: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_round(
                res,
                self.as_ptr(),
                decimals.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Element-wise reciprocal and square root.
    #[default_device]
    pub fn rsqrt_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_rsqrt(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise sine.
    #[default_device]
    pub fn sin_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sin(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise square.
    #[default_device]
    pub fn square_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_square(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise real part from a complex array.
    #[default_device]
    pub fn real_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_real(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Element-wise imag part from a complex array.
    #[default_device]
    pub fn imag_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_imag(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }
}

/// Element-wise absolute value.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops};
///
/// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
/// let result = ops::abs(&array).unwrap();
/// ```
#[generate_macro]
#[default_device]
pub fn abs_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().abs_device(stream)
}

/// Element-wise inverse cosine.
#[generate_macro]
#[default_device]
pub fn acos_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arccos(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise inverse hyperbolic cosine.
#[generate_macro]
#[default_device]
pub fn acosh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arccosh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::add`].
#[generate_macro]
#[default_device]
pub fn add_device(
    lhs: impl AsRef<Array>,
    rhs: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    lhs.as_ref().add_device(rhs, stream)
}

/// Element-wise inverse sine.
#[generate_macro]
#[default_device]
pub fn asin_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arcsin(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise inverse hyperbolic sine.
#[generate_macro]
#[default_device]
pub fn asinh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arcsinh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise inverse tangent.
#[generate_macro]
#[default_device]
pub fn atan_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arctan(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise inverse hyperbolic tangent.
#[generate_macro]
#[default_device]
pub fn atanh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_arctanh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise ceiling.
#[generate_macro]
#[default_device]
pub fn ceil_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_ceil(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// A custom trait for the bound of the clip operation.
///
/// This trait is only implemented for tuples of the form `(Min, Max)`, `(Min, ())`, and `((),
/// Max)`. The `Min` and `Max` types must implement the `ScalarOrArray` trait.
pub trait ClipBound<'min, 'max>: Sealed {
    /// Convert the bound into a tuple of optional minimum and maximum values.
    fn into_min_max(
        self,
    ) -> (
        Option<impl ScalarOrArray<'min>>,
        Option<impl ScalarOrArray<'max>>,
    );
}

impl<'min, Min> ClipBound<'min, 'min> for (Min, ())
where
    Min: ScalarOrArray<'min> + Sealed,
{
    fn into_min_max(
        self,
    ) -> (
        Option<impl ScalarOrArray<'min>>,
        Option<impl ScalarOrArray<'min>>,
    ) {
        (Some(self.0), Option::<Min>::None)
    }
}

impl<'max, Max> ClipBound<'max, 'max> for ((), Max)
where
    Max: ScalarOrArray<'max> + Sealed,
{
    fn into_min_max(
        self,
    ) -> (
        Option<impl ScalarOrArray<'max>>,
        Option<impl ScalarOrArray<'max>>,
    ) {
        (Option::<Max>::None, Some(self.1))
    }
}

impl<'min, 'max, Min, Max> ClipBound<'min, 'max> for (Min, Max)
where
    Min: ScalarOrArray<'min> + Sealed,
    Max: ScalarOrArray<'max> + Sealed,
{
    fn into_min_max(
        self,
    ) -> (
        Option<impl ScalarOrArray<'min>>,
        Option<impl ScalarOrArray<'max>>,
    ) {
        (Some(self.0), Some(self.1))
    }
}

/// Clip the values of the array between the given minimum and maximum.
///
/// If either `a_min` or `a_max` are None, then corresponding edge is ignored. At least one of
/// `a_min` and `a_max` cannot be `None`. The input `a` and the limits must broadcast with one
/// another.
///
/// # Params
///
/// - `a`: Input array.
/// - `bound`: minimum and/or maximum values to clip the array to.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, ops::clip, array};
///
/// let a = array!([1.0, 4.0, 3.0, 8.0, 5.0]);
/// let expected = array!([2.0, 4.0, 3.0, 6.0, 5.0]);
/// let clipped = clip(&a, (2.0, 6.0)).unwrap();
/// assert_eq!(clipped, expected);
/// ```
#[generate_macro]
#[default_device]
pub fn clip_device<'min, 'max>(
    a: impl AsRef<Array>,
    bound: impl ClipBound<'min, 'max>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (a_min, a_max) = bound.into_min_max();

    // This is needed to keep the lifetime of the min/max arrays in scope.
    let a_min = a_min.map(|min| min.into_owned_or_ref_array());
    let a_max = a_max.map(|max| max.into_owned_or_ref_array());

    unsafe {
        let min_ptr = match &a_min {
            Some(a_min) => a_min.as_ref().as_ptr(),
            None => mlx_sys::mlx_array_new(),
        };
        let max_ptr = match &a_max {
            Some(a_max) => a_max.as_ref().as_ptr(),
            None => mlx_sys::mlx_array_new(),
        };

        Array::try_from_op(|res| {
            mlx_sys::mlx_clip(
                res,
                a.as_ref().as_ptr(),
                min_ptr,
                max_ptr,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// Element-wise cosine.
#[generate_macro]
#[default_device]
pub fn cos_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().cos_device(stream)
}

/// Element-wise hyperbolic cosine.
#[generate_macro]
#[default_device]
pub fn cosh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_cosh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Convert angles from radians to degrees.
#[generate_macro]
#[default_device]
pub fn degrees_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_degrees(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::divide`].
#[generate_macro]
#[default_device]
pub fn divide_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().divide_device(b, stream)
}

/// Element-wise quotient and remainder.
///
/// The fuction `divmod(a, b)` is equivalent to but faster than `(a // b, a % b)`. The function uses
/// numpy-style broadcasting semantics. Either or both input arrays can also be scalars.
///
/// Returns Ok((quotient, remainder)) if the operation was successful.
#[generate_macro]
#[default_device]
pub fn divmod_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();

    let vec = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_divmod(res, a_ptr, b_ptr, stream.as_ref().as_ptr())
    })?;

    let vals: SmallVec<[_; 2]> = vec.try_into_values()?;
    let mut iter = vals.into_iter();
    let quotient = iter.next().unwrap();
    let remainder = iter.next().unwrap();

    Ok((quotient, remainder))
}

/// Element-wise error function.
#[generate_macro]
#[default_device]
pub fn erf_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_erf(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise inverse error function.
#[generate_macro]
#[default_device]
pub fn erfinv_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_erfinv(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::exp`].
#[generate_macro]
#[default_device]
pub fn exp_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().exp_device(stream)
}

/// Element-wise exponential minus 1.
#[generate_macro]
#[default_device]
pub fn expm1_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_expm1(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::floor`].
#[generate_macro]
#[default_device]
pub fn floor_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().floor_device(stream)
}

/// See [`Array::floor_divide`].
#[generate_macro]
#[default_device]
pub fn floor_divide_device(
    a: impl AsRef<Array>,
    other: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().floor_divide_device(other, stream)
}

/// See [`Array::log`].
#[generate_macro]
#[default_device]
pub fn log_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().log_device(stream)
}

/// See [`Array::log10`].
#[generate_macro]
#[default_device]
pub fn log10_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().log10_device(stream)
}

/// See [`Array::log1p`].
#[generate_macro]
#[default_device]
pub fn log1p_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().log1p_device(stream)
}

/// See [`Array::log2`].
#[generate_macro]
#[default_device]
pub fn log2_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().log2_device(stream)
}

/// Element-wise log-add-exp.
///
/// This is a numerically stable log-add-exp of two arrays with numpy-style broadcasting semantics.
/// Either or both input arrays can also be scalars.
///
/// The computation is is a numerically stable version of `log(exp(a) + exp(b))`.
#[generate_macro]
#[default_device]
pub fn logaddexp_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_logaddexp(res, a_ptr, b_ptr, stream.as_ref().as_ptr())
    })
}

/// See [`Array::matmul`].
#[generate_macro]
#[default_device]
pub fn matmul_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().matmul_device(b, stream)
}

/// Element-wise maximum.
///
/// Take the element-wise max of two arrays with numpy-style broadcasting semantics. Either or both
/// input arrays can also be scalars.
#[generate_macro]
#[default_device]
pub fn maximum_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_maximum(res, a_ptr, b_ptr, stream.as_ref().as_ptr())
    })
}

/// Element-wise minimum.
///
/// Take the element-wise min of two arrays with numpy-style broadcasting semantics. Either or both
/// input arrays can also be scalars.
#[generate_macro]
#[default_device]
pub fn minimum_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_minimum(res, a_ptr, b_ptr, stream.as_ref().as_ptr())
    })
}

/// See [`Array::multiply`].
#[generate_macro]
#[default_device]
pub fn multiply_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().multiply_device(b, stream)
}

/// See [`Array::negative`].
#[generate_macro]
#[default_device]
pub fn negative_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().negative_device(stream)
}

/// See [`Array::power`].
#[generate_macro]
#[default_device]
pub fn power_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().power_device(b, stream)
}

/// Convert angles from degrees to radians.
#[generate_macro]
#[default_device]
pub fn radians_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_radians(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::reciprocal`].
#[generate_macro]
#[default_device]
pub fn reciprocal_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().reciprocal_device(stream)
}

/// See [`Array::remainder`].
#[generate_macro]
#[default_device]
pub fn remainder_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().remainder_device(b, stream)
}

/// See [`Array::round`].
#[generate_macro]
#[default_device]
pub fn round_device(
    a: impl AsRef<Array>,
    decimals: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().round_device(decimals, stream)
}

/// See [`Array::rsqrt`].
#[generate_macro]
#[default_device]
pub fn rsqrt_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().rsqrt_device(stream)
}

/// Element-wise logistic sigmoid.
///
/// See the [python API
/// docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html#mlx.core.sigmoid)
/// for more information
#[generate_macro]
#[default_device]
pub fn sigmoid_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sigmoid(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise sign.
#[generate_macro]
#[default_device]
pub fn sign_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sign(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// See [`Array::sin`].
#[generate_macro]
#[default_device]
pub fn sin_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().sin_device(stream)
}

/// Element-wise hyperbolic sine.
#[generate_macro]
#[default_device]
pub fn sinh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_sinh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Perform the softmax along the given axis.
///
/// See the [python API
/// docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html#mlx.core.softmax)
/// for more information.
#[generate_macro]
#[default_device]
pub fn softmax_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    precise: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let precise = precise.into().unwrap_or(false);
    let s = stream.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_softmax_axes(
            res,
            a.as_ref().as_ptr(),
            axes.as_ptr(),
            axes.len(),
            precise,
            s,
        )
    })
}

/// Similar to [`softmax_axes`] but with a single axis.
#[generate_macro]
#[default_device]
pub fn softmax_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    precise: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let precise = precise.into().unwrap_or(false);
    let s = stream.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_softmax_axis(res, a.as_ref().as_ptr(), axis, precise, s)
    })
}

/// Similar to [`softmax_axes`] but with no axis specified.
#[generate_macro]
#[default_device]
pub fn softmax_device(
    a: impl AsRef<Array>,
    precise: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let precise = precise.into().unwrap_or(false);
    let s = stream.as_ref().as_ptr();

    Array::try_from_op(|res| unsafe { mlx_sys::mlx_softmax(res, a.as_ref().as_ptr(), precise, s) })
}

/// See [`Array::sqrt`].
#[generate_macro]
#[default_device]
pub fn sqrt_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    a.as_ref().sqrt_device(stream)
}

/// See [`Array::square`].
#[generate_macro]
#[default_device]
pub fn square_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().square_device(stream)
}

/// See [`Array::subtract`].
#[generate_macro]
#[default_device]
pub fn subtract_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().subtract_device(b, stream)
}

/// See [`Array::tan`].
#[generate_macro]
#[default_device]
pub fn tan_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tan(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise hyperbolic tangent.
#[generate_macro]
#[default_device]
pub fn tanh_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tanh(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise real part from a complex array.
#[generate_macro]
#[default_device]
pub fn real_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_real(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Element-wise imaginary part from a complex array.
#[generate_macro]
#[default_device]
pub fn imag_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_imag(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Matrix multiplication with block masking.
///
/// See the [python API docs](
/// https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html#mlx.core.block_masked_mm
/// ) for more information.
#[generate_macro]
#[default_device]
pub fn block_masked_mm_device<'mo, 'lhs, 'rhs>(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] block_size: impl Into<Option<i32>>,
    #[optional] mask_out: impl Into<Option<&'mo Array>>,
    #[optional] mask_lhs: impl Into<Option<&'lhs Array>>,
    #[optional] mask_rhs: impl Into<Option<&'rhs Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();
    unsafe {
        let mask_out_ptr = mask_out
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());
        let mask_lhs_ptr = mask_lhs
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());
        let mask_rhs_ptr = mask_rhs
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());

        Array::try_from_op(|res| {
            mlx_sys::mlx_block_masked_mm(
                res,
                a_ptr,
                b_ptr,
                block_size.into().unwrap_or(32),
                mask_out_ptr,
                mask_lhs_ptr,
                mask_rhs_ptr,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// Matrix multiplication with addition and optional scaling.
///
/// Perform the (possibly batched) matrix multiplication of two arrays and add to the result with
/// optional scaling factors.
///
/// # Params
///
/// - `c`: input array,
/// - `a`: input array,
/// - `b`: input array,
/// - `alpha`: Scaling factor for the matrix product of `a` and `b` (default: `1`)
/// - `beta`: Scaling factor for `c` (default: `1`)
#[generate_macro]
#[default_device]
pub fn addmm_device(
    c: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] alpha: impl Into<Option<f32>>,
    #[optional] beta: impl Into<Option<f32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_ptr = c.as_ref().as_ptr();
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();
    let alpha = alpha.into().unwrap_or(1.0);
    let beta = beta.into().unwrap_or(1.0);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_addmm(
            res,
            c_ptr,
            a_ptr,
            b_ptr,
            alpha,
            beta,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the
/// last axes.
#[generate_macro]
#[default_device]
pub fn inner_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_inner(res, a.as_ptr(), b.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the outer product of two 1-D arrays, if the array’s passed are not 1-D a flatten op will
/// be run beforehand.
#[generate_macro]
#[default_device]
pub fn outer_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_outer(res, a.as_ptr(), b.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the tensor dot product along the specified axes.
#[generate_macro]
#[default_device]
pub fn tensordot_axes_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    axes_a: &[i32],
    axes_b: &[i32],
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tensordot(
            res,
            a.as_ptr(),
            b.as_ptr(),
            axes_a.as_ptr(),
            axes_a.len(),
            axes_b.as_ptr(),
            axes_b.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`tensordot_axes`] but with a single axis.
#[generate_macro]
#[default_device]
pub fn tensordot_axis_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_tensordot_axis(res, a.as_ptr(), b.as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;
    use crate::{
        array, complex64,
        ops::{all_close, arange, broadcast_to, eye, full, linspace, ones, reshape, split},
        transforms::eval,
        Dtype, StreamOrDevice,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_abs() {
        let data = [1i32, 2, -3, -4, -5];
        let array = Array::from_slice(&data, &[5]);
        let result = array.abs().unwrap();

        let data: &[i32] = result.as_slice();
        assert_eq!(data, [1, 2, 3, 4, 5]);

        // test that previous array is not modified and valid
        let data: &[i32] = array.as_slice();
        assert_eq!(data, [1, 2, -3, -4, -5]);
    }

    #[test]
    fn test_add() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let c = &a + &b;

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[5.0, 7.0, 9.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_add_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0], &[2]);

        let c = a.add(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_sub() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let c = &a - &b;

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[-3.0, -3.0, -3.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0], &[2]);
        let c = a.subtract(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_neg() {
        let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]);
        let b = a.negative().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[-1.0, -2.0, -3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_neg_bool() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = a.negative();
        assert!(b.is_err());
    }

    #[test]
    fn test_logical_not() {
        let a: Array = false.into();
        let b = a.logical_not().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true]);
    }

    #[test]
    fn test_mul() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let c = &a * &b;

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[4.0, 10.0, 18.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0], &[2]);
        let c = a.multiply(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_nan_to_num() {
        let a = array!([1.0, 2.0, f32::NAN, 4.0, 5.0]);
        let b = a.nan_to_num(0.0, 1.0, 0.0).unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.0, 0.0, 4.0, 5.0]);
    }

    #[test]
    fn test_div() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let c = &a / &b;

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[0.25, 0.4, 0.5]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_div_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0], &[2]);
        let c = a.divide(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_pow() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);

        let c = a.power(&b).unwrap();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[1.0, 8.0, 81.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pow_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 3.0], &[2]);
        let c = a.power(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_rem() {
        let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
        let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);

        let c = &a % &b;

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[1.0, 3.0, 2.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[10.0, 11.0, 12.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_rem_invalid_broadcast() {
        let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
        let b = Array::from_slice(&[3.0, 4.0], &[2]);
        let c = a.remainder(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_sqrt() {
        let a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
        let b = a.sqrt().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_cos() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let b = a.cos().unwrap();

        let b_expected = array!([1.0, 0.54030234, -0.41614687]);
        assert_array_all_close!(b, b_expected);

        // check a is not modified
        let a_expected = array!([0.0, 1.0, 2.0]);
        assert_array_all_close!(a, a_expected);
    }

    #[test]
    fn test_exp() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let b = a.exp().unwrap();

        let b_expected = array!([1.0, 2.7182817, 7.389056]);
        assert_array_all_close!(b, b_expected);

        // check a is not modified
        let a_expected = array!([0.0, 1.0, 2.0]);
        assert_array_all_close!(a, a_expected);
    }

    #[test]
    fn test_floor() {
        let a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
        let b = a.floor().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.1, 1.9, 2.5]);
    }

    #[test]
    fn test_floor_complex64() {
        let val = complex64::new(1.0, 2.0);
        let a = Array::from_complex(val);
        let b = a.floor_device(StreamOrDevice::default());
        assert!(b.is_err());
    }

    #[test]
    fn test_floor_divide() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let c = a.floor_divide(&b).unwrap();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[0.0, 0.0, 0.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_floor_divide_complex64() {
        let val = complex64::new(1.0, 2.0);
        let a = Array::from_complex(val);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.floor_divide_device(&b, StreamOrDevice::default());
        assert!(c.is_err());
    }

    #[test]
    fn test_floor_divide_invalid_broadcast() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0], &[2]);
        let c = a.floor_divide_device(&b, StreamOrDevice::default());
        assert!(c.is_err());
    }

    #[test]
    fn test_is_nan() {
        let a = Array::from_slice(&[1.0, f32::NAN, 3.0], &[3]);
        let b = a.is_nan().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, &[false, true, false]);
    }

    #[test]
    fn test_is_inf() {
        let a = Array::from_slice(&[1.0, f32::INFINITY, 3.0], &[3]);
        let b = a.is_inf().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, &[false, true, false]);
    }

    #[test]
    fn test_is_finite() {
        let a = Array::from_slice(&[1.0, f32::INFINITY, 3.0], &[3]);
        let b = a.is_finite().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, &[true, false, true]);
    }

    #[test]
    fn test_is_neg_inf() {
        let a = Array::from_slice(&[1.0, f32::NEG_INFINITY, 3.0], &[3]);
        let b = a.is_neg_inf().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, &[false, true, false]);
    }

    #[test]
    fn test_is_pos_inf() {
        let a = Array::from_slice(&[1.0, f32::INFINITY, 3.0], &[3]);
        let b = a.is_pos_inf().unwrap();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, &[false, true, false]);
    }

    #[test]
    fn test_log() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = a.log().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.6931472, 1.0986123]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_log2() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
        let b = a.log2().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_log10() {
        let a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
        let b = a.log10().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 10.0, 100.0]);
    }

    #[test]
    fn test_log1p() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = a.log1p().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.6931472, 1.0986123, 1.3862944]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[9.0, 39.5, 4.0, 13.0, 116.5, 12.0]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, &[1, 2, 3, 4]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[-5.0, 37.5, 4., 7., 1., 0.]);
    }

    #[test]
    fn test_matmul_ndim_zero() {
        let a: Array = 1.0.into();
        let b = Array::from_slice::<i32>(&[1], &[1]);
        let c = a.matmul(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_matmul_ndim_one() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let c = a.matmul(&b);
        assert!(c.is_ok());
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let b = Array::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[2, 5]);
        let c = a.matmul(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_matmul_non_float_output_type() {
        let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let b = Array::from_slice(&[5, 37, 4, 7, 1, 0], &[2, 3]);

        let c = a.matmul(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_reciprocal() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let b = a.reciprocal().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.5, 0.25]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_round() {
        let a = Array::from_slice(&[1.1, 2.9, 3.5], &[3]);
        let b = a.round(None).unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 3.0, 4.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.1, 2.9, 3.5]);
    }

    #[test]
    fn test_rsqrt() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let b = a.rsqrt().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.70710677, 0.5]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_sin() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let b = a.sin().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.841471, 0.9092974]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_square() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = a.square().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 4.0, 9.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    // The unit tests below are adapted from the original mlx c++ codebase.

    #[test]
    fn test_unary_neg() {
        let x = array!(1.0);
        assert_eq!(negative(&x).unwrap().item::<f32>(), -1.0);
        assert_eq!((-x).item::<f32>(), -1.0);

        // works on empty array
        assert_eq!(-array!(), array!());

        // Throws on bool
        let x = array!(true);
        assert!(negative(&x).is_err());
    }

    #[test]
    fn test_unary_abs() {
        let x = array!([-1.0, 0.0, 1.0]);
        assert_eq!(abs(&x).unwrap(), array!([1.0, 0.0, 1.0]));

        // works on empty array
        assert_eq!(abs(array!()).unwrap(), array!());

        // int32
        let x = array!([-1, 0, 1]);
        assert_eq!(abs(&x).unwrap(), array!([1, 0, 1]));

        // uint32
        let x = array!([1u32, 0, 1]);
        assert_eq!(abs(&x).unwrap(), array!([1u32, 0, 1]));

        // bool
        let x = array!([false, true]);
        assert_eq!(abs(&x).unwrap(), array!([false, true]));
    }

    #[test]
    fn test_unary_sign() {
        let x = array!([-1.0, 0.0, 1.0]);
        assert_eq!(sign(&x).unwrap(), x);

        // works on empty array
        assert_eq!(sign(array!()).unwrap(), array!());

        // int32
        let x = array!([-1, 0, 1]);
        assert_eq!(sign(&x).unwrap(), x);

        // uint32
        let x = array!([1u32, 0, 1]);
        assert_eq!(sign(&x).unwrap(), x);

        // bool
        let x = array!([false, true]);
        assert_eq!(sign(&x).unwrap(), x);
    }

    const NEG_INF: f32 = f32::NEG_INFINITY;

    #[test]
    fn test_unary_floor_ceil() {
        let x = array![1.0];
        assert_eq!(floor(&x).unwrap().item::<f32>(), 1.0);
        assert_eq!(ceil(&x).unwrap().item::<f32>(), 1.0);

        let x = array![1.5];
        assert_eq!(floor(&x).unwrap().item::<f32>(), 1.0);
        assert_eq!(ceil(&x).unwrap().item::<f32>(), 2.0);

        let x = array![-1.5];
        assert_eq!(floor(&x).unwrap().item::<f32>(), -2.0);
        assert_eq!(ceil(&x).unwrap().item::<f32>(), -1.0);

        let x = array![NEG_INF];
        assert_eq!(floor(&x).unwrap().item::<f32>(), NEG_INF);
        assert_eq!(ceil(&x).unwrap().item::<f32>(), NEG_INF);

        let x = array!([1.0, 1.0]).as_type::<complex64>().unwrap();
        assert!(floor(&x).is_err());
        assert!(ceil(&x).is_err());
    }

    #[test]
    fn test_unary_round() {
        let x = array!([0.5, -0.5, 1.5, -1.5, 2.3, 2.6]);
        assert_eq!(round(&x, None).unwrap(), array!([0, 0, 2, -2, 2, 3]));

        let x = array!([11, 222, 32]);
        assert_eq!(round(&x, -1).unwrap(), array!([10, 220, 30]));
    }

    #[test]
    fn test_unary_exp() {
        let x = array![0.0];
        assert_eq!(exp(&x).unwrap().item::<f32>(), 1.0);

        let x = array![2.0];
        assert_float_eq! {
            exp(&x).unwrap().item::<f32>(),
            2.0f32.exp(),
            abs <= 1e-5
        };

        assert_eq!(exp(array!()).unwrap(), array!());

        let x = array![NEG_INF];
        assert_eq!(exp(&x).unwrap().item::<f32>(), 0.0);

        // Integer input type
        let x = array![2];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            exp(&x).unwrap().item::<f32>(),
            2.0f32.exp(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&array!(1.0), &[2, 2, 2]).unwrap();
        let res = exp(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(1.0f32.exp())).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.exp(), 2.0f32.exp()], &[2, 1]);
        assert!(all_close(exp(&x[0]).unwrap(), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_expm1() {
        let x = array![-1.0];
        assert_float_eq! {
            expm1(&x).unwrap().item::<f32>(),
            (-1.0f32).exp_m1(),
            abs <= 1e-5
        };

        let x = array![1.0];
        assert_float_eq! {
            expm1(&x).unwrap().item::<f32>(),
            1.0f32.exp_m1(),
            abs <= 1e-5
        };

        // Integer input type
        let x = array![1];
        assert_eq!(expm1(&x).unwrap().dtype(), Dtype::Float32);
        assert_float_eq! {
            expm1(&x).unwrap().item::<f32>(),
            1.0f32.exp_m1(),
            abs <= 1e-5
        };
    }

    #[test]
    fn test_unary_sin() {
        let x = array![0.0];
        assert_eq!(sin(&x).unwrap().item::<f32>(), 0.0);

        let x = array![std::f32::consts::PI / 2.0];
        assert_float_eq! {
            sin(&x).unwrap().item::<f32>(),
            (std::f32::consts::PI / 2.0f32).sin(),
            abs <= 1e-5
        };

        assert_eq!(sin(array!()).unwrap(), array!());

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            sin(&x).unwrap().item::<f32>(),
            0.0f32.sin(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&array!(1.0), &[2, 2, 2]).unwrap();
        let res = sin(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(1.0f32.sin())).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.sin(), 2.0f32.sin()], &[2, 1]);
        assert!(all_close(sin(&x[0]).unwrap(), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_cos() {
        let x = array![0.0];
        assert_float_eq! {
            cos(&x).unwrap().item::<f32>(),
            0.0f32.cos(),
            abs <= 1e-5
        };

        let x = array![std::f32::consts::PI / 2.0];
        assert_float_eq! {
            cos(&x).unwrap().item::<f32>(),
            (std::f32::consts::PI / 2.0f32).cos(),
            abs <= 1e-5
        };

        assert_eq!(cos(array!()).unwrap(), array!());

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            cos(&x).unwrap().item::<f32>(),
            0.0f32.cos(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&array!(1.0), &[2, 2, 2]).unwrap();
        let res = cos(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(1.0f32.cos())).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.cos(), 2.0f32.cos()], &[2, 1]);
        assert!(all_close(cos(&x[0]).unwrap(), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_degrees() {
        let x = array![0.0];
        assert_eq!(degrees(&x).unwrap().item::<f32>(), 0.0);

        let x = array![std::f32::consts::PI / 2.0];
        assert_eq!(degrees(&x).unwrap().item::<f32>(), 90.0);

        assert_eq!(degrees(array!()).unwrap(), array!());

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_eq!(degrees(&x).unwrap().item::<f32>(), 0.0);

        // Input is irregularly strided
        let x = broadcast_to(&array!(std::f32::consts::PI / 2.0), &[2, 2, 2]).unwrap();
        let res = degrees(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(90.0)).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let angles = Array::from_slice(&[0.0, PI / 2.0, PI, 1.5 * PI], &[2, 2]);
        let x = split(&angles, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0, 180.0], &[2, 1]);
        assert!(
            all_close(degrees(&x[0]).unwrap(), &expected, None, None, None)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn test_unary_radians() {
        let x = array![0.0];
        assert_eq!(radians(&x).unwrap().item::<f32>(), 0.0);

        let x = array![90.0];
        assert_eq!(
            radians(&x).unwrap().item::<f32>(),
            std::f32::consts::PI / 2.0
        );

        assert_eq!(radians(array!()).unwrap(), array!());

        // Integer input type
        let x = array![90];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_eq!(
            radians(&x).unwrap().item::<f32>(),
            std::f32::consts::PI / 2.0
        );

        // Input is irregularly strided
        let x = broadcast_to(&array!(90.0), &[2, 2, 2]).unwrap();
        let res = radians(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(std::f32::consts::PI / 2.0)).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let angles = Array::from_slice(&[0.0, 90.0, 180.0, 270.0], &[2, 2]);
        let x = split(&angles, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0, PI], &[2, 1]);
        assert!(
            all_close(radians(&x[0]).unwrap(), &expected, None, None, None)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn test_unary_log() {
        let x = array![0.0];
        assert_eq!(log(&x).unwrap().item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log(&x).unwrap().item::<f32>(), 0.0);

        // Integer input type
        let x = array![1];
        assert_eq!(log(&x).unwrap().dtype(), Dtype::Float32);
        assert_eq!(log(&x).unwrap().item::<f32>(), 0.0);

        // Input is irregularly strided
        let x = broadcast_to(&array!(1.0), &[2, 2, 2]).unwrap();
        let res = log(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(0.0)).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = split(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[1.0f32.ln(), 3.0f32.ln()], &[2, 1]);
        assert!(all_close(log(&x[0]).unwrap(), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_log2() {
        let x = array![0.0];
        assert_eq!(log2(&x).unwrap().item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log2(&x).unwrap().item::<f32>(), 0.0);

        let x = array![1024.0];
        assert_eq!(log2(&x).unwrap().item::<f32>(), 10.0);
    }

    #[test]
    fn test_unary_log10() {
        let x = array![0.0];
        assert_eq!(log10(&x).unwrap().item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log10(&x).unwrap().item::<f32>(), 0.0);

        let x = array![1000.0];
        assert_eq!(log10(&x).unwrap().item::<f32>(), 3.0);
    }

    #[test]
    fn test_unary_log1p() {
        let x = array![-1.0];
        assert_float_eq! {
            log1p(&x).unwrap().item::<f32>(),
            (-1.0f32).ln_1p(),
            abs <= 1e-5
        };

        let x = array![1.0];
        assert_float_eq! {
            log1p(&x).unwrap().item::<f32>(),
            1.0f32.ln_1p(),
            abs <= 1e-5
        };

        // Integer input type
        let x = array![1];
        assert_eq!(log1p(&x).unwrap().dtype(), Dtype::Float32);
        assert_float_eq! {
            log1p(&x).unwrap().item::<f32>(),
            1.0f32.ln_1p(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&array!(1.0), &[2, 2, 2]).unwrap();
        let res = log1p(&x).unwrap();
        let expected = Array::full::<f32>(&[2, 2, 2], array!(1.0f32.ln_1p())).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = split(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[1.0f32.ln_1p(), 3.0f32.ln_1p()], &[2, 1]);
        assert!(
            all_close(log1p(&x[0]).unwrap(), &expected, None, None, None)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn test_unary_sigmoid() {
        let x = array![0.0];
        assert_float_eq! {
            sigmoid(&x).unwrap().item::<f32>(),
            0.5,
            abs <= 1e-5
        };

        // Integer input type
        let x = array![0];
        assert_eq!(sigmoid(&x).unwrap().dtype(), Dtype::Float32);
        assert_float_eq! {
            sigmoid(&x).unwrap().item::<f32>(),
            0.5,
            abs <= 1e-5
        };

        let inf = f32::INFINITY;
        let x = array![inf];
        assert_eq!(sigmoid(&x).unwrap().item::<f32>(), 1.0);

        let x = array![-inf];
        assert_eq!(sigmoid(&x).unwrap().item::<f32>(), 0.0);
    }

    #[test]
    fn test_unary_square() {
        let x = array![3.0];
        assert_eq!(square(&x).unwrap().item::<f32>(), 9.0);

        let x = array![2];
        assert_eq!(square(&x).unwrap().item::<i32>(), 4);

        let x = Array::full::<f32>(&[3, 3], array!(2.0)).unwrap();
        assert!(all_close(
            square(&x).unwrap(),
            Array::full::<f32>(&[3, 3], array!(4.0)).unwrap(),
            None,
            None,
            None
        )
        .unwrap()
        .item::<bool>());
    }

    #[test]
    fn test_unary_sqrt_rsqrt() {
        let x = array![4.0];
        assert_eq!(sqrt(&x).unwrap().item::<f32>(), 2.0);
        assert_eq!(rsqrt(&x).unwrap().item::<f32>(), 0.5);

        let x = Array::full::<f32>(&[3, 3], array!(9.0)).unwrap();
        assert!(all_close(
            sqrt(&x).unwrap(),
            Array::full::<f32>(&[3, 3], array!(3.0)).unwrap(),
            None,
            None,
            None
        )
        .unwrap()
        .item::<bool>());

        let x = array![4i32];
        assert_eq!(sqrt(&x).unwrap().item::<f32>(), 2.0);
        assert_eq!(rsqrt(&x).unwrap().item::<f32>(), 0.5);
    }

    #[test]
    fn test_unary_reciprocal() {
        let x = array![8.0];
        assert_eq!(reciprocal(&x).unwrap().item::<f32>(), 0.125);

        let x = array![2];
        let out = reciprocal(&x).unwrap();
        assert_eq!(out.dtype(), Dtype::Float32);
        assert_eq!(out.item::<f32>(), 0.5);

        let x = Array::full::<f32>(&[3, 3], array!(2.0)).unwrap();
        assert!(all_close(
            reciprocal(&x).unwrap(),
            Array::full::<f32>(&[3, 3], array!(0.5)).unwrap(),
            None,
            None,
            None
        )
        .unwrap()
        .item::<bool>());
    }

    #[test]
    fn test_unary_real_imag() {
        let x = Array::from_complex(complex64::new(0.0, 1.0));
        assert_eq!(real(&x).unwrap(), Array::from_f32(0.0));
        assert_eq!(imag(&x).unwrap(), Array::from_f32(1.0));
    }

    #[test]
    fn test_binary_add() {
        let x = array![1.0];
        let y = array![1.0];
        let z = add(&x, &y).unwrap();
        assert_eq!(z.item::<f32>(), 2.0);

        let z = &x + y;
        assert_eq!(z.item::<f32>(), 2.0);

        let z = add(z, &x).unwrap();
        assert_eq!(z.item::<f32>(), 3.0);

        // Chain a few adds:
        let mut out = x.deep_clone();
        for _ in 0..10 {
            out = add(&out, &x).unwrap();
        }
        assert_eq!(out.item::<f32>(), 11.0);

        // Works for different shapes
        let x = array!([1.0, 2.0, 3.0]);
        let y = array!([1.0, 2.0, 3.0]);
        let z = add(&x, &y).unwrap();
        assert_eq!(z.shape(), &[3]);
        assert_eq!(z, array!([2.0, 4.0, 6.0]));

        // Works with scalars
        let x = array!([1.0, 2.0, 3.0]);
        let y = &x + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        assert_eq!(y, array!([3.0, 4.0, 5.0]));
        let y = &x + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        assert_eq!(y, array!([3.0, 4.0, 5.0]));

        // Check type promotion
        let y = x + 2;
        assert_eq!(y.dtype(), Dtype::Float32);

        let y = array!([1, 2, 3]) + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        // assert!(array_equal(&y, &array![3.0, 4.0, 5.0]).item::<bool>());
        assert_eq!(y, array!([3.0, 4.0, 5.0]));

        // Broadcasting works
        let x = broadcast_to(&array!(1.0), &[10]).unwrap();
        let y = broadcast_to(&array!(2.0), &[10]).unwrap();
        let z = add(&x, &y).unwrap();
        assert_eq!(z, full::<f32>(&[10], array!(3.0)).unwrap());

        let x = Array::from_slice(&[1.0, 2.0], &[1, 2]);
        let y = Array::from_slice(&[1.0, 2.0], &[2, 1]);
        let z = add(&x, &y).unwrap();
        assert_eq!(z.shape(), &[2, 2]);
        assert_eq!(z, Array::from_slice(&[2.0, 3.0, 3.0, 4.0], &[2, 2]));

        let x = ones::<f32>(&[3, 2, 1]).unwrap();
        let z = x + 2.0;
        assert_eq!(z.shape(), &[3, 2, 1]);
        let expected = Array::from_slice(&[3.0, 3.0, 3.0, 3.0, 3.0, 3.0], &[3, 2, 1]);
        assert_eq!(z, expected);

        // Works for empty arrays
        let x = array!();
        let y = array!();
        let z = x + y;
        z.eval().unwrap();
        assert_eq!(z.size(), 0);
        assert_eq!(z.shape(), &[0]);
    }

    #[test]
    fn test_binary_sub() {
        let x = array!([3.0, 2.0, 1.0]);
        let y = array!([1.0, 1.0, 1.0]);
        assert_eq!(x - y, array!([2.0, 1.0, 0.0]));
    }

    #[test]
    fn test_binary_mul() {
        let x = array!([1.0, 2.0, 3.0]);
        let y = array!([2.0, 2.0, 2.0]);
        assert_eq!(x * y, array!([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_binary_div() {
        let x = array![1.0];
        let y = array![1.0];
        assert_eq!(divide(&x, &y).unwrap().item::<f32>(), 1.0);

        let x = array![1.0];
        let y = array![0.5];
        assert_eq!(divide(&x, &y).unwrap().item::<f32>(), 2.0);

        let x = array![1.0];
        let y = array![4.0];
        assert_eq!(divide(&x, &y).unwrap().item::<f32>(), 0.25);

        let x = array![true];
        let y = array![true];
        assert_eq!(divide(&x, &y).unwrap().item::<f32>(), 1.0);

        let x = array![false];
        let y = array![true];
        assert_eq!(divide(&x, &y).unwrap().item::<f32>(), 0.0);

        let x = array![true];
        let y = array![false];
        assert!(divide(&x, &y).unwrap().item::<f32>().is_infinite());

        let x = array![false];
        let y = array![false];
        assert!(divide(&x, &y).unwrap().item::<f32>().is_nan());
    }

    #[test]
    fn test_binary_maximum_minimum() {
        let x = array![1.0];
        let y = array![0.0];
        assert_eq!(maximum(&x, &y).unwrap().item::<f32>(), 1.0);
        assert_eq!(minimum(&x, &y).unwrap().item::<f32>(), 0.0);

        let y = array![2.0];
        assert_eq!(maximum(&x, &y).unwrap().item::<f32>(), 2.0);
        assert_eq!(minimum(&x, &y).unwrap().item::<f32>(), 1.0);
    }

    #[test]
    fn test_binary_logaddexp() {
        let x = array![0.0];
        let y = array![0.0];
        assert_float_eq! {
            logaddexp(&x, &y).unwrap().item::<f32>(),
            2.0f32.ln(),
            abs <= 1e-5
        };

        let x = array!([0u32]);
        let y = array!([10000u32]);
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), 10000.0);

        let x = array![f32::INFINITY];
        let y = array![3.0];
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), f32::INFINITY);

        let x = array![f32::NEG_INFINITY];
        let y = array![3.0];
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), 3.0);

        let x = array![f32::NEG_INFINITY];
        let y = array![f32::NEG_INFINITY];
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), f32::NEG_INFINITY);

        let x = array![f32::INFINITY];
        let y = array![f32::INFINITY];
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), f32::INFINITY);

        let x = array![f32::NEG_INFINITY];
        let y = array![f32::INFINITY];
        assert_eq!(logaddexp(&x, &y).unwrap().item::<f32>(), f32::INFINITY);
    }

    #[test]
    fn test_basic_clip() {
        let a = array!([1.0, 4.0, 3.0, 8.0, 5.0]);
        let expected = array!([2.0, 4.0, 3.0, 6.0, 5.0]);
        let clipped = clip(&a, (array!(2.0), array!(6.0))).unwrap();
        assert_eq!(clipped, expected);

        // Test with scalar
        let clipped = clip(&a, (2.0, 6.0)).unwrap();
        assert_eq!(clipped, expected);
    }

    #[test]
    fn test_clip_with_only_min() {
        let a = array!([-1.0, 1.0, 0.0, 5.0]);
        let expected = array!([0.0, 1.0, 0.0, 5.0]);
        let clipped = clip(&a, (array!(0.0), ())).unwrap();
        assert_eq!(clipped, expected);

        // Test with scalar
        let clipped = clip(&a, (0.0, ())).unwrap();
        assert_eq!(clipped, expected);
    }

    #[test]
    fn test_clip_with_only_max() {
        let a = array!([2.0, 3.0, 4.0, 5.0]);
        let expected = array!([2.0, 3.0, 4.0, 4.0]);
        let clipped = clip(&a, ((), array!(4.0))).unwrap();
        assert_eq!(clipped, expected);

        // Test with scalar
        let clipped = clip(&a, ((), 4.0)).unwrap();
        assert_eq!(clipped, expected);
    }

    #[test]
    fn test_tensordot() {
        let x = reshape(arange::<_, f32>(None, 60.0, None).unwrap(), &[3, 4, 5]).unwrap();
        let y = reshape(arange::<_, f32>(None, 24.0, None).unwrap(), &[4, 3, 2]).unwrap();
        let z = tensordot_axes(&x, &y, &[1i32, 0], &[0i32, 1]).unwrap();
        let expected = Array::from_slice(
            &[4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306],
            &[5, 2],
        );
        assert_eq!(z, expected);

        let x = reshape(arange::<_, f32>(None, 360.0, None).unwrap(), &[3, 4, 5, 6]).unwrap();
        let y = reshape(arange::<_, f32>(None, 360.0, None).unwrap(), &[6, 4, 5, 3]).unwrap();
        assert!(tensordot_axes(&x, &y, &[2, 1, 3], &[1, 2, 0]).is_err());

        let x = reshape(arange::<_, f32>(None, 60.0, None).unwrap(), &[3, 4, 5]).unwrap();
        let y = reshape(arange::<_, f32>(None, 120.0, None).unwrap(), &[4, 5, 6]).unwrap();

        let z = tensordot_axis(&x, &y, 2).unwrap();
        let expected = Array::from_slice(
            &[
                14820.0, 15010.0, 15200.0, 15390.0, 15580.0, 15770.0, 37620.0, 38210.0, 38800.0,
                39390.0, 39980.0, 40570.0, 60420.0, 61410.0, 62400.0, 63390.0, 64380.0, 65370.0,
            ],
            &[3, 6],
        );
        assert_eq!(z, expected);
    }

    #[test]
    fn test_outer() {
        let x = arange::<_, f32>(1.0, 5.0, None).unwrap();
        let y = arange::<_, f32>(1.0, 4.0, None).unwrap();
        let z = outer(&x, &y).unwrap();
        let expected = Array::from_slice(
            &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
            &[4, 3],
        );
        assert_eq!(z, expected);

        let x = ones::<f32>(&[5]).unwrap();
        let y = linspace::<_, f32>(-2.0, 2.0, 5).unwrap();
        let z = outer(&x, &y).unwrap();
        let expected = Array::from_slice(
            &[
                -2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0,
                -2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0,
            ],
            &[5, 5],
        );
        assert_eq!(z, expected);
    }

    #[test]
    fn test_inner() {
        let x = reshape(arange::<_, f32>(None, 5.0, None).unwrap(), &[1, 5]).unwrap();
        let y = reshape(arange::<_, f32>(None, 6.0, None).unwrap(), &[2, 3]).unwrap();
        assert!(inner(&x, &y).is_err());

        let x = array!([1.0, 2.0, 3.0]);
        let y = array!([0.0, 1.0, 0.0]);
        let z = inner(&x, &y).unwrap();
        assert_eq!(z.item::<f32>(), 2.0);

        let x = reshape(arange::<_, f32>(None, 24.0, None).unwrap(), &[2, 3, 4]).unwrap();
        let y = arange::<_, f32>(None, 4.0, None).unwrap();
        let z = inner(&x, &y).unwrap();
        let expected = Array::from_slice(&[14.0, 38.0, 62.0, 86.0, 110.0, 134.0], &[2, 3]);
        assert_eq!(z, expected);

        let x = reshape(arange::<_, f32>(None, 2.0, None).unwrap(), &[1, 1, 2]).unwrap();
        let y = reshape(arange::<_, f32>(None, 6.0, None).unwrap(), &[3, 2]).unwrap();
        let z = inner(&x, &y).unwrap();
        let expected = Array::from_slice(&[1.0, 3.0, 5.0], &[1, 1, 3]);
        assert_eq!(z, expected);

        let x = eye::<f32>(2, None, None).unwrap();
        let y = Array::from_f32(7.0);
        let z = inner(&x, &y).unwrap();
        let expected = Array::from_slice(&[7.0, 0.0, 0.0, 7.0], &[2, 2]);
        assert_eq!(z, expected);
    }

    #[test]
    fn test_divmod() {
        let x = array!([1.0, 2.0, 3.0]);
        let y = array!([1.0, 1.0, 1.0]);
        let out = divmod(&x, &y).unwrap();
        assert_eq!(out.0, array!([1.0, 2.0, 3.0]));
        assert_eq!(out.1, array!([0.0, 0.0, 0.0]));

        let x = array!([5.0, 6.0, 7.0]);
        let y = array!([2.0, 2.0, 2.0]);
        let out = divmod(&x, &y).unwrap();
        assert_eq!(out.0, array!([2.0, 3.0, 3.0]));
        assert_eq!(out.1, array!([1.0, 0.0, 1.0]));

        let x = array!([5.0, 6.0, 7.0]);
        let y = array!([2.0, 2.0, 2.0]);
        let out = divmod(&x, &y).unwrap();
        assert_eq!(out.0, array!([2.0, 3.0, 3.0]));
        assert_eq!(out.1, array!([1.0, 0.0, 1.0]));

        let x = array![complex64::new(1.0, 0.0)];
        let y = array![complex64::new(2.0, 0.0)];
        assert!(divmod(&x, &y).is_err());

        // Check that we can eval on both outputs
        let x = array![1.0];
        let y = array![2.0];
        let (quo, rem) = divmod(&x, &y).unwrap();
        eval([&quo, &rem]).unwrap();
        assert_eq!(quo.item::<f32>(), 0.0);
        assert_eq!(rem.item::<f32>(), 1.0);

        // Check nested in the graph
        let x = array![1.0];
        let y = array![2.0];
        let (quo, rem) = divmod(&x, &y).unwrap();
        let z = quo + rem;
        assert_eq!(z.item::<f32>(), 1.0);

        // Check that we can still eval when one output goes out of scope
        let mut out_holder = {
            let (quo, _) = divmod(&x, &y).unwrap();
            vec![quo]
        };
        eval(out_holder.iter()).unwrap();
        assert_eq!(out_holder[0].item::<f32>(), 0.0);

        // Check that we can still eval when the other output goes out of scope
        out_holder.clear();
        let out_holder = {
            let (_, rem) = divmod(&x, &y).unwrap();
            vec![rem]
        };
        eval(out_holder.iter()).unwrap();
        assert_eq!(out_holder[0].item::<f32>(), 1.0);
    }
}
