use crate::array::Array;
use crate::error::Exception;
use crate::prelude::ScalarOrArray;
use crate::stream::StreamOrDevice;

use crate::utils::{IntoOption, VectorArray};
use crate::Stream;
use mlx_macros::default_device;
use smallvec::SmallVec;

// // Element wise free functions
// abs(_:stream:)
// acos(_:stream:)
// acosh(_:stream:)
// add(_:_:stream:)
// asin(_:stream:)
// asinh(_:stream:)
// atan(_:stream:)
// atanh(_:stream:)
// ceil(_:stream:)
// clip(_:min:max:stream:)
// cos(_:stream:)
// cosh(_:stream:)
// degrees(_:stream:)
// divide(_:_:stream:)
// divmod(_:_:stream:)
// erf(_:stream:)
// erfInverse(_:stream:)
// exp(_:stream:)
// expm1(_:stream:)
// floor(_:stream:)
// floorDivide(_:_:stream:)
// isNaN(_:stream:)
// isInf(_:stream:)
// isPosInf(_:stream:)
// isNegInf(_:stream:)
// log(_:stream:)
// log10(_:stream:)
// log1p(_:stream:)
// log2(_:stream:)
// logAddExp(_:_:stream:)
// logicalAnd(_:_:stream:)
// logicalNot(_:stream:)
// logicalOr(_:_:stream:)
// matmul(_:_:stream:)
// maximum(_:_:stream:)
// minimum(_:_:stream:)
// multiply(_:_:stream:)
// negative(_:stream:)
// notEqual(_:_:stream:)
// pow(_:_:stream:)-7pe7j
// pow(_:_:stream:)-49xi0
// pow(_:_:stream:)-8ie9c
// radians(_:stream:)
// reciprocal(_:stream:)
// remainder(_:_:stream:)
// round(_:decimals:stream:)
// rsqrt(_:stream:)
// sigmoid(_:stream:)
// sign(_:stream:)
// sin(_:stream:)
// sinh(_:stream:)
// softmax(_:axes:precise:stream:)
// sqrt(_:stream:)
// square(_:stream:)
// subtract(_:_:stream:)
// tan(_:stream:)
// tanh(_:stream:)

// // Vector, Matrix, and Tensor Products
// matmul(_:_:stream:)
// blockMaskedMM(_:_:blockSize:maskOut:maskLHS:maskRHS:stream:)
// addMM(_:_:_:alpha:beta:stream:)
// inner(_:_:stream:)
// outer(_:_:stream:)
// tensordot(_:_:axes:stream:)-3qkgq
// tensordot(_:_:axes:stream:)-8yqyi

impl Array {
    /// Element-wise absolute value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
    /// let mut result = array.abs();
    ///
    /// let data: &[i32] = result.as_slice();
    /// // data == [1, 2, 3, 4, 5]
    /// ```
    #[default_device]
    pub fn abs_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_abs(self.c_array, stream.as_ref().as_ptr())) }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.add_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [5.0, 7.0, 9.0]
    /// ```
    #[default_device]
    pub fn add_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_add(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.subtract_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [-3.0, -3.0, -3.0]
    /// ```
    #[default_device]
    pub fn subtract_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_subtract(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Unary element-wise negation. Returns an error if the array is of type bool.
    ///
    /// Negate the values in the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.neg().unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [-1.0, -2.0, -3.0]
    /// ```
    #[default_device]
    pub fn negative_device(&self, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_negative(self.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise multiplication returning an error if arrays are not broadcastable.
    ///
    /// Multiply two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.multiply(&b).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [4.0, 10.0, 18.0]
    /// ```
    #[default_device]
    pub fn multiply_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_multiply(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.divide_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    #[default_device]
    pub fn divide_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_divide(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);
    /// let mut c = a.power_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 8.0, 81.0]
    /// ```
    #[default_device]
    pub fn power_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_power(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
    /// let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);
    /// let mut c = a.remainder_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 3.0, 2.0]
    /// ```
    #[default_device]
    pub fn remainder_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_remainder(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise square root
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
    /// let mut b = a.sqrt_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn sqrt_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_sqrt(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise cosine
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.cos_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.54030234, -0.41614687]
    /// ```
    #[default_device]
    pub fn cos_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_cos(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise exponential.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    ///
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.exp_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.7182817, 7.389056]
    /// ```
    #[default_device]
    pub fn exp_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_exp(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise floor returning an error if the array is of type complex64.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
    /// let mut b = a.floor_device(StreamOrDevice::default()).unwrap();
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn floor_device(&self, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_floor(self.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.floor_divide_device(&b, StreamOrDevice::default()).unwrap();
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    #[default_device]
    pub fn floor_divide_device<'a>(
        &self,
        other: impl ScalarOrArray<'a>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_floor_divide(self.c_array, other.into_owned_or_ref_array().as_ref().as_ptr(), stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise natural logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 0.6931472, 1.0986123]
    /// ```
    #[default_device]
    pub fn log_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise base-2 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
    /// let mut b = a.log2_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn log2_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log2(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise base-10 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
    /// let mut b = a.log10_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn log10_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log10(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise natural log of one plus the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log1p_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.6931472, 1.0986123, 1.3862944]
    /// ```
    #[default_device]
    pub fn log1p_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log1p(self.c_array, stream.as_ref().as_ptr())) }
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
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
    /// let b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);
    ///
    /// // produces a [2, 3] result
    /// let mut c = a.matmul_device(&b, StreamOrDevice::default());
    /// ```
    #[default_device]
    pub fn matmul_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_matmul(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise reciprocal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
    /// let mut b = a.reciprocal_device(StreamOrDevice::default());
    ///
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.5, 0.25]
    /// ```
    #[default_device]
    pub fn reciprocal_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_reciprocal(
                self.c_array,
                stream.as_ref().as_ptr(),
            ))
        }
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
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_round(
                self.c_array,
                decimals.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            ))
        }
    }

    /// Element-wise reciprocal and square root.
    #[default_device]
    pub fn rsqrt_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_rsqrt(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise sine.
    #[default_device]
    pub fn sin_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_sin(self.c_array, stream.as_ref().as_ptr())) }
    }

    /// Element-wise square.
    #[default_device]
    pub fn square_device(&self, stream: impl AsRef<Stream>) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_square(self.c_array, stream.as_ref().as_ptr())) }
    }
}

/// Element-wise absolute value.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops};
///
/// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
/// let result = ops::abs(&array);
/// ```
#[default_device]
pub fn abs_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.abs_device(stream)
}

/// Element-wise inverse cosine.
#[default_device]
pub fn acos_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_arccos(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn acosh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_arccosh(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn add_device<'a, 'b>(
    lhs: impl ScalarOrArray<'a>,
    rhs: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    lhs.into_owned_or_ref_array()
        .as_ref()
        .add_device(rhs, stream)
}

#[default_device]
pub fn asin_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_arcsin(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn asinh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_arcsinh(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn atan_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_arctan(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn atanh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_arctanh(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn ceil_device(array: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_ceil(array.c_array, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn clip_device<'min, 'max>(
    array: &Array,
    min: impl ScalarOrArray<'min>,
    max: Option<impl ScalarOrArray<'max>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let min_ptr = min.into_owned_or_ref_array().as_ref().as_ptr();
    let max_ptr = max
        .map(|max| max.into_owned_or_ref_array().as_ref().as_ptr())
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_clip(array.as_ptr(), min_ptr, max_ptr, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn cos_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.cos_device(stream)
}

#[default_device]
pub fn cosh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_cosh(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn degrees_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_degrees(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn divide_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.into_owned_or_ref_array()
        .as_ref()
        .divide_device(b, stream)
}

#[default_device]
pub fn divmod_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array), Exception> {
    let a_ptr = a.into_owned_or_ref_array().as_ref().as_ptr();
    let b_ptr = b.into_owned_or_ref_array().as_ref().as_ptr();

    unsafe {
        let c_vec = try_catch_c_ptr_expr! {
            mlx_sys::mlx_divmod(a_ptr, b_ptr, stream.as_ref().as_ptr())
        };
        let vec = VectorArray::from_ptr(c_vec);
        let vals: SmallVec<[_; 2]> = vec.into_values();
        let mut iter = vals.into_iter();
        let quotient = iter.next().unwrap();
        let remainder = iter.next().unwrap();

        Ok((quotient, remainder))
    }
}

#[default_device]
pub fn erf_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_erf(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn erfinv_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_erfinv(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn exp_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.exp_device(stream)
}

#[default_device]
pub fn expm1_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_expm1(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn floor_device(array: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    array.floor_device(stream)
}

#[default_device]
pub fn floor_divide_device<'a>(
    array: &Array,
    other: impl ScalarOrArray<'a>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    array.floor_divide_device(other, stream)
}

#[default_device]
pub fn log_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.log_device(stream)
}

#[default_device]
pub fn log10_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.log10_device(stream)
}

#[default_device]
pub fn log1p_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.log1p_device(stream)
}

#[default_device]
pub fn log2_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.log2_device(stream)
}

#[default_device]
pub fn log_add_exp_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let a_ptr = a.into_owned_or_ref_array().as_ref().as_ptr();
    let b_ptr = b.into_owned_or_ref_array().as_ref().as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_logaddexp(a_ptr, b_ptr, stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn matmul_device(
    lhs: &Array,
    rhs: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    lhs.matmul_device(rhs, stream)
}

#[default_device]
pub fn maximum_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let a_ptr = a.into_owned_or_ref_array().as_ref().as_ptr();
    let b_ptr = b.into_owned_or_ref_array().as_ref().as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_maximum(a_ptr, b_ptr, stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn minimum_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let a_ptr = a.into_owned_or_ref_array().as_ref().as_ptr();
    let b_ptr = b.into_owned_or_ref_array().as_ref().as_ptr();

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_minimum(a_ptr, b_ptr, stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn multiply_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.into_owned_or_ref_array()
        .as_ref()
        .multiply_device(b, stream)
}

#[default_device]
pub fn negative_device(array: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    array.negative_device(stream)
}

#[default_device]
pub fn power_device<'a>(
    a: &Array,
    b: impl ScalarOrArray<'a>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.power_device(b, stream)
}

#[default_device]
pub fn radians_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_radians(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn reciprocal_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.reciprocal_device(stream)
}

#[default_device]
pub fn remainder_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.into_owned_or_ref_array()
        .as_ref()
        .remainder_device(b, stream)
}

#[default_device]
pub fn round_device(
    array: &Array,
    decimals: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    array.round_device(decimals, stream)
}

#[default_device]
pub fn rsqrt_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.rsqrt_device(stream)
}

#[default_device]
pub fn sigmoid_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_sigmoid(
            array.c_array,
            stream.as_ref().as_ptr(),
        ))
    }
}

#[default_device]
pub fn sign_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_sign(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn sin_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.sin_device(stream)
}

#[default_device]
pub fn sinh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_sinh(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn softmax_device<'a>(
    array: &Array,
    axes: impl IntoOption<&'a [i32]>,
    precise: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let precise = precise.into().unwrap_or(false);
    let s = stream.as_ref().as_ptr();

    unsafe {
        let c_array = match axes.into_option() {
            Some(axes) => {
                mlx_sys::mlx_softmax(array.as_ptr(), axes.as_ptr(), axes.len(), precise, s)
            }
            None => mlx_sys::mlx_softmax_all(array.as_ptr(), precise, s),
        };

        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn sqrt_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.sqrt_device(stream)
}

#[default_device]
pub fn square_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    array.square_device(stream)
}

#[default_device]
pub fn subtract_device<'a, 'b>(
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.into_owned_or_ref_array()
        .as_ref()
        .subtract_device(b, stream)
}

#[default_device]
pub fn tan_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_tan(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn tanh_device(array: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_tanh(array.c_array, stream.as_ref().as_ptr())) }
}

#[default_device]
pub fn block_masked_mm_device<'mo, 'lhs, 'rhs>(
    a: &Array,
    b: &Array,
    block_size: impl Into<Option<i32>>,
    mask_out: impl Into<Option<&'mo Array>>,
    mask_lhs: impl Into<Option<&'lhs Array>>,
    mask_rhs: impl Into<Option<&'rhs Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mask_out_ptr = mask_out
        .into()
        .map(|m| m.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let mask_lhs_ptr = mask_lhs
        .into()
        .map(|m| m.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let mask_rhs_ptr = mask_rhs
        .into()
        .map(|m| m.as_ptr())
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_block_masked_mm(
                a_ptr,
                b_ptr,
                block_size.into().unwrap_or(32),
                mask_out_ptr,
                mask_lhs_ptr,
                mask_rhs_ptr,
                stream.as_ref().as_ptr()
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn addmm_device<'c, 'a, 'b>(
    c: impl ScalarOrArray<'c>,
    a: impl ScalarOrArray<'a>,
    b: impl ScalarOrArray<'b>,
    alpha: impl Into<Option<f32>>,
    beta: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let c_ptr = c.into_owned_or_ref_array().as_ref().as_ptr();
    let a_ptr = a.into_owned_or_ref_array().as_ref().as_ptr();
    let b_ptr = b.into_owned_or_ref_array().as_ref().as_ptr();
    let alpha = alpha.into().unwrap_or(1.0);
    let beta = beta.into().unwrap_or(1.0);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_addmm(
                c_ptr,
                a_ptr,
                b_ptr,
                alpha,
                beta,
                stream.as_ref().as_ptr()
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn inner_device(a: &Array, b: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_inner(a.as_ptr(), b.as_ptr(), stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn outer_device(a: &Array, b: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_outer(a.as_ptr(), b.as_ptr(), stream.as_ref().as_ptr())
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[derive(Debug)]
pub enum TensorDotDims<'a> {
    Int(i32),
    List((&'a [i32], &'a [i32])),
}

impl From<i32> for TensorDotDims<'_> {
    fn from(i: i32) -> Self {
        TensorDotDims::Int(i)
    }
}

impl<'a> From<(&'a [i32], &'a [i32])> for TensorDotDims<'a> {
    fn from((lhs, rhs): (&'a [i32], &'a [i32])) -> Self {
        TensorDotDims::List((lhs, rhs))
    }
}

#[default_device]
pub fn tensordot_device<'a>(
    a: &Array,
    b: &Array,
    dims: impl Into<TensorDotDims<'a>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let s = stream.as_ref().as_ptr();

    unsafe {
        let c_array = match dims.into() {
            TensorDotDims::Int(dim) => mlx_sys::mlx_tensordot_along_axis(a_ptr, b_ptr, dim, s),
            TensorDotDims::List((lhs, rhs)) => mlx_sys::mlx_tensordot(
                a_ptr,
                b_ptr,
                lhs.as_ptr(),
                lhs.len(),
                rhs.as_ptr(),
                rhs.len(),
                s,
            ),
        };

        Ok(Array::from_ptr(c_array))
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;
    use crate::{
        array, complex64,
        ops::{all_close, broadcast_to, full, ones, split_equal},
        Dtype,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_abs() {
        let data = [1i32, 2, -3, -4, -5];
        let mut array = Array::from_slice(&data, &[5]);
        let mut result = array.abs();

        let data: &[i32] = result.as_slice();
        assert_eq!(data, [1, 2, 3, 4, 5]);

        // test that previous array is not modified and valid
        let data: &[i32] = array.as_slice();
        assert_eq!(data, [1, 2, -3, -4, -5]);
    }

    #[test]
    fn test_add() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a + &b;

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
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a - &b;

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
        let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.negative().unwrap();

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
        let mut b = a.logical_not();

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true]);
    }

    #[test]
    fn test_mul() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a * &b;

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
    fn test_div() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a / &b;

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
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);

        let mut c = a.power(&b).unwrap();

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
        let mut a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
        let mut b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);

        let mut c = &a % &b;

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
        let mut a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
        let mut b = a.sqrt();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_cos() {
        let mut a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.cos();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.54030234, -0.41614687]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_exp() {
        let mut a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.exp();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.7182817, 7.389056]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_floor() {
        let mut a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
        let mut b = a.floor().unwrap();

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
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = a.floor_divide(&b).unwrap();

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
    fn test_log() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.log();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.6931472, 1.0986123]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_log2() {
        let mut a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
        let mut b = a.log2();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_log10() {
        let mut a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
        let mut b = a.log10();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 10.0, 100.0]);
    }

    #[test]
    fn test_log1p() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.log1p();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.6931472, 1.0986123, 1.3862944]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matmul() {
        let mut a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let mut b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);

        let mut c = a.matmul(&b).unwrap();

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
        let mut a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let mut b = a.reciprocal();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.5, 0.25]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_round() {
        let mut a = Array::from_slice(&[1.1, 2.9, 3.5], &[3]);
        let mut b = a.round(None);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 3.0, 4.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.1, 2.9, 3.5]);
    }

    #[test]
    fn test_rsqrt() {
        let mut a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let mut b = a.rsqrt();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.70710677, 0.5]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_sin() {
        let mut a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.sin();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.841471, 0.9092974]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_square() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.square();

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
        assert_eq!(-array![], array![]);

        // Throws on bool
        let x = array!(true);
        assert!(negative(&x).is_err());
    }

    #[test]
    fn test_unary_abs() {
        let x = array![-1.0, 0.0, 1.0];
        assert_eq!(abs(&x), array![1.0, 0.0, 1.0]);

        // works on empty array
        assert_eq!(abs(&array![]), array![]);

        // int32
        let x = array![-1, 0, 1];
        assert_eq!(abs(&x), array![1, 0, 1]);

        // uint32
        let x = array![1u32, 0, 1];
        assert_eq!(abs(&x), array![1u32, 0, 1]);

        // bool
        let x = array![false, true];
        assert_eq!(abs(&x), array![false, true]);
    }

    #[test]
    fn test_unary_sign() {
        let x = array![-1.0, 0.0, 1.0];
        assert_eq!(sign(&x), x);

        // works on empty array
        assert_eq!(sign(&array![]), array![]);

        // int32
        let x = array![-1, 0, 1];
        assert_eq!(sign(&x), x);

        // uint32
        let x = array![1u32, 0, 1];
        assert_eq!(sign(&x), x);

        // bool
        let x = array![false, true];
        assert_eq!(sign(&x), x);
    }

    const NEG_INF: f32 = std::f32::NEG_INFINITY;

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

        let x = array![1.0, 1.0].as_type::<complex64>();
        assert!(floor(&x).is_err());
        assert!(ceil(&x).is_err());
    }

    #[test]
    fn test_unary_round() {
        let x = array![0.5, -0.5, 1.5, -1.5, 2.3, 2.6];
        assert_eq!(round(&x, None), array![0, 0, 2, -2, 2, 3]);

        let x = array![11, 222, 32];
        assert_eq!(round(&x, -1), array![10, 220, 30]);
    }

    #[test]
    fn test_unary_exp() {
        let x = array![0.0];
        assert_eq!(exp(&x).item::<f32>(), 1.0);

        let x = array![2.0];
        assert_float_eq! {
            exp(&x).item::<f32>(),
            2.0f32.exp(),
            abs <= 1e-5
        };

        assert_eq!(exp(&array![]), array![]);

        let x = array![NEG_INF];
        assert_eq!(exp(&x).item::<f32>(), 0.0);

        // Integer input type
        let x = array![2];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            exp(&x).item::<f32>(),
            2.0f32.exp(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(1.0), &[2, 2, 2]).unwrap();
        let res = exp(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 1.0f32.exp()).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split_equal(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.exp(), 2.0f32.exp()], &[2, 1]);
        assert!(all_close(&exp(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_expm1() {
        let x = array![-1.0];
        assert_float_eq! {
            expm1(&x).item::<f32>(),
            (-1.0f32).exp_m1(),
            abs <= 1e-5
        };

        let x = array![1.0];
        assert_float_eq! {
            expm1(&x).item::<f32>(),
            1.0f32.exp_m1(),
            abs <= 1e-5
        };

        // Integer input type
        let x = array![1];
        assert_eq!(expm1(&x).dtype(), Dtype::Float32);
        assert_float_eq! {
            expm1(&x).item::<f32>(),
            1.0f32.exp_m1(),
            abs <= 1e-5
        };
    }

    #[test]
    fn test_unary_sin() {
        let x = array![0.0];
        assert_eq!(sin(&x).item::<f32>(), 0.0);

        let x = array![std::f32::consts::PI / 2.0];
        assert_float_eq! {
            sin(&x).item::<f32>(),
            (std::f32::consts::PI / 2.0f32).sin(),
            abs <= 1e-5
        };

        assert_eq!(sin(&array![]), array![]);

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            sin(&x).item::<f32>(),
            0.0f32.sin(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(1.0), &[2, 2, 2]).unwrap();
        let res = sin(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 1.0f32.sin()).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split_equal(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.sin(), 2.0f32.sin()], &[2, 1]);
        assert!(all_close(&sin(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_cos() {
        let x = array![0.0];
        assert_float_eq! {
            cos(&x).item::<f32>(),
            0.0f32.cos(),
            abs <= 1e-5
        };

        let x = array![std::f32::consts::PI / 2.0];
        assert_float_eq! {
            cos(&x).item::<f32>(),
            (std::f32::consts::PI / 2.0f32).cos(),
            abs <= 1e-5
        };

        assert_eq!(cos(&array![]), array![]);

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_float_eq! {
            cos(&x).item::<f32>(),
            0.0f32.cos(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(1.0), &[2, 2, 2]).unwrap();
        let res = cos(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 1.0f32.cos()).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let x = split_equal(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0f32.cos(), 2.0f32.cos()], &[2, 1]);
        assert!(all_close(&cos(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_degrees() {
        let x = array![0.0];
        assert_eq!(degrees(&x).item::<f32>(), 0.0);

        let x = array![std::f32::consts::PI / 2.0];
        assert_eq!(degrees(&x).item::<f32>(), 90.0);

        assert_eq!(degrees(&array![]), array![]);

        // Integer input type
        let x = array![0];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_eq!(degrees(&x).item::<f32>(), 0.0);

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(std::f32::consts::PI / 2.0), &[2, 2, 2]).unwrap();
        let res = degrees(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 90.0).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let angles = Array::from_slice(&[0.0, PI / 2.0, PI, 1.5 * PI], &[2, 2]);
        let x = split_equal(&angles, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0, 180.0], &[2, 1]);
        assert!(all_close(&degrees(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_radians() {
        let x = array![0.0];
        assert_eq!(radians(&x).item::<f32>(), 0.0);

        let x = array![90.0];
        assert_eq!(radians(&x).item::<f32>(), std::f32::consts::PI / 2.0);

        assert_eq!(radians(&array![]), array![]);

        // Integer input type
        let x = array![90];
        assert_eq!(x.dtype(), Dtype::Int32);
        assert_eq!(radians(&x).item::<f32>(), std::f32::consts::PI / 2.0);

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(90.0), &[2, 2, 2]).unwrap();
        let res = radians(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], std::f32::consts::PI / 2.0).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let angles = Array::from_slice(&[0.0, 90.0, 180.0, 270.0], &[2, 2]);
        let x = split_equal(&angles, 2, 1).unwrap();
        let expected = Array::from_slice(&[0.0, PI], &[2, 1]);
        assert!(all_close(&radians(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_log() {
        let x = array![0.0];
        assert_eq!(log(&x).item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log(&x).item::<f32>(), 0.0);

        // Integer input type
        let x = array![1];
        assert_eq!(log(&x).dtype(), Dtype::Float32);
        assert_eq!(log(&x).item::<f32>(), 0.0);

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(1.0), &[2, 2, 2]).unwrap();
        let res = log(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 0.0).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = split_equal(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[1.0f32.ln(), 3.0f32.ln()], &[2, 1]);
        assert!(all_close(&log(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_log2() {
        let x = array![0.0];
        assert_eq!(log2(&x).item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log2(&x).item::<f32>(), 0.0);

        let x = array![1024.0];
        assert_eq!(log2(&x).item::<f32>(), 10.0);
    }

    #[test]
    fn test_unary_log10() {
        let x = array![0.0];
        assert_eq!(log10(&x).item::<f32>(), NEG_INF);

        let x = array![1.0];
        assert_eq!(log10(&x).item::<f32>(), 0.0);

        let x = array![1000.0];
        assert_eq!(log10(&x).item::<f32>(), 3.0);
    }

    #[test]
    fn test_unary_log1p() {
        let x = array![-1.0];
        assert_float_eq! {
            log1p(&x).item::<f32>(),
            (-1.0f32).ln_1p(),
            abs <= 1e-5
        };

        let x = array![1.0];
        assert_float_eq! {
            log1p(&x).item::<f32>(),
            1.0f32.ln_1p(),
            abs <= 1e-5
        };

        // Integer input type
        let x = array![1];
        assert_eq!(log1p(&x).dtype(), Dtype::Float32);
        assert_float_eq! {
            log1p(&x).item::<f32>(),
            1.0f32.ln_1p(),
            abs <= 1e-5
        };

        // Input is irregularly strided
        let x = broadcast_to(&Array::from_float(1.0), &[2, 2, 2]).unwrap();
        let res = log1p(&x);
        let expected = Array::full::<f32>(&[2, 2, 2], 1.0f32.ln_1p()).unwrap();
        assert!(all_close(&res, &expected, None, None, None)
            .unwrap()
            .item::<bool>());

        let data = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = split_equal(&data, 2, 1).unwrap();
        let expected = Array::from_slice(&[1.0f32.ln_1p(), 3.0f32.ln_1p()], &[2, 1]);
        assert!(all_close(&log1p(&x[0]), &expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_unary_sigmoid() {
        let x = array![0.0];
        assert_float_eq! {
            sigmoid(&x).item::<f32>(),
            0.5,
            abs <= 1e-5
        };

        // Integer input type
        let x = array![0];
        assert_eq!(sigmoid(&x).dtype(), Dtype::Float32);
        assert_float_eq! {
            sigmoid(&x).item::<f32>(),
            0.5,
            abs <= 1e-5
        };

        let inf = std::f32::INFINITY;
        let x = array![inf];
        assert_eq!(sigmoid(&x).item::<f32>(), 1.0);

        let x = array![-inf];
        assert_eq!(sigmoid(&x).item::<f32>(), 0.0);
    }

    #[test]
    fn test_unary_square() {
        let x = array![3.0];
        assert_eq!(square(&x).item::<f32>(), 9.0);

        let x = array![2];
        assert_eq!(square(&x).item::<i32>(), 4);

        let x = Array::full::<f32>(&[3, 3], 2.0).unwrap();
        assert!(all_close(
            &square(&x),
            &Array::full::<f32>(&[3, 3], 4.0).unwrap(),
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
        assert_eq!(sqrt(&x).item::<f32>(), 2.0);
        assert_eq!(rsqrt(&x).item::<f32>(), 0.5);

        let x = Array::full::<f32>(&[3, 3], 9.0).unwrap();
        assert!(all_close(
            &sqrt(&x),
            &Array::full::<f32>(&[3, 3], 3.0).unwrap(),
            None,
            None,
            None
        )
        .unwrap()
        .item::<bool>());

        let x = array![4i32];
        assert_eq!(sqrt(&x).item::<f32>(), 2.0);
        assert_eq!(rsqrt(&x).item::<f32>(), 0.5);
    }

    #[test]
    fn test_unary_reciprocal() {
        let x = array![8.0];
        assert_eq!(reciprocal(&x).item::<f32>(), 0.125);

        let x = array![2];
        let mut out = reciprocal(&x);
        assert_eq!(out.dtype(), Dtype::Float32);
        assert_eq!(out.item::<f32>(), 0.5);

        let x = Array::full::<f32>(&[3, 3], 2.0).unwrap();
        assert!(all_close(
            &reciprocal(&x),
            &Array::full::<f32>(&[3, 3], 0.5).unwrap(),
            None,
            None,
            None
        )
        .unwrap()
        .item::<bool>());
    }

    #[test]
    fn test_binary_add() {
        let x = array![1.0];
        let y = array![1.0];
        let mut z = add(&x, &y).unwrap();
        assert_eq!(z.item::<f32>(), 2.0);

        let mut z = &x + y;
        assert_eq!(z.item::<f32>(), 2.0);

        let mut z = add(z, &x).unwrap();
        assert_eq!(z.item::<f32>(), 3.0);

        // Chain a few adds:
        let mut out = x.clone();
        for _ in 0..10 {
            out = add(&out, &x).unwrap();
        }
        assert_eq!(out.item::<f32>(), 11.0);

        // Works for different shapes
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let z = add(&x, &y).unwrap();
        assert_eq!(z.shape(), &[3]);
        assert_eq!(z, array![2.0, 4.0, 6.0]);

        // Works with scalars
        let x = array![1.0, 2.0, 3.0];
        let y = &x + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        assert_eq!(y, array![3.0, 4.0, 5.0]);
        let y = &x + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        assert_eq!(y, array![3.0, 4.0, 5.0]);

        // Check type promotion
        let y = x + 2;
        assert_eq!(y.dtype(), Dtype::Float32);

        let y = array![1, 2, 3] + 2.0;
        assert_eq!(y.dtype(), Dtype::Float32);
        // assert!(array_equal(&y, &array![3.0, 4.0, 5.0]).item::<bool>());
        assert_eq!(y, array![3.0, 4.0, 5.0]);

        // Broadcasting works
        let x = broadcast_to(&array![1.0], &[10]).unwrap();
        let y = broadcast_to(&array![2.0], &[10]).unwrap();
        let z = add(&x, &y).unwrap();
        assert_eq!(z, full::<f32>(&[10], 3.0).unwrap());

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
        let x = array![];
        let y = array![];
        let mut z = x + y;
        z.eval();
        assert_eq!(z.size(), 0);
        assert_eq!(z.shape(), &[0]);
    }

    #[test]
    fn test_binary_sub() {
        let x = array![3.0, 2.0, 1.0];
        let y = array![1.0, 1.0, 1.0];
        assert_eq!(x - y, array![2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_binary_mul() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 2.0, 2.0];
        assert_eq!(x * y, array![2.0, 4.0, 6.0]);
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
            log_add_exp(&x, &y).unwrap().item::<f32>(),
            2.0f32.ln(),
            abs <= 1e-5
        };

        let x = array![0u32];
        let y = array![10000u32];
        assert_eq!(log_add_exp(&x, &y).unwrap().item::<f32>(), 10000.0);

        let x = array![std::f32::INFINITY];
        let y = array![3.0];
        assert_eq!(
            log_add_exp(&x, &y).unwrap().item::<f32>(),
            std::f32::INFINITY
        );

        let x = array![std::f32::NEG_INFINITY];
        let y = array![3.0];
        assert_eq!(log_add_exp(&x, &y).unwrap().item::<f32>(), 3.0);

        let x = array![std::f32::NEG_INFINITY];
        let y = array![std::f32::NEG_INFINITY];
        assert_eq!(
            log_add_exp(&x, &y).unwrap().item::<f32>(),
            std::f32::NEG_INFINITY
        );

        let x = array![std::f32::INFINITY];
        let y = array![std::f32::INFINITY];
        assert_eq!(
            log_add_exp(&x, &y).unwrap().item::<f32>(),
            std::f32::INFINITY
        );

        let x = array![std::f32::NEG_INFINITY];
        let y = array![std::f32::INFINITY];
        assert_eq!(
            log_add_exp(&x, &y).unwrap().item::<f32>(),
            std::f32::INFINITY
        );
    }
}
