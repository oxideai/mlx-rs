use crate::array::Array;
use crate::error::Exception;
use crate::stream::StreamOrDevice;

use crate::Stream;
use mlx_macros::default_device;

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
    /// # Example
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.add_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to add
    #[default_device]
    pub fn add_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_add(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise subtraction returning an error if arrays are not broadcastable.
    ///
    /// Subtract two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.sub_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [-3.0, -3.0, -3.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to subtract
    #[default_device]
    pub fn sub_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_subtract(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Unary element-wise negation.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the array is of type bool.
    #[default_device]
    pub fn neg_device(&self, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_negative(self.c_array, stream.as_ref().as_ptr())
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
    /// let mut c = a.mul_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [4.0, 10.0, 18.0]
    /// ```
    #[default_device]
    pub fn mul_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_multiply(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise division returning an error if arrays are not broadcastable.
    ///
    /// Divide two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.div_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    #[default_device]
    pub fn div_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_divide(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise power operation returning an error if arrays are not broadcastable if they have different shapes.
    ///
    /// Raise the elements of the array to the power of the elements of another array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);
    /// let mut c = a.pow_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 8.0, 81.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to raise to the power of
    #[default_device]
    pub fn pow_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_power(self.c_array, other.c_array, stream.as_ref().as_ptr())
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// Element-wise remainder of division returning an error if arrays are not broadcastable.
    ///
    /// Computes the remainder of dividing `lhs` with `rhs` with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
    /// let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);
    /// let mut c = a.rem_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 3.0, 2.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    #[default_device]
    pub fn rem_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_remainder(self.c_array, other.c_array, stream.as_ref().as_ptr())
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
    /// let mut b = a.floor_device(StreamOrDevice::default());
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
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.floor_divide_device(&b, StreamOrDevice::default());
    ///
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    #[default_device]
    pub fn floor_divide_device(
        &self,
        other: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_floor_divide(self.c_array, other.c_array, stream.as_ref().as_ptr())
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
    /// # Example
    /// ```rust
    /// use mlx_rs::prelude::*;
    /// let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
    /// let b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);
    ///
    /// // produces a [2, 3] result
    /// let mut c = a.matmul_device(&b, StreamOrDevice::default());
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to multiply
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex64;
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
        let c = a.sub(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_neg() {
        let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.neg().unwrap();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[-1.0, -2.0, -3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_neg_bool() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = a.neg();
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
        let c = a.mul(&b);
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
        let c = a.div(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_pow() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);

        let mut c = a.pow(&b).unwrap();

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
        let c = a.pow(&b);
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
        let c = a.rem(&b);
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
}
