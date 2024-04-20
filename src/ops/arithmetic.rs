use crate::array::Array;
use crate::stream::StreamOrDevice;
use mlx_macros::default_device;

impl Array {
    /// Element-wise absolute value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let array = Array::from_slice(&[1i32, 2, -3, -4, -5], &[5]);
    /// let mut result = array.abs();
    ///
    /// result.eval();
    /// let data: &[i32] = result.as_slice();
    /// // data == [1, 2, 3, 4, 5]
    /// ```
    ///
    /// # Params
    ///
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn abs_device(&self, stream: StreamOrDevice) -> Array {
        let ctx = stream.as_ptr();

        unsafe { Array::from_ptr(mlx_sys::mlx_abs(self.c_array, ctx)) }
    }

    /// Element-wise addition.
    ///
    /// Add two arrays with <doc:broadcasting>.
    ///
    /// # Example
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.add_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to add
    /// - stream: stream or device to evaluate on
    pub fn add_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_add(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise subtraction.
    ///
    /// Subtract two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.sub_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [-3.0, -3.0, -3.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to subtract
    /// - stream: stream or device to evaluate on
    pub fn sub_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_subtract(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Unary element-wise negation.
    ///
    /// Negate the values in the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = -&a;
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [-1.0, -2.0, -3.0]
    /// ```
    ///
    /// # Params
    ///
    /// - stream: stream or device to evaluate on
    pub fn neg_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_negative(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise multiplication.
    ///
    /// Multiply two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.mul_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [4.0, 10.0, 18.0]
    /// ```
    pub fn mul_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_multiply(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise division.
    ///
    /// Divide two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.div_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    /// - stream: stream or device to evaluate on
    pub fn div_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_divide(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise power operation.
    ///
    /// Raise the elements of the array to the power of the elements of another array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);
    /// let mut c = a.pow_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 8.0, 81.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to raise to the power of
    /// - stream: stream or device to evaluate on
    pub fn pow_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_power(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise remainder of division.
    ///
    /// Computes the remainder of dividing `lhs` with `rhs` with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
    /// let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);
    /// let mut c = a.rem_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [1.0, 3.0, 2.0]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    /// - stream: stream or device to evaluate on
    pub fn rem_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_remainder(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise square root
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
    /// let mut b = a.sqrt_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn sqrt_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_sqrt(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise cosine
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.cos_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.54030234, -0.41614687]
    /// ```
    #[default_device]
    pub fn cos_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_cos(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise exponential.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    ///
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
    /// let mut b = a.exp_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 2.7182817, 7.389056]
    /// ```
    #[default_device]
    pub fn exp_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_exp(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise floor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
    /// let mut b = a.floor_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn floor_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_floor(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise integer division..
    ///
    /// Divide two arrays with <doc:broadcasting>.
    ///
    /// If either array is a floating point type then it is equivalent to calling [floor()] after `/`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    /// let mut c = a.floor_divide_device(&b, Default::default());
    ///
    /// c.eval();
    /// let c_data: &[f32] = c.as_slice();
    /// // c_data == [0.25, 0.4, 0.5]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to divide
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn floor_divide_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_floor_divide(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise natural logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 0.6931472, 1.0986123]
    /// ```
    #[default_device]
    pub fn log_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise base-2 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
    /// let mut b = a.log2_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0, 3.0]
    /// ```
    #[default_device]
    pub fn log2_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log2(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise base-10 logarithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
    /// let mut b = a.log10_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.0, 1.0, 2.0]
    /// ```
    #[default_device]
    pub fn log10_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log10(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise natural log of one plus the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    /// let mut b = a.log1p_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [0.6931472, 1.0986123, 1.3862944]
    /// ```
    #[default_device]
    pub fn log1p_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_log1p(self.c_array, stream.as_ptr())) }
    }

    /// Matrix multiplication.
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
    ///   standard <doc:broadcasting>.
    ///
    /// # Example
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
    /// let b = Array::from_slice(&[-5.0, 37.5, 4., 7., 1., 0.], &[2, 3]);
    ///
    /// // produces a [2, 3] result
    /// let mut c = a.matmul_device(&b, Default::default());
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to multiply
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn matmul_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_matmul(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise reciprocal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
    /// let mut b = a.reciprocal_device(Default::default());
    ///
    /// b.eval();
    /// let b_data: &[f32] = b.as_slice();
    /// // b_data == [1.0, 0.5, 0.25]
    /// ```
    #[default_device]
    pub fn reciprocal_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_reciprocal(self.c_array, stream.as_ptr())) }
    }

    /// Round to the given number of decimals.
    ///
    /// # Params
    ///
    /// - decimals: number of decimals to round to - default is 0 if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn round_device(&self, decimals: impl Into<Option<i32>>, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_round(
                self.c_array,
                decimals.into().unwrap_or(0),
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise reciprocal and square root.
    #[default_device]
    pub fn rsqrt_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_rsqrt(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise sine.
    #[default_device]
    pub fn sin_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_sin(self.c_array, stream.as_ptr())) }
    }

    /// Element-wise square.
    #[default_device]
    pub fn square_device(&self, stream: StreamOrDevice) -> Array {
        unsafe { Array::from_ptr(mlx_sys::mlx_square(self.c_array, stream.as_ptr())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Pow;

    #[test]
    fn test_abs() {
        let data = [1i32, 2, -3, -4, -5];
        let array = Array::from_slice(&data, &[5]);
        let mut result = array.abs();

        result.eval();
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

        let mut c = &a + &b;
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[5.0, 7.0, 9.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a - &b;
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[-3.0, -3.0, -3.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_neg() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = -&a;
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[-1.0, -2.0, -3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mul() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a * &b;
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[4.0, 10.0, 18.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_div() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = &a / &b;
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[0.25, 0.4, 0.5]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pow() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[2.0, 3.0, 4.0], &[3]);

        let mut c = a.pow(&b);
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[1.0, 8.0, 81.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rem() {
        let a = Array::from_slice(&[10.0, 11.0, 12.0], &[3]);
        let b = Array::from_slice(&[3.0, 4.0, 5.0], &[3]);

        let mut c = &a % &b;
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[1.0, 3.0, 2.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[10.0, 11.0, 12.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sqrt() {
        let a = Array::from_slice(&[1.0, 4.0, 9.0], &[3]);
        let mut b = a.sqrt();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_cos() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.cos();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.54030234, -0.41614687]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_exp() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.exp();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 2.7182817, 7.389056]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_floor() {
        let a = Array::from_slice(&[0.1, 1.9, 2.5], &[3]);
        let mut b = a.floor();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.1, 1.9, 2.5]);
    }

    #[test]
    fn test_floor_divide() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);

        let mut c = a.floor_divide(&b);
        c.eval();

        let c_data: &[f32] = c.as_slice();
        assert_eq!(c_data, &[0.0, 0.0, 0.0]);

        // check a and b are not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_log() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.log();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.6931472, 1.0986123]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_log2() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0, 8.0], &[4]);
        let mut b = a.log2();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0, 3.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_log10() {
        let a = Array::from_slice(&[1.0, 10.0, 100.0], &[3]);
        let mut b = a.log10();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 1.0, 2.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 10.0, 100.0]);
    }

    #[test]
    fn test_log1p() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.log1p();
        b.eval();

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

        let mut c = a.matmul(&b);
        c.eval();

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
    fn test_reciprocal() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let mut b = a.reciprocal();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.5, 0.25]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_round() {
        let a = Array::from_slice(&[1.1, 2.9, 3.5], &[3]);
        let mut b = a.round(None);
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 3.0, 4.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.1, 2.9, 3.5]);
    }

    #[test]
    fn test_rsqrt() {
        let a = Array::from_slice(&[1.0, 2.0, 4.0], &[3]);
        let mut b = a.rsqrt();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 0.70710677, 0.5]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_sin() {
        let a = Array::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let mut b = a.sin();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[0.0, 0.841471, 0.9092974]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_square() {
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut b = a.square();
        b.eval();

        let b_data: &[f32] = b.as_slice();
        assert_eq!(b_data, &[1.0, 4.0, 9.0]);

        // check a is not modified
        let a_data: &[f32] = a.as_slice();
        assert_eq!(a_data, &[1.0, 2.0, 3.0]);
    }
}
