use crate::array::Array;
use crate::stream::StreamOrDevice;

impl Array {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Pow;

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
}
