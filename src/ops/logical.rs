use crate::array::Array;
use crate::error::DataStoreError;
use crate::stream::StreamOrDevice;
use crate::utils::is_broadcastable;
use mlx_macros::default_device;

impl Array {
    /// Element-wise equality.
    ///
    /// Equality comparison on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.eq(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn eq_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_eq_device(other, stream).unwrap()
    }

    /// Element-wise equality without broadcasting checks.
    ///
    /// Equality comparison on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.eq_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn eq_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_equal(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise equality returning an error if the arrays are not broadcastable.
    ///
    /// Equality comparison on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_eq(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_eq_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.eq_device_unchecked(other, stream) })
    }

    /// Element-wise less than or equal.
    ///
    /// Less than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.le(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn le_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_le_device(other, stream).unwrap()
    }

    /// Element-wise less than or equal without broadcasting checks.
    ///
    /// Less than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.le_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn le_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_less_equal(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise less than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Less than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_le(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_le_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.le_device_unchecked(other, stream) })
    }

    /// Element-wise greater than or equal.
    ///
    /// Greater than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ge(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn ge_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_ge_device(other, stream).unwrap()
    }

    /// Element-wise greater than or equal without broadcasting checks.
    ///
    /// Greater than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.ge_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn ge_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_greater_equal(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise greater than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Greater than or equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_ge(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_ge_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.ge_device_unchecked(other, stream) })
    }

    /// Element-wise not equal.
    ///
    /// Not equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ne(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn ne_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_ne_device(other, stream).unwrap()
    }

    /// Element-wise not equal without broadcasting checks.
    ///
    /// Not equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.ne_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn ne_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_not_equal(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise not equal returning an error if the arrays are not broadcastable.
    ///
    /// Not equal on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_ne(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_ne_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.ne_device_unchecked(other, stream) })
    }

    /// Element-wise less than.
    ///
    /// Less than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.lt(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn lt_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_lt_device(other, stream).unwrap()
    }

    /// Element-wise less than without broadcasting checks.
    ///
    /// Less than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.lt_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn lt_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_less(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise less than returning an error if the arrays are not broadcastable.
    ///
    /// Less than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_lt(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_lt_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.lt_device_unchecked(other, stream) })
    }

    /// Element-wise greater than.
    ///
    /// Greater than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.gt(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn gt_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_gt_device(other, stream).unwrap()
    }

    /// Element-wise greater than without broadcasting checks.
    ///
    /// Greater than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.gt_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn gt_device_unchecked(&self, other: &Array, stream: StreamOrDevice) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_greater(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise greater than returning an error if the arrays are not broadcastable.
    ///
    /// Greater than on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_gt(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_gt_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.gt_device_unchecked(other, stream) })
    }

    /// Element-wise logical and.
    ///
    /// Logical and on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_and(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn logical_and_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_logical_and_device(other, stream).unwrap()
    }

    /// Element-wise logical and without broadcasting checks.
    ///
    /// Logical and on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = unsafe { a.logical_and_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn logical_and_device_unchecked(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_logical_and(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise logical and returning an error if the arrays are not broadcastable.
    ///
    /// Logical and on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.try_logical_and(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, false, false]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_logical_and_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.logical_and_device_unchecked(other, stream) })
    }

    /// Element-wise logical or.
    ///
    /// Logical or on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_or(&b);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn logical_or_device(&self, other: &Array, stream: StreamOrDevice) -> Array {
        self.try_logical_or_device(other, stream).unwrap()
    }

    /// Element-wise logical or without broadcasting checks.
    ///
    /// Logical or on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = unsafe { a.logical_or_unchecked(&b) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn logical_or_device_unchecked(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_logical_or(
                self.c_array,
                other.c_array,
                stream.as_ptr(),
            ))
        }
    }

    /// Element-wise logical or returning an error if the arrays are not broadcastable.
    ///
    /// Logical or on two arrays with <doc:broadcasting>.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.try_logical_or(&b).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_logical_or_device(
        &self,
        other: &Array,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.logical_or_device_unchecked(other, stream) })
    }

    /// Approximate comparison of two arrays.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```text
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]).pow(&(0.5.into()));
    /// let mut c = a.all_close(&b, None, None, None);
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - rtol: relative tolerance = defaults to 1e-5 when None
    /// - atol: absolute tolerance - defaults to 1e-8 when None
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn all_close_device(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_all_close_device(other, rtol, atol, equal_nan, stream)
            .unwrap()
    }

    /// Approximate comparison of two arrays without validating inputs.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```text
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]).pow(&(0.5.into()));
    /// let mut c = unsafe { a.all_close_unchecked(&b, None, None, None) };
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - rtol: relative tolerance = defaults to 1e-5 when None
    /// - atol: absolute tolerance - defaults to 1e-8 when None
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate inputs.
    #[default_device]
    pub unsafe fn all_close_device_unchecked(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        let is_close = self.is_close_device_unchecked(other, rtol, atol, equal_nan, stream.clone());
        is_close.all_device(None, stream)
    }

    /// Approximate comparison of two arrays returning an error if the inputs aren't valid.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```text
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]).pow(&(0.5.into()));
    /// let mut c = a.try_all_close(&b, None, None, None).unwrap();
    ///
    /// c.eval();
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - rtol: relative tolerance = defaults to 1e-5 when None
    /// - atol: absolute tolerance - defaults to 1e-8 when None
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_all_close_device(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        let is_close = self.try_is_close_device(other, rtol, atol, equal_nan, stream.clone());
        is_close
            .map(|is_close| is_close.all_device(None, stream))
            .map_err(|error| error)
    }

    /// Returns a boolean array where two arrays are element-wise equal within a tolerance.
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
    /// Unlike [self.array_eq] this function supports <doc:broadcasting>.
    #[default_device]
    pub fn is_close_device(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_is_close_device(other, rtol, atol, equal_nan, stream)
            .unwrap()
    }

    /// Returns a boolean array where two arrays are element-wise equal within a tolerance without broadcasting checks.
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
    /// Unlike [self.array_eq] this function supports <doc:broadcasting>.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arrays are broadcastable.
    #[default_device]
    pub unsafe fn is_close_device_unchecked(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_isclose(
                self.c_array,
                other.c_array,
                rtol.into().unwrap_or(1e-5),
                atol.into().unwrap_or(1e-8),
                equal_nan.into().unwrap_or(false),
                stream.as_ptr(),
            ))
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
    /// Unlike [self.array_eq] this function supports <doc:broadcasting>.
    #[default_device]
    pub fn try_is_close_device(
        &self,
        other: &Array,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Result<Array, DataStoreError> {
        // represents atol and rtol being broadcasted to operate on other
        if !is_broadcastable(&[], other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        if !is_broadcastable(self.shape(), other.shape()) {
            return Err(DataStoreError::BroadcastError);
        }

        Ok(unsafe { self.is_close_device_unchecked(other, rtol, atol, equal_nan, stream) })
    }

    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `True` if and only if the arrays
    /// have the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
    ///
    /// let c = a.array_eq(&b, None);
    /// // c == [true]
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn array_eq_device(
        &self,
        other: &Array,
        equal_nan: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            Array::from_ptr(mlx_sys::mlx_array_equal(
                self.c_array,
                other.c_array,
                equal_nan.into().unwrap_or(false),
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
    fn test_eq() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.eq(&b);

        c.eval();
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
        let c = a.try_eq(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_le() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.le(&b);

        c.eval();
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
        let c = a.try_le(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ge() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.ge(&b);

        c.eval();
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
        let c = a.try_ge(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ne() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.ne(&b);

        c.eval();
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
        let c = a.try_ne(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_lt() {
        let a = Array::from_slice(&[1, 0, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.lt(&b);

        c.eval();
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
        let c = a.try_lt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_gt() {
        let a = Array::from_slice(&[1, 4, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.gt(&b);

        c.eval();
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
        let c = a.try_gt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_and() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let mut c = a.logical_and(&b);

        c.eval();
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
        let c = a.try_logical_and(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_or() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let mut c = a.logical_or(&b);

        c.eval();
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
        let c = a.try_logical_or(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_all_close() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4]).pow(&(0.5.into()));
        let mut c = a.all_close(&b, 1e-5, None, None);

        c.eval();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_all_close_invalid_broadcast() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3., 4.], &[5]);
        let c = a.try_all_close(&b, 1e-5, None, None);
        assert!(c.is_err());
    }

    #[test]
    fn test_is_close_false() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let mut c = a.is_close(&b, None, None, false);

        c.eval();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);
    }

    #[test]
    fn test_is_close_true() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let mut c = a.is_close(&b, 0.1, 0.2, true);

        c.eval();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);
    }

    #[test]
    fn test_is_close_invalid_broadcast() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3, 4.4], &[4]);
        let c = a.try_is_close(&b, None, None, false);
        assert!(c.is_err());
    }

    #[test]
    fn test_array_eq() {
        let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let mut c = a.array_eq(&b, None);

        c.eval();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }
}