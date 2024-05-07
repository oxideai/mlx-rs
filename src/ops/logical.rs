use crate::array::Array;
use crate::error::{DataStoreError, OperationError};
use crate::stream::StreamOrDevice;
use crate::utils::{axes_or_default_to_all, can_reduce_shape, is_broadcastable};
use mlx_macros::default_device;

impl Array {
    /// Element-wise equality.
    ///
    /// Equality comparison on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.eq(&b);
    ///
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
    /// Equality comparison on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.eq_unchecked(&b) };
    ///
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
    /// Equality comparison on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_eq(&b).unwrap();
    ///
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
    /// Less than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.le(&b);
    ///
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
    /// Less than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.le_unchecked(&b) };
    ///
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
    /// Less than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_le(&b).unwrap();
    ///
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
    /// Greater than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ge(&b);
    ///
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
    /// Greater than or equal on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.ge_unchecked(&b) };
    ///
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
    /// Greater than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_ge(&b).unwrap();
    ///
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
    /// Not equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ne(&b);
    ///
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
    /// Not equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.ne_unchecked(&b) };
    ///
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
    /// Not equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_ne(&b).unwrap();
    ///
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
    /// Less than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.lt(&b);
    ///
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
    /// Less than on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.lt_unchecked(&b) };
    ///
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
    /// Less than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_lt(&b).unwrap();
    ///
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
    /// Greater than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.gt(&b);
    ///
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
    /// Greater than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = unsafe { a.gt_unchecked(&b) };
    ///
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
    /// Greater than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.try_gt(&b).unwrap();
    ///
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
    /// Logical and on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_and(&b);
    ///
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
    /// Logical and on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = unsafe { a.logical_and_unchecked(&b) };
    ///
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
    /// Logical and on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.try_logical_and(&b).unwrap();
    ///
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
    /// Logical or on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_or(&b);
    ///
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
    /// Logical or on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = unsafe { a.logical_or_unchecked(&b) };
    ///
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
    /// Logical or on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.try_logical_or(&b).unwrap();
    ///
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
        unsafe {
            Array::from_ptr(mlx_sys::mlx_allclose(
                self.c_array,
                other.c_array,
                rtol.into().unwrap_or(1e-5),
                atol.into().unwrap_or(1e-8),
                equal_nan.into().unwrap_or(false),
                stream.as_ptr(),
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
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx::Array;
    /// let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt();
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]).pow(&(0.5.into()));
    /// let mut c = a.try_all_close(&b, None, None, None).unwrap();
    ///
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
        is_close.map(|is_close| is_close.all_device(None, None, stream))
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
    /// Unlike [self.array_eq] this function supports [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
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
    /// Unlike [self.array_eq] this function supports [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
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
    /// Unlike [self.array_eq] this function supports [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
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
    /// Compare two arrays for equality. Returns `true` iff the arrays have
    /// the same shape and their values are equal. The arrays need not have
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

    /// An `or` reduction over the given axes.
    ///
    ///  # Example
    /// ```rust
    /// use mlx::Array;
    ///
    /// let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    ///
    /// // will produce a scalar Array with true -- some of the values are non-zero
    /// let all = array.any(None, None);
    ///
    /// // produces an Array([true, true, true, true]) -- all rows have non-zeros
    /// let all_rows = array.any(&[0][..], None);
    /// ```
    ///
    /// # Params
    /// - axes: axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: if `true` keep reduced axis as singleton dimension -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn any_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_any_device(axes, keep_dims, stream).unwrap()
    }

    /// An `or` reduction over the given axes without validating axes are valid for the array.
    ///
    ///  # Example
    /// ```rust
    /// use mlx::Array;
    ///
    /// let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    ///
    /// // will produce a scalar Array with true -- some of the values are non-zero
    /// let all = unsafe { array.any_unchecked(None, None) };
    ///
    /// // produces an Array([true, true, true, true]) -- all rows have non-zeros
    /// let all_rows = unsafe { array.any_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    /// - axes: axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: if `true` keep reduced axis as singleton dimension -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn any_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            Array::from_ptr(mlx_sys::mlx_any(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ptr(),
            ))
        }
    }

    /// An `or` reduction over the given axes returning an error if the axes are invalid.
    ///
    ///  # Example
    /// ```rust
    /// use mlx::Array;
    ///
    /// let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    ///
    /// // will produce a scalar Array with true -- some of the values are non-zero
    /// let all = array.try_any(None, None).unwrap();
    ///
    /// // produces an Array([true, true, true, true]) -- all rows have non-zeros
    /// let all_rows = array.try_any(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    /// - axes: axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: if `true` keep reduced axis as singleton dimension -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_any_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: StreamOrDevice,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_any(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ptr(),
            ))
        })
    }
}

/// Select from `a` or `b` according to `condition`.
///
/// The condition and input arrays must be the same shape or [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting)
/// with each another.
///
/// # Params:
/// - condition: condition array
/// - a: input selected from where condition is non-zero or `true`
/// - b: input selected from where condition is zero or `false`
/// - stream: stream or device to evaluate on
#[default_device]
pub fn which_device(condition: &Array, a: &Array, b: &Array, stream: StreamOrDevice) -> Array {
    try_which_device(condition, a, b, stream).unwrap()
}

/// Select from `a` or `b` according to `condition` without broadcasting checks.
///
/// The condition and input arrays must be the same shape or [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting)
/// with each another.
///
/// # Params
/// - condition: condition array
/// - a: input selected from where condition is non-zero or `true`
/// - b: input selected from where condition is zero or `false`
/// - stream: stream or device to evaluate on
///
/// # Safety
///
/// This function is unsafe because it does not check if the arrays are broadcastable.
#[default_device]
pub unsafe fn which_device_unchecked(
    condition: &Array,
    a: &Array,
    b: &Array,
    stream: StreamOrDevice,
) -> Array {
    unsafe {
        Array::from_ptr(mlx_sys::mlx_where(
            condition.c_array,
            a.c_array,
            b.c_array,
            stream.as_ptr(),
        ))
    }
}

/// Select from `a` or `b` according to `condition` returning an error if the arrays are not broadcastable.
///
/// The condition and input arrays must be the same shape or [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting)
/// with each another.
///
/// # Params
/// - condition: condition array
/// - a: input selected from where condition is non-zero or `true`
/// - b: input selected from where condition is zero or `false`
/// - stream: stream or device to evaluate on
#[default_device]
pub fn try_which_device(
    condition: &Array,
    a: &Array,
    b: &Array,
    stream: StreamOrDevice,
) -> Result<Array, DataStoreError> {
    if !is_broadcastable(condition.shape(), a.shape())
        || !is_broadcastable(condition.shape(), b.shape())
    {
        return Err(DataStoreError::BroadcastError);
    }

    Ok(unsafe { which_device_unchecked(condition, a, b, stream) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Pow;

    #[test]
    fn test_eq() {
        let mut a = Array::from_slice(&[1, 2, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.eq(&b);

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
        let mut a = Array::from_slice(&[1, 2, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.le(&b);

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
        let mut a = Array::from_slice(&[1, 2, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.ge(&b);

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
        let mut a = Array::from_slice(&[1, 2, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.ne(&b);

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
        let mut a = Array::from_slice(&[1, 0, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.lt(&b);

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
        let mut a = Array::from_slice(&[1, 4, 3], &[3]);
        let mut b = Array::from_slice(&[1, 2, 3], &[3]);
        let mut c = a.gt(&b);

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
        let mut a = Array::from_slice(&[true, false, true], &[3]);
        let mut b = Array::from_slice(&[true, true, false], &[3]);
        let mut c = a.logical_and(&b);

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
        let mut a = Array::from_slice(&[true, false, true], &[3]);
        let mut b = Array::from_slice(&[true, true, false], &[3]);
        let mut c = a.logical_or(&b);

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

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);
    }

    #[test]
    fn test_is_close_true() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let mut c = a.is_close(&b, 0.1, 0.2, true);

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

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_any() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let mut all = array.any(&[0][..], None);

        let results: &[bool] = all.as_slice();
        assert_eq!(results, &[true, true, true, true]);
    }

    #[test]
    fn test_any_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let mut all = array.any(&[][..], None);

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_any_out_of_bounds() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[12]);
        let result = array.try_any(&[1][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_duplicate_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let result = array.try_any(&[0, 0][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_which() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);
        let mut c = which(&condition, &a, &b);

        let c_data: &[i32] = c.as_slice();
        assert_eq!(c_data, [1, 5, 3]);
    }

    #[test]
    fn test_which_invalid_broadcast() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6, 7], &[4]);
        let c = try_which(&condition, &a, &b);
        assert!(c.is_err());
    }
}
