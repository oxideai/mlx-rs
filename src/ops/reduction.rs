use crate::array::Array;
use crate::error::OperationError;
use crate::stream::StreamOrDevice;
use crate::utils::{axes_or_default_to_all, can_reduce_shape};
use crate::Stream;
use mlx_macros::default_device;

impl Array {
    /// An `and` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.all(&[0][..], None);
    ///
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    #[default_device]
    pub fn all_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_all_device(axes, keep_dims, stream).unwrap()
    }

    /// An `and` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = unsafe { a.all_unchecked(&[0][..], None) };
    ///
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn all_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_all_axes(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// An `and` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.try_all(&[0][..], None).unwrap();
    ///
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    #[default_device]
    pub fn try_all_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_all_axes(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// A `product` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = array.prod(&[0][..], None);
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: The stream to execute the operation on
    #[default_device]
    pub fn prod_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_prod_device(axes, keep_dims, stream).unwrap()
    }

    /// A `product` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = unsafe { array.prod_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn prod_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_prod(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// A `product` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = array.try_prod(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_prod_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_prod(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// A `max` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = array.max(&[0][..], None);
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn max_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_max_device(axes, keep_dims, stream).unwrap()
    }

    /// A `max` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = unsafe { array.max_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn max_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_max(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// A `max` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = array.try_max(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_max_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        if self.size() == 0 {
            return Err(OperationError::NotSupported(
                "max reduction is not supported for empty arrays".to_string(),
            ));
        }

        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_max(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// Sum reduce the array over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = array.sum(&[0][..], None);
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn sum_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_sum_device(axes, keep_dims, stream).unwrap()
    }

    /// Sum reduce the array over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = unsafe { array.sum_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn sum_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_sum(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// Sum reduce the array over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = array.try_sum(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_sum_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_sum(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// A `mean` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean(&[0][..], None);
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn mean_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_mean_device(axes, keep_dims, stream).unwrap()
    }

    /// A `mean` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = unsafe { array.mean_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn mean_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_mean(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// A `mean` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.try_mean(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_mean_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        let ndim = self.ndim() as i32;
        for &axis in axes.iter() {
            if axis < -ndim || axis >= ndim {
                return Err(OperationError::AxisOutOfBounds {
                    axis,
                    dim: ndim as usize,
                });
            }
        }

        // check done by inner sum
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_mean(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// A `min` reduction over the given axes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = array.min(&[0][..], None);
    /// ```
    ///
    /// # Params
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn min_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_min_device(axes, keep_dims, stream).unwrap()
    }

    /// A `min` reduction over the given axes without validating axes are valid for the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = unsafe { array.min_unchecked(&[0][..], None) };
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn min_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_min(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// A `min` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = array.try_min(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_min_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        if self.size() == 0 {
            return Err(OperationError::NotSupported(
                "min reduction is not supported for empty arrays".to_string(),
            ));
        }

        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_min(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// Compute the variance(s) over the given axes
    ///
    /// # Params
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn variance_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_variance_device(axes, keep_dims, ddof, stream)
            .unwrap()
    }

    /// Compute the variance(s) over the given axes without validating axes are valid for the array.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn variance_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_var(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            ddof.into().unwrap_or(0),
            stream.as_ref().as_ptr(),
        ))
    }

    /// Compute the variance(s) over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_variance_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // add the mean check on a
        let ndim = self.ndim() as i32;
        for &axis in axes.iter() {
            if axis < -ndim || axis >= ndim {
                return Err(OperationError::AxisOutOfBounds {
                    axis,
                    dim: ndim as usize,
                });
            }
        }

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_var(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            ))
        })
    }

    /// A `log-sum-exp` reduction over the given axes.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn log_sum_exp_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_log_sum_exp_device(axes, keep_dims, stream)
            .unwrap()
    }

    /// A `log-sum-exp` reduction over the given axes without validating axes are valid for the array.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not validate that the axes are valid for the array.
    #[default_device]
    pub unsafe fn log_sum_exp_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        Array::from_ptr(mlx_sys::mlx_logsumexp(
            self.c_array,
            axes.as_ptr(),
            axes.len(),
            keep_dims.into().unwrap_or(false),
            stream.as_ref().as_ptr(),
        ))
    }

    /// A `log-sum-exp` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    /// - stream: stream or device to evaluate on
    #[default_device]
    pub fn try_log_sum_exp_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, OperationError> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        // verify reducing shape only if axes are provided
        if !axes.is_empty() {
            can_reduce_shape(self.shape(), &axes)?;
        }

        Ok(unsafe {
            Array::from_ptr(mlx_sys::mlx_logsumexp(
                self.c_array,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_all() {
        let array = Array::from_slice(&[true, false, true, false], &[2, 2]);

        assert_eq!(array.all(None, None).item::<bool>(), false);
        assert_eq!(array.all(None, true).shape(), &[1, 1]);
        assert_eq!(array.all(&[0, 1][..], None).item::<bool>(), false);

        let mut result = array.all(&[0][..], None);
        assert_eq!(result.as_slice::<bool>(), &[true, false]);

        let mut result = array.all(&[1][..], None);
        assert_eq!(result.as_slice::<bool>(), &[false, false]);
    }

    #[test]
    fn test_all_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let mut all = array.all(&[][..], None);

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_prod() {
        let x = Array::from_slice(&[1, 2, 3, 3], &[2, 2]);
        assert_eq!(x.prod(None, None).item::<i32>(), 18);

        let mut y = x.prod(None, true);
        assert_eq!(y.item::<i32>(), 18);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.prod(&[0][..], None);
        assert_eq!(result.as_slice::<i32>(), &[3, 6]);

        let mut result = x.prod(&[1][..], None);
        assert_eq!(result.as_slice::<i32>(), &[2, 9])
    }

    #[test]
    fn test_prod_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.prod(&[][..], None);

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_max() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.max(None, None).item::<i32>(), 4);
        let mut y = x.max(None, true);
        assert_eq!(y.item::<i32>(), 4);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.max(&[0][..], None);
        assert_eq!(result.as_slice::<i32>(), &[3, 4]);

        let mut result = x.max(&[1][..], None);
        assert_eq!(result.as_slice::<i32>(), &[2, 4]);
    }

    #[test]
    fn test_max_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.max(&[][..], None);

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_sum() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.sum(&[0][..], None);

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[9, 17]);
    }

    #[test]
    fn test_sum_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.sum(&[][..], None);

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_mean() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.mean(None, None).item::<f32>(), 2.5);
        let mut y = x.mean(None, true);
        assert_eq!(y.item::<f32>(), 2.5);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.mean(&[0][..], None);
        assert_eq!(result.as_slice::<f32>(), &[2.0, 3.0]);

        let mut result = x.mean(&[1][..], None);
        assert_eq!(result.as_slice::<f32>(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.mean(&[][..], None);

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    #[test]
    fn test_mean_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.try_mean(&[2][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.min(None, None).item::<i32>(), 1);
        let mut y = x.min(None, true);
        assert_eq!(y.item::<i32>(), 1);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.min(&[0][..], None);
        assert_eq!(result.as_slice::<i32>(), &[1, 2]);

        let mut result = x.min(&[1][..], None);
        assert_eq!(result.as_slice::<i32>(), &[1, 3]);
    }

    #[test]
    fn test_min_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.min(&[][..], None);

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_var() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.variance(None, None, None).item::<f32>(), 1.25);
        let mut y = x.variance(None, true, None);
        assert_eq!(y.item::<f32>(), 1.25);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.variance(&[0][..], None, None);
        assert_eq!(result.as_slice::<f32>(), &[1.0, 1.0]);

        let mut result = x.variance(&[1][..], None, None);
        assert_eq!(result.as_slice::<f32>(), &[0.25, 0.25]);

        let x = Array::from_slice(&[1.0, 2.0], &[2]);
        let mut out = x.variance(None, None, Some(3));
        assert_eq!(out.item::<f32>(), f32::INFINITY);
    }

    #[test]
    fn test_var_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.variance(&[][..], None, 0);

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log_sum_exp() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.log_sum_exp(&[0][..], None);

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.3132615, 9.313262]);
    }

    #[test]
    fn test_log_sum_exp_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.log_sum_exp(&[][..], None);

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }
}
