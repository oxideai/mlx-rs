use crate::array::Array;
use crate::error::Exception;
use crate::stream::StreamOrDevice;
use crate::utils::{axes_or_default_to_all};
use crate::Stream;
use mlx_macros::default_device;

impl Array {
    /// An `and` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.all(&[0][..], None).unwrap();
    ///
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn all_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_all_axes(
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

    /// A `product` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = array.prod(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn prod_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_prod(
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

    /// A `max` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = array.max(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn max_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_max(
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

    /// Sum reduce the array over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = array.sum(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    #[default_device]
    pub fn sum_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_sum(
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

    /// A `mean` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn mean_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_mean(
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

    /// A `min` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = array.min(&[0][..], None).unwrap();
    /// ```
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn min_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_min(
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

    /// Compute the variance(s) over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    #[default_device]
    pub fn variance_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_var(
                    self.c_array,
                    axes.as_ptr(),
                    axes.len(),
                    keep_dims.into().unwrap_or(false),
                    ddof.into().unwrap_or(0),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }

    /// A `log-sum-exp` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn log_sum_exp_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);

        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_logsumexp(
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

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_all() {
        let array = Array::from_slice(&[true, false, true, false], &[2, 2]);

        assert_eq!(array.all(None, None).unwrap().item::<bool>(), false);
        assert_eq!(array.all(None, true).unwrap().shape(), &[1, 1]);
        assert_eq!(array.all(&[0, 1][..], None).unwrap().item::<bool>(), false);

        let mut result = array.all(&[0][..], None).unwrap();
        assert_eq!(result.as_slice::<bool>(), &[true, false]);

        let mut result = array.all(&[1][..], None).unwrap();
        assert_eq!(result.as_slice::<bool>(), &[false, false]);
    }

    #[test]
    fn test_all_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let mut all = array.all(&[][..], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_prod() {
        let x = Array::from_slice(&[1, 2, 3, 3], &[2, 2]);
        assert_eq!(x.prod(None, None).unwrap().item::<i32>(), 18);

        let mut y = x.prod(None, true).unwrap();
        assert_eq!(y.item::<i32>(), 18);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.prod(&[0][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[3, 6]);

        let mut result = x.prod(&[1][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[2, 9])
    }

    #[test]
    fn test_prod_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.prod(&[][..], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_max() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.max(None, None).unwrap().item::<i32>(), 4);
        let mut y = x.max(None, true).unwrap();
        assert_eq!(y.item::<i32>(), 4);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.max(&[0][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[3, 4]);

        let mut result = x.max(&[1][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[2, 4]);
    }

    #[test]
    fn test_max_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.max(&[][..], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_sum() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.sum(&[0][..], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[9, 17]);
    }

    #[test]
    fn test_sum_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.sum(&[][..], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_mean() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.mean(None, None).unwrap().item::<f32>(), 2.5);
        let mut y = x.mean(None, true).unwrap();
        assert_eq!(y.item::<f32>(), 2.5);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.mean(&[0][..], None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[2.0, 3.0]);

        let mut result = x.mean(&[1][..], None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.mean(&[][..], None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    #[test]
    fn test_mean_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.mean(&[2][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.min(None, None).unwrap().item::<i32>(), 1);
        let mut y = x.min(None, true).unwrap();
        assert_eq!(y.item::<i32>(), 1);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.min(&[0][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 2]);

        let mut result = x.min(&[1][..], None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 3]);
    }

    #[test]
    fn test_min_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.min(&[][..], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_var() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.variance(None, None, None).unwrap().item::<f32>(), 1.25);
        let mut y = x.variance(None, true, None).unwrap();
        assert_eq!(y.item::<f32>(), 1.25);
        assert_eq!(y.shape(), &[1, 1]);

        let mut result = x.variance(&[0][..], None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[1.0, 1.0]);

        let mut result = x.variance(&[1][..], None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[0.25, 0.25]);

        let x = Array::from_slice(&[1.0, 2.0], &[2]);
        let mut out = x.variance(None, None, Some(3)).unwrap();
        assert_eq!(out.item::<f32>(), f32::INFINITY);
    }

    #[test]
    fn test_var_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.variance(&[][..], None, 0).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log_sum_exp() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.log_sum_exp(&[0][..], None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.3132615, 9.313262]);
    }

    #[test]
    fn test_log_sum_exp_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let mut result = array.log_sum_exp(&[][..], None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }
}
