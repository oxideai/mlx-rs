use crate::{
    dtype::Dtype,
    error::AsSliceError,
    sealed::Sealed,
    utils::{guard::Guarded, SUCCESS},
    Stream, StreamOrDevice,
};
use mlx_internal_macros::default_device;
use mlx_sys::mlx_array;
use num_complex::Complex;
use std::{
    ffi::{c_void, CStr},
    iter::Sum,
};

mod element;
mod operators;

pub use element::ArrayElement;

// Not using Complex64 because `num_complex::Complex64` is actually Complex<f64>

/// Type alias for `num_complex::Complex<f32>`.
#[allow(non_camel_case_types)]
pub type complex64 = Complex<f32>;

/// An n-dimensional array.
#[repr(transparent)]
pub struct Array {
    pub(crate) c_array: mlx_array,
}

impl Sealed for Array {}

impl Sealed for &Array {}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let status = mlx_sys::mlx_array_tostring(&mut mlx_str as *mut _, self.c_array);
            if status != SUCCESS {
                return Err(std::fmt::Error);
            }
            let ptr = mlx_sys::mlx_string_data(mlx_str);
            let c_str = CStr::from_ptr(ptr);
            write!(f, "{:?}", c_str)?;
            mlx_sys::mlx_string_free(mlx_str);
            Ok(())
        }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // TODO: check memory leak with some tool?

        // Decrease the reference count
        unsafe { mlx_sys::mlx_array_free(self.c_array) };
    }
}

unsafe impl Send for Array {}

impl PartialEq for Array {
    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `true` iff the arrays have
    /// the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// If you're looking for element-wise equality, use the [Array::eq()] method.
    fn eq(&self, other: &Self) -> bool {
        self.array_eq(other, None).unwrap().item()
    }
}

impl Array {
    /// Create a new array from an existing mlx_array pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the reference count of the array is properly incremented with
    /// `mlx_sys::mlx_retain`.
    pub unsafe fn from_ptr(c_array: mlx_array) -> Array {
        Self { c_array }
    }

    /// Get the underlying mlx_array pointer.
    pub fn as_ptr(&self) -> mlx_array {
        self.c_array
    }

    /// New array from a bool scalar.
    pub fn from_bool(val: bool) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_new_bool(val) };
        Array { c_array }
    }

    /// New array from an int scalar.
    pub fn from_int(val: i32) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_new_int(val) };
        Array { c_array }
    }

    /// New array from a float scalar.
    pub fn from_float(val: f32) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_new_float(val) };
        Array { c_array }
    }

    /// New array from a complex scalar.
    pub fn from_complex(val: complex64) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_new_complex(val.re, val.im) };
        Array { c_array }
    }

    /// New array from existing buffer.
    ///
    /// # Parameters
    ///
    /// - `data`: A buffer which will be copied.
    /// - `shape`: Shape of the array.
    ///
    /// # Panic
    ///
    /// - Panics if the product of the shape is not equal to the length of the data.
    /// - Panics if the shape is too large.
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        let dim = if shape.len() > i32::MAX as usize {
            panic!("Shape is too large")
        } else {
            shape.len() as i32
        };

        // Validate data size and shape
        assert_eq!(data.len(), shape.iter().product::<i32>() as usize);

        let c_array = unsafe {
            mlx_sys::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                dim,
                T::DTYPE.into(),
            )
        };

        Array { c_array }
    }

    /// New array from an iterator.
    ///
    /// This is a convenience method that is equivalent to
    ///
    /// ```rust, ignore
    /// let data: Vec<T> = iter.collect();
    /// Array::from_slice(&data, shape)
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let data = vec![1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_iter(data.clone(), &[5]);
    /// assert_eq!(array.as_slice::<i32>(), &data[..]);
    /// ```
    pub fn from_iter<T: ArrayElement, I: IntoIterator<Item = T>>(iter: I, shape: &[i32]) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        Self::from_slice(&data, shape)
    }

    /// The size of the array’s datatype in bytes.
    pub fn item_size(&self) -> usize {
        unsafe { mlx_sys::mlx_array_itemsize(self.c_array) }
    }

    /// Number of elements in the array.
    pub fn size(&self) -> usize {
        unsafe { mlx_sys::mlx_array_size(self.c_array) }
    }

    /// The strides of the array.
    pub fn strides(&self) -> &[usize] {
        let ndim = self.ndim();
        if ndim == 0 {
            // The data pointer may be null which would panic even if len is 0
            return &[];
        }

        unsafe {
            let data = mlx_sys::mlx_array_strides(self.c_array);
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The number of bytes in the array.
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_sys::mlx_array_nbytes(self.c_array) }
    }

    /// The array’s dimension.
    pub fn ndim(&self) -> usize {
        unsafe { mlx_sys::mlx_array_ndim(self.c_array) }
    }

    /// The shape of the array.
    ///
    /// Returns: a pointer to the sizes of each dimension.
    pub fn shape(&self) -> &[i32] {
        let ndim = self.ndim();
        if ndim == 0 {
            // The data pointer may be null which would panic even if len is 0
            return &[];
        }

        unsafe {
            let data = mlx_sys::mlx_array_shape(self.c_array);
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The shape of the array in a particular dimension.
    ///
    /// # Panic
    ///
    /// - Panics if the array is scalar.
    /// - Panics if `dim` is negative and `dim + ndim` overflows
    /// - Panics if the dimension is out of bounds.
    pub fn dim(&self, dim: i32) -> i32 {
        let dim = if dim.is_negative() {
            (self.ndim() as i32).checked_add(dim).unwrap()
        } else {
            dim
        };

        // This will panic on a scalar array
        unsafe { mlx_sys::mlx_array_dim(self.c_array, dim) }
    }

    /// The array element type.
    pub fn dtype(&self) -> Dtype {
        let dtype = unsafe { mlx_sys::mlx_array_dtype(self.c_array) };
        Dtype::try_from(dtype).unwrap()
    }

    // TODO: document that mlx is lazy
    /// Evaluate the array.
    pub fn eval(&self) -> crate::error::Result<()> {
        <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_array_eval(self.c_array) })
    }

    /// Access the value of a scalar array.
    /// If `T` does not match the array's `dtype` this will convert the type first.
    ///
    /// _Note: This will evaluate the array._
    pub fn item<T: ArrayElement>(&self) -> T {
        self.try_item().unwrap()
    }

    /// Access the value of a scalar array returning an error if the array is not a scalar.
    /// If `T` does not match the array's `dtype` this will convert the type first.
    ///
    /// _Note: This will evaluate the array._
    pub fn try_item<T: ArrayElement>(&self) -> crate::error::Result<T> {
        self.eval()?;

        // Evaluate the array, so we have content to work with in the conversion
        self.eval()?;

        // Though `mlx_array_item_<dtype>` returns a status code, it doesn't
        // return any non-success status code even if the dtype doesn't match.
        if self.dtype() != T::DTYPE {
            let new_array = Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_astype(
                    res,
                    self.c_array,
                    T::DTYPE.into(),
                    Stream::default().as_ptr(),
                )
            })?;
            new_array.eval()?;
            return T::array_item(&new_array);
        }

        T::array_item(self)
    }

    /// Returns a slice of the array data without validating the dtype.
    ///
    /// # Safety
    ///
    /// This is unsafe because the underlying data ptr is not checked for null or if the desired
    /// dtype matches the actual dtype of the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// unsafe {
    ///    let slice = array.as_slice_unchecked::<i32>();
    ///    assert_eq!(slice, &[1, 2, 3, 4, 5]);
    /// }
    /// ```
    pub unsafe fn as_slice_unchecked<T: ArrayElement>(&self) -> &[T] {
        self.eval().unwrap();

        unsafe {
            let data = T::array_data(self);
            let size = self.size();
            std::slice::from_raw_parts(data, size)
        }
    }

    /// Returns a slice of the array data returning an error if the dtype does not match the actual dtype.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// let slice = array.try_as_slice::<i32>();
    /// assert_eq!(slice, Ok(&data[..]));
    /// ```
    pub fn try_as_slice<T: ArrayElement>(&self) -> Result<&[T], AsSliceError> {
        if self.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: self.dtype(),
            });
        }

        self.eval()?;

        unsafe {
            let size = self.size();
            let data = T::array_data(self);
            if data.is_null() || size == 0 {
                return Err(AsSliceError::Null);
            }

            Ok(std::slice::from_raw_parts(data, size))
        }
    }

    /// Returns a slice of the array data.
    /// This method requires a mutable reference (`&self`) because it evaluates the array.
    ///
    /// # Panics
    ///
    /// Panics if the array is not evaluated or if the desired dtype does not match the actual dtype
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// let slice = array.as_slice::<i32>();
    /// assert_eq!(slice, &data[..]);
    /// ```
    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.try_as_slice().unwrap()
    }

    /// Clone the array by copying the data.
    ///
    /// This is named `deep_clone` to avoid confusion with the `Clone` trait.
    pub fn deep_clone(&self) -> Self {
        unsafe {
            let dtype = self.dtype();
            let shape = self.shape();
            let data = match dtype {
                Dtype::Bool => mlx_sys::mlx_array_data_bool(self.c_array) as *const c_void,
                Dtype::Uint8 => mlx_sys::mlx_array_data_uint8(self.c_array) as *const c_void,
                Dtype::Uint16 => mlx_sys::mlx_array_data_uint16(self.c_array) as *const c_void,
                Dtype::Uint32 => mlx_sys::mlx_array_data_uint32(self.c_array) as *const c_void,
                Dtype::Uint64 => mlx_sys::mlx_array_data_uint64(self.c_array) as *const c_void,
                Dtype::Int8 => mlx_sys::mlx_array_data_int8(self.c_array) as *const c_void,
                Dtype::Int16 => mlx_sys::mlx_array_data_int16(self.c_array) as *const c_void,
                Dtype::Int32 => mlx_sys::mlx_array_data_int32(self.c_array) as *const c_void,
                Dtype::Int64 => mlx_sys::mlx_array_data_int64(self.c_array) as *const c_void,
                Dtype::Float16 => mlx_sys::mlx_array_data_float16(self.c_array) as *const c_void,
                Dtype::Float32 => mlx_sys::mlx_array_data_float32(self.c_array) as *const c_void,
                Dtype::Bfloat16 => mlx_sys::mlx_array_data_bfloat16(self.c_array) as *const c_void,
                Dtype::Complex64 => {
                    mlx_sys::mlx_array_data_complex64(self.c_array) as *const c_void
                }
            };

            let new_c_array =
                mlx_sys::mlx_array_new_data(data, shape.as_ptr(), shape.len() as i32, dtype.into());

            Array::from_ptr(new_c_array)
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        Array::try_from_op(|res| unsafe { mlx_sys::mlx_array_set(res, self.c_array) })
            // Exception may be thrown when calling `new` in cpp.
            .expect("Failed to clone array")
    }
}

impl Sum for Array {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Array::from_int(0), |acc, x| acc.add(&x).unwrap())
    }
}

/// Stop gradients from being computed.
///
/// The operation is the identity but it prevents gradients from flowing
/// through the array.
#[default_device]
pub fn stop_gradient_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> crate::error::Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_stop_gradient(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

impl From<bool> for Array {
    fn from(value: bool) -> Self {
        Array::from_bool(value)
    }
}

impl From<i32> for Array {
    fn from(value: i32) -> Self {
        Array::from_int(value)
    }
}

impl From<f32> for Array {
    fn from(value: f32) -> Self {
        Array::from_float(value)
    }
}

impl From<complex64> for Array {
    fn from(value: complex64) -> Self {
        Array::from_complex(value)
    }
}

impl<T> From<T> for Array
where
    Array: FromNested<T>,
{
    fn from(value: T) -> Self {
        Array::from_nested(value)
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

/// A helper trait to construct `Array` from scalar values.
///
/// This trait is intended to be used with the macro [`array!`] but can be used directly if needed.
pub trait FromScalar<T>
where
    T: ArrayElement,
{
    /// Create an array from a scalar value.
    fn from_scalar(val: T) -> Array;
}

impl FromScalar<bool> for Array {
    fn from_scalar(val: bool) -> Array {
        Array::from_bool(val)
    }
}

impl FromScalar<i32> for Array {
    fn from_scalar(val: i32) -> Array {
        Array::from_int(val)
    }
}

impl FromScalar<f32> for Array {
    fn from_scalar(val: f32) -> Array {
        Array::from_float(val)
    }
}

impl FromScalar<complex64> for Array {
    fn from_scalar(val: complex64) -> Array {
        Array::from_complex(val)
    }
}

/// A helper trait to construct `Array` from nested arrays or slices.
///
/// Given that this is not intended for use other than the macro [`array!`], this trait is added
/// instead of directly implementing `From` for `Array` to avoid conflicts with other `From`
/// implementations.
///
/// Beware that this is subject to change in the future should we find a better way to implement
/// the macro without creating conflicts.
pub trait FromNested<T> {
    /// Create an array from nested arrays or slices.
    fn from_nested(data: T) -> Array;
}

impl<T: ArrayElement> FromNested<&[T]> for Array {
    fn from_nested(data: &[T]) -> Self {
        Array::from_slice(data, &[data.len() as i32])
    }
}

impl<T: ArrayElement, const N: usize> FromNested<[T; N]> for Array {
    fn from_nested(data: [T; N]) -> Self {
        Array::from_slice(&data, &[N as i32])
    }
}

impl<T: ArrayElement, const N: usize> FromNested<&[T; N]> for Array {
    fn from_nested(data: &[T; N]) -> Self {
        Array::from_slice(data, &[N as i32])
    }
}

impl<T: ArrayElement + Copy> FromNested<&[&[T]]> for Array {
    fn from_nested(data: &[&[T]]) -> Self {
        // check that all rows have the same length
        let row_len = data[0].len();
        assert!(
            data.iter().all(|row| row.len() == row_len),
            "Rows must have the same length"
        );

        let shape = [data.len() as i32, row_len as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<[&[T]; N]> for Array {
    fn from_nested(data: [&[T]; N]) -> Self {
        // check that all rows have the same length
        let row_len = data[0].len();
        assert!(
            data.iter().all(|row| row.len() == row_len),
            "Rows must have the same length"
        );

        let shape = [N as i32, row_len as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<&[[T; N]]> for Array {
    fn from_nested(data: &[[T; N]]) -> Self {
        let shape = [data.len() as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<&[&[T; N]]> for Array {
    fn from_nested(data: &[&[T; N]]) -> Self {
        let shape = [data.len() as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<[[T; N]; M]> for Array {
    fn from_nested(data: [[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<&[[T; N]; M]> for Array {
    fn from_nested(data: &[[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<&[&[T; N]; M]> for Array {
    fn from_nested(data: &[&[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy> FromNested<&[&[&[T]]]> for Array {
    fn from_nested(data: &[&[&[T]]]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [data.len() as i32, len_2d as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<[&[&[T]]; N]> for Array {
    fn from_nested(data: [&[&[T]]; N]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [N as i32, len_2d as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<&[[&[T]; N]]> for Array {
    fn from_nested(data: &[[&[T]; N]]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [data.len() as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize> FromNested<&[&[[T; N]]]> for Array {
    fn from_nested(data: &[&[[T; N]]]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        let shape = [data.len() as i32, len_2d as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<[[&[T]; N]; M]> for Array {
    fn from_nested(data: [[&[T]; N]; M]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [M as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<&[[&[T]; N]; M]> for Array {
    fn from_nested(data: &[[&[T]; N]; M]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [M as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize> FromNested<&[&[[T; N]]; M]> for Array {
    fn from_nested(data: &[&[[T; N]]; M]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        let shape = [M as i32, len_2d as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<[[[T; N]; M]; O]> for Array
{
    fn from_nested(data: [[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[[[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[&[[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[&[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[[&[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[[&[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: ArrayElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[&[&[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[&[&[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scalar_array_from_bool() {
        let array = Array::from_bool(true);
        assert!(array.item::<bool>());
        assert_eq!(array.item_size(), 1);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 1);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Bool);
    }

    #[test]
    fn new_scalar_array_from_int() {
        let array = Array::from_int(42);
        assert_eq!(array.item::<i32>(), 42);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Int32);
    }

    #[test]
    fn new_scalar_array_from_float() {
        let array = Array::from_float(3.14);
        assert_eq!(array.item::<f32>(), 3.14);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Float32);
    }

    #[test]
    fn new_scalar_array_from_complex() {
        let val = complex64::new(1.0, 2.0);
        let array = Array::from_complex(val);
        assert_eq!(array.item::<complex64>(), val);
        assert_eq!(array.item_size(), 8);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 8);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Complex64);
    }

    #[test]
    fn new_array_from_single_element_slice() {
        let data = [1i32];
        let array = Array::from_slice(&data, &[1]);
        assert_eq!(array.as_slice::<i32>(), &data[..]);
        assert_eq!(array.item::<i32>(), 1);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 1);
        assert_eq!(array.shape(), &[1]);
        assert_eq!(array.dtype(), Dtype::Int32);
    }

    #[test]
    fn new_array_from_multi_element_slice() {
        let data = [1i32, 2, 3, 4, 5];
        let array = Array::from_slice(&data, &[5]);
        assert_eq!(array.as_slice::<i32>(), &data[..]);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 5);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 20);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 5);
        assert_eq!(array.shape(), &[5]);
        assert_eq!(array.dtype(), Dtype::Int32);
    }

    #[test]
    fn new_2d_array_from_slice() {
        let data = [1i32, 2, 3, 4, 5, 6];
        let array = Array::from_slice(&data, &[2, 3]);
        assert_eq!(array.as_slice::<i32>(), &data[..]);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 6);
        assert_eq!(array.strides(), &[3, 1]);
        assert_eq!(array.nbytes(), 24);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.dim(0), 2);
        assert_eq!(array.dim(1), 3);
        assert_eq!(array.dim(-1), 3); // negative index
        assert_eq!(array.dim(-2), 2); // negative index
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Int32);
    }

    #[test]
    fn deep_cloned_array_has_different_ptr() {
        let data = [1i32, 2, 3, 4, 5];
        let orig = Array::from_slice(&data, &[5]);
        let clone = orig.deep_clone();

        // Data should be the same
        assert_eq!(orig.as_slice::<i32>(), clone.as_slice::<i32>());

        // Addr of `mlx_array` should be different
        assert_ne!(orig.as_ptr().ctx, clone.as_ptr().ctx);

        // Addr of data should be different
        assert_ne!(
            orig.as_slice::<i32>().as_ptr(),
            clone.as_slice::<i32>().as_ptr()
        );
    }

    #[test]
    fn test_array_eq() {
        let data = [1i32, 2, 3, 4, 5];
        let array1 = Array::from_slice(&data, &[5]);
        let array2 = Array::from_slice(&data, &[5]);
        let array3 = Array::from_slice(&[1i32, 2, 3, 4, 6], &[5]);

        assert_eq!(&array1, &array2);
        assert_ne!(&array1, &array3);
    }

    #[test]
    fn test_array_item_non_scalar() {
        let data = [1i32, 2, 3, 4, 5];
        let array = Array::from_slice(&data, &[5]);
        assert!(array.try_item::<i32>().is_err());
    }

    #[test]
    fn test_item_type_conversion() {
        let array = Array::from_float(1.0);
        assert_eq!(array.item::<i32>(), 1);
        assert_eq!(array.item::<complex64>(), complex64::new(1.0, 0.0));
        assert_eq!(array.item::<u8>(), 1);

        assert_eq!(array.as_slice::<f32>(), &[1.0]);
    }
}
