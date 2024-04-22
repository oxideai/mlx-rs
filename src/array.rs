use std::ffi::c_void;
use std::ops::Add;

use half::{bf16, f16};
use mlx_sys::mlx_array;
use num_complex::Complex;

use crate::{dtype::Dtype, error::AsSliceError, sealed::Sealed, StreamOrDevice};

// Not using Complex64 because `num_complex::Complex64` is actually Complex<f64>
#[allow(non_camel_case_types)]
pub type complex64 = Complex<f32>;

/// A marker trait for array elements.
pub trait ArrayElement: Sealed {
    const DTYPE: Dtype;

    fn scalar_array_item(array: &Array) -> Self;

    fn array_data(array: &Array) -> *const Self;
}

macro_rules! impl_array_element {
    ($type:ty, $dtype:expr, $mlx_item_fn:ident, $mlx_data_fn:ident) => {
        impl Sealed for $type {}
        impl ArrayElement for $type {
            const DTYPE: Dtype = $dtype;

            fn scalar_array_item(array: &Array) -> Self {
                unsafe { mlx_sys::$mlx_item_fn(array.c_array) }
            }

            fn array_data(array: &Array) -> *const Self {
                unsafe { mlx_sys::$mlx_data_fn(array.c_array) }
            }
        }
    };
}

impl_array_element!(bool, Dtype::Bool, mlx_array_item_bool, mlx_array_data_bool);
impl_array_element!(u8, Dtype::Uint8, mlx_array_item_uint8, mlx_array_data_uint8);
impl_array_element!(
    u16,
    Dtype::Uint16,
    mlx_array_item_uint16,
    mlx_array_data_uint16
);
impl_array_element!(
    u32,
    Dtype::Uint32,
    mlx_array_item_uint32,
    mlx_array_data_uint32
);
impl_array_element!(
    u64,
    Dtype::Uint64,
    mlx_array_item_uint64,
    mlx_array_data_uint64
);
impl_array_element!(i8, Dtype::Int8, mlx_array_item_int8, mlx_array_data_int8);
impl_array_element!(
    i16,
    Dtype::Int16,
    mlx_array_item_int16,
    mlx_array_data_int16
);
impl_array_element!(
    i32,
    Dtype::Int32,
    mlx_array_item_int32,
    mlx_array_data_int32
);
impl_array_element!(
    i64,
    Dtype::Int64,
    mlx_array_item_int64,
    mlx_array_data_int64
);
impl_array_element!(
    f32,
    Dtype::Float32,
    mlx_array_item_float32,
    mlx_array_data_float32
);

impl Sealed for f16 {}

impl ArrayElement for f16 {
    const DTYPE: Dtype = Dtype::Float16;

    fn scalar_array_item(array: &Array) -> Self {
        let val = unsafe { mlx_sys::mlx_array_item_float16(array.c_array) };
        f16::from_bits(val.0)
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_float16(array.c_array) as *const Self }
    }
}

impl Sealed for bf16 {}

impl ArrayElement for bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;

    fn scalar_array_item(array: &Array) -> Self {
        let val = unsafe { mlx_sys::mlx_array_item_bfloat16(array.c_array) };
        bf16::from_bits(val)
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_bfloat16(array.c_array) as *const Self }
    }
}

impl Sealed for complex64 {}

impl ArrayElement for complex64 {
    const DTYPE: Dtype = Dtype::Complex64;

    fn scalar_array_item(array: &Array) -> Self {
        let bindgen_complex64 = unsafe { mlx_sys::mlx_array_item_complex64(array.c_array) };

        Self {
            re: bindgen_complex64.re,
            im: bindgen_complex64.im,
        }
    }

    fn array_data(array: &Array) -> *const Self {
        // complex64 has the same memory layout as __BindgenComplex<f32>
        unsafe { mlx_sys::mlx_array_data_complex64(array.c_array) as *const Self }
    }
}

// TODO: `mlx` differs from `numpy` in the way it handles out of bounds indices. It's behavior is more
// like `jax`. See the related issue here https://github.com/ml-explore/mlx/issues/206.
//
// The issue says it would use the last element if the index is out of bounds. But testing with
// python seems more like undefined behavior. Here we will use the last element if the index is
// is out of bounds.
pub struct Array {
    pub(crate) c_array: mlx_array,
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = crate::utils::mlx_describe(self.c_array as *mut c_void)
            .unwrap_or_else(|| "Array".to_string());
        write!(f, "{:?}", description)
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = crate::utils::mlx_describe(self.c_array as *mut c_void)
            .unwrap_or_else(|| "Array".to_string());
        write!(f, "{:?}", description)
    }
}

impl<'a> Add for &'a Array {
    type Output = Array;
    fn add(self, rhs: Self) -> Self::Output {
        self.add_device(rhs, StreamOrDevice::default())
    }
}

// TODO: Clone should probably NOT be implemented because the underlying pointer is atomically
// reference counted but not guarded by a mutex.

impl Drop for Array {
    fn drop(&mut self) {
        // TODO: check memory leak with some tool?

        // Decrease the reference count
        unsafe { mlx_sys::mlx_free(self.c_array as *mut c_void) };
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

    // TODO: should this be unsafe?
    pub fn as_ptr(&self) -> mlx_array {
        self.c_array
    }

    /// New array from a bool scalar.
    pub fn from_bool(val: bool) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_from_bool(val) };
        Array { c_array }
    }

    /// New array from an int scalar.
    pub fn from_int(val: i32) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_from_int(val) };
        Array { c_array }
    }

    /// New array from a float scalar.
    pub fn from_float(val: f32) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_from_float(val) };
        Array { c_array }
    }

    /// New array from a complex scalar.
    pub fn from_complex(val: complex64) -> Array {
        let c_array = unsafe { mlx_sys::mlx_array_from_complex(val.re, val.im) };
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
            mlx_sys::mlx_array_from_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                dim,
                T::DTYPE.into(),
            )
        };

        Array { c_array }
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
        let dtype = unsafe { mlx_sys::mlx_array_get_dtype(self.c_array) };
        Dtype::try_from(dtype).unwrap()
    }

    // TODO: document that mlx is lazy
    /// Evaluate the array.
    pub fn eval(&mut self) {
        // This clearly modifies the array, so it should be mutable
        unsafe { mlx_sys::mlx_array_eval(self.c_array) };
    }

    /// Access the value of a scalar array.
    pub fn item<T: ArrayElement>(&self) -> T {
        // TODO: check and perform type conversion from the inner type to the desired output type
        T::scalar_array_item(self)
    }

    /// Returns a slice of the array data.
    ///
    /// # Safety
    ///
    /// This is unsafe because the underlying data ptr is not checked for null or if the desired
    /// dtype matches the actual dtype of the array.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let array = Array::from_slice(&data[..], &[5]);
    ///
    /// unsafe {
    ///    let slice = array.as_slice_unchecked::<i32>();
    ///    assert_eq!(slice, &[1, 2, 3, 4, 5]);
    /// }
    /// ```
    pub unsafe fn as_slice_unchecked<T: ArrayElement>(&self) -> &[T] {
        unsafe {
            let data = T::array_data(self);
            let size = self.size();
            std::slice::from_raw_parts(data, size)
        }
    }

    /// Returns a slice of the array data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let array = Array::from_slice(&data[..], &[5]);
    ///
    /// let slice = array.try_as_slice::<i32>();
    /// assert_eq!(slice, Ok(&data[..]));
    /// ```
    pub fn try_as_slice<T: ArrayElement>(&self) -> Result<&[T], AsSliceError> {
        if self.size() == 0 {
            return Err(AsSliceError::Null);
        }

        if self.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch);
        }

        Ok(unsafe { self.as_slice_unchecked() })
    }

    /// Returns a slice of the array data.
    ///
    /// # Panics
    ///
    /// Panics if the array is not evaluated or if the desired dtype does not match the actual dtype
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let array = Array::from_slice(&data[..], &[5]);
    ///
    /// let slice = array.as_slice::<i32>();
    /// assert_eq!(slice, &data[..]);
    /// ```
    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.try_as_slice().unwrap()
    }
}

impl From<bool> for Array {
    fn from(val: bool) -> Self {
        Array::from_bool(val)
    }
}

impl From<i32> for Array {
    fn from(val: i32) -> Self {
        Array::from_int(val)
    }
}

impl From<f32> for Array {
    fn from(val: f32) -> Self {
        Array::from_float(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scalar_array_from_bool() {
        let array = Array::from_bool(true);
        assert_eq!(array.item::<bool>(), true);
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
}
