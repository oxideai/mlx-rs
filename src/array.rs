use std::ffi::c_void;

use half::{bf16, f16};
use mlx_sys::mlx_array;
use num_complex::Complex;

use crate::sealed::Sealed;

// TODO: camel case?
// Not using Complex64 because `num_complex::Complex64` is actually Complex<f64>
#[allow(non_camel_case_types)]
pub type c64 = Complex<f32>;

/// Array element type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Dtype {
    Bool = mlx_sys::mlx_array_dtype__MLX_BOOL,
    Uint8 = mlx_sys::mlx_array_dtype__MLX_UINT8,
    Uint16 = mlx_sys::mlx_array_dtype__MLX_UINT16,
    Uint32 = mlx_sys::mlx_array_dtype__MLX_UINT32,
    Uint64 = mlx_sys::mlx_array_dtype__MLX_UINT64,
    Int8 = mlx_sys::mlx_array_dtype__MLX_INT8,
    Int16 = mlx_sys::mlx_array_dtype__MLX_INT16,
    Int32 = mlx_sys::mlx_array_dtype__MLX_INT32,
    Int64 = mlx_sys::mlx_array_dtype__MLX_INT64,
    Float16 = mlx_sys::mlx_array_dtype__MLX_FLOAT16,
    Float32 = mlx_sys::mlx_array_dtype__MLX_FLOAT32,
    Bfloat16 = mlx_sys::mlx_array_dtype__MLX_BFLOAT16,
    Complex64 = mlx_sys::mlx_array_dtype__MLX_COMPLEX64,
}

impl TryFrom<u32> for Dtype {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            mlx_sys::mlx_array_dtype__MLX_BOOL => Ok(Dtype::Bool),
            mlx_sys::mlx_array_dtype__MLX_UINT8 => Ok(Dtype::Uint8),
            mlx_sys::mlx_array_dtype__MLX_UINT16 => Ok(Dtype::Uint16),
            mlx_sys::mlx_array_dtype__MLX_UINT32 => Ok(Dtype::Uint32),
            mlx_sys::mlx_array_dtype__MLX_UINT64 => Ok(Dtype::Uint64),
            mlx_sys::mlx_array_dtype__MLX_INT8 => Ok(Dtype::Int8),
            mlx_sys::mlx_array_dtype__MLX_INT16 => Ok(Dtype::Int16),
            mlx_sys::mlx_array_dtype__MLX_INT32 => Ok(Dtype::Int32),
            mlx_sys::mlx_array_dtype__MLX_INT64 => Ok(Dtype::Int64),
            mlx_sys::mlx_array_dtype__MLX_FLOAT16 => Ok(Dtype::Float16),
            mlx_sys::mlx_array_dtype__MLX_FLOAT32 => Ok(Dtype::Float32),
            mlx_sys::mlx_array_dtype__MLX_BFLOAT16 => Ok(Dtype::Bfloat16),
            mlx_sys::mlx_array_dtype__MLX_COMPLEX64 => Ok(Dtype::Complex64),
            _ => Err(value),
        }
    }
}

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
                unsafe { mlx_sys::$mlx_item_fn(array.ptr) }
            }

            fn array_data(array: &Array) -> *const Self {
                unsafe { mlx_sys::$mlx_data_fn(array.ptr) }
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
        let val = unsafe { mlx_sys::mlx_array_item_float16(array.ptr) };
        f16::from_bits(val.0)
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_float16(array.ptr) as *const Self }
    }
}

impl Sealed for bf16 {}

impl ArrayElement for bf16 {
    const DTYPE: Dtype = Dtype::Bfloat16;

    fn scalar_array_item(array: &Array) -> Self {
        let val = unsafe { mlx_sys::mlx_array_item_bfloat16(array.ptr) };
        bf16::from_bits(val)
    }

    fn array_data(array: &Array) -> *const Self {
        unsafe { mlx_sys::mlx_array_data_bfloat16(array.ptr) as *const Self }
    }
}

impl Sealed for c64 {}

impl ArrayElement for c64 {
    const DTYPE: Dtype = Dtype::Complex64;

    fn scalar_array_item(array: &Array) -> Self {
        let bindgen_c64 = unsafe { mlx_sys::mlx_array_item_complex64(array.ptr) };

        Self {
            re: bindgen_c64.re,
            im: bindgen_c64.im,
        }
    }

    fn array_data(array: &Array) -> *const Self {
        // c64 has the same memory layout as __BindgenComplex<f32>
        unsafe { mlx_sys::mlx_array_data_complex64(array.ptr) as *const Self }
    }
}

pub struct Array {
    ptr: mlx_array,
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = crate::utils::mlx_describe(self.ptr as *mut c_void)
            .unwrap_or_else(|| "Array".to_string());
        write!(f, "{:?}", description)
    }
}

// TODO: Clone should probably NOT be implemented because the underlying pointer is atomically
// reference counted but not guarded by a mutex.

impl Drop for Array {
    fn drop(&mut self) {
        // TODO: check memory leak with some tool?

        // Decrease the reference count
        unsafe { mlx_sys::mlx_free(self.ptr as *mut c_void) };
    }
}

impl Array {
    pub unsafe fn from_ptr(ptr: mlx_array) -> Array {
        Array { ptr }
    }

    // TODO: should this be unsafe?
    pub fn as_ptr(&self) -> mlx_array {
        self.ptr
    }

    /// New array from a bool scalar.
    pub fn from_bool(val: bool) -> Array {
        let ptr = unsafe { mlx_sys::mlx_array_from_bool(val) };
        Array { ptr }
    }

    /// New array from a int scalar.
    pub fn from_int(val: i32) -> Array {
        let ptr = unsafe { mlx_sys::mlx_array_from_int(val) };
        Array { ptr }
    }

    /// New array from a float scalar.
    pub fn from_float(val: f32) -> Array {
        let ptr = unsafe { mlx_sys::mlx_array_from_float(val) };
        Array { ptr }
    }

    /// New array from a complex scalar.
    pub fn from_complex(val: c64) -> Array {
        let ptr = unsafe { mlx_sys::mlx_array_from_complex(val.re, val.im) };
        Array { ptr }
    }

    /// New array from existing buffer.
    ///
    /// # Parameters
    ///
    /// - `data`: A buffer which will be copied.
    /// - `shape`: Shape of the array.
    /// - `dim`: Number of dimensions (size of shape).
    pub fn from_data<T: ArrayElement>(data: &[T], shape: &[i32], dim: i32) -> Self {
        let ptr = unsafe {
            mlx_sys::mlx_array_from_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                dim,
                T::DTYPE as u32,
            )
        };

        Array { ptr }
    }

    /// The size of the array’s datatype in bytes.
    pub fn itemsize(&self) -> usize {
        unsafe { mlx_sys::mlx_array_itemsize(self.ptr) }
    }

    /// Number of elements in the array.
    pub fn size(&self) -> usize {
        unsafe { mlx_sys::mlx_array_size(self.ptr) }
    }

    /// The strides of the array.
    pub fn strides(&self) -> &[usize] {
        let ndim = self.ndim();

        unsafe {
            let data = mlx_sys::mlx_array_strides(self.ptr);
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The number of bytes in the array.
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_sys::mlx_array_nbytes(self.ptr) }
    }

    /// The array’s dimension.
    pub fn ndim(&self) -> usize {
        unsafe { mlx_sys::mlx_array_ndim(self.ptr) }
    }

    /// The shape of the array.
    ///
    /// Returns: a pointer to the sizes of each dimension.
    pub fn shape(&self) -> &[i32] {
        let ndim = self.ndim();

        unsafe {
            let data = mlx_sys::mlx_array_shape(self.ptr);
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The shape of the array in a particular dimension.
    ///
    /// # Panic
    ///
    /// Panics if the array is scalar.
    pub fn dim(&self, dim: i32) -> i32 {
        // This will panic on a scalar array
        unsafe { mlx_sys::mlx_array_dim(self.ptr, dim) }
    }

    /// The array element type.
    pub fn dtype(&self) -> Dtype {
        let dtype = unsafe { mlx_sys::mlx_array_get_dtype(self.ptr) };
        Dtype::try_from(dtype).unwrap()
    }

    /// Evaluate the array.
    pub fn eval(&mut self) {
        // This clearly modifies the array, so it should be mutable
        unsafe { mlx_sys::mlx_array_eval(self.ptr) };
    }

    /// Access the value of a scalar array.
    pub fn item<T: ArrayElement>(&self) -> T {
        T::scalar_array_item(self)
    }

    /// Returns a pointer to the array data
    pub fn data<T: ArrayElement>(&self) -> &[T] {
        // TODO: check below after ops are implemented
        // Array must be evaluated, otherwise returns NULL.

        let data = T::array_data(self);
        let size = self.size();
        unsafe { std::slice::from_raw_parts(data, size) }
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
        assert_eq!(array.itemsize(), 1);
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
        assert_eq!(array.itemsize(), 4);
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
        assert_eq!(array.itemsize(), 4);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Float32);
    }

    #[test]
    fn new_scalar_array_from_complex() {
        let val = c64 { re: 1.0, im: 2.0 };
        let array = Array::from_complex(val);
        assert_eq!(array.item::<c64>(), val);
        assert_eq!(array.itemsize(), 8);
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
        let array = Array::from_data(&data, &[1], 1);
        assert_eq!(array.data::<i32>(), &data);
        assert_eq!(array.item::<i32>(), 1);
        assert_eq!(array.itemsize(), 4);
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
        let array = Array::from_data(&data, &[5], 1);
        assert_eq!(array.data::<i32>(), &data);
        assert_eq!(array.itemsize(), 4);
        assert_eq!(array.size(), 5);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 20);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 5);
        assert_eq!(array.shape(), &[5]);
        assert_eq!(array.dtype(), Dtype::Int32);
    }

    // // TODO: fatal runtime error: Rust cannot catch foreign exceptions
    // #[test]
    // #[should_panic]
    // fn get_item_from_multi_element_array_should_panic() {
    //     let data = [1, 2, 3, 4, 5];
    //     let array = Array::from_data(&data, &[5], 1);
    //     array.item::<i32>();
    // }
}
