mod ops;

use crate::{array::kind, array::kind::Kind};
use num_complex::Complex32;
use std::ffi::c_void;

// TODO: Clone should probably NOT be implemented because the underlying pointer is atomically
// reference counted but not guarded by a mutex.
pub struct Array {
    pub(super) c_array: mlx_sys::mlx_array,
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = crate::utils::mlx_describe(self.c_array as *mut c_void)
            .unwrap_or_else(|| "Array".to_string());
        write!(f, "{:?}", description)
    }
}

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
    pub unsafe fn from_ptr(c_array: mlx_sys::mlx_array) -> Array {
        Self { c_array }
    }

    // TODO: should this be unsafe?
    pub fn as_ptr(&self) -> mlx_sys::mlx_array {
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
    pub fn from_complex(val: Complex32) -> Array {
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
    pub fn from_slice<T: kind::Element>(data: &[T], shape: &[i32]) -> Self {
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
                T::KIND.into(),
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
    pub fn dtype(&self) -> Kind {
        let dtype = unsafe { mlx_sys::mlx_array_get_dtype(self.c_array) };
        Kind::try_from(dtype).unwrap()
    }

    // TODO: document that mlx is lazy
    /// Evaluate the array.
    pub fn eval(&mut self) {
        // This clearly modifies the array, so it should be mutable
        unsafe { mlx_sys::mlx_array_eval(self.c_array) };
    }

    /// Access the value of a scalar array.
    pub fn item<T: kind::Element>(&self) -> T {
        // TODO: check and perform type conversion from the inner type to the desired output type
        T::array_item(self)
    }

    /// Returns a pointer to the array data
    ///
    /// Returns `None` if the array is not evaluated.
    pub fn as_slice<T: kind::Element>(&self) -> Option<&[T]> {
        // TODO: type conversion from the inner type to the desired output type

        let data = T::array_data(self);
        if data.is_null() {
            return None;
        }
        let size = self.size();
        unsafe { Some(std::slice::from_raw_parts(data, size)) }
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
        assert_eq!(array.dtype(), Kind::Bool);
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
        assert_eq!(array.dtype(), Kind::Int32);
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
        assert_eq!(array.dtype(), Kind::Float32);
    }

    #[test]
    fn new_scalar_array_from_complex() {
        let val = Complex32::new(1.0, 2.0);
        let array = Array::from_complex(val);
        assert_eq!(array.item::<Complex32>(), val);
        assert_eq!(array.item_size(), 8);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 8);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Kind::Complex64);
    }

    #[test]
    fn new_array_from_single_element_slice() {
        let data = [1i32];
        let array = Array::from_slice(&data, &[1]);
        assert_eq!(array.as_slice::<i32>(), Some(&data[..]));
        assert_eq!(array.item::<i32>(), 1);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 1);
        assert_eq!(array.shape(), &[1]);
        assert_eq!(array.dtype(), Kind::Int32);
    }

    #[test]
    fn new_array_from_multi_element_slice() {
        let data = [1i32, 2, 3, 4, 5];
        let array = Array::from_slice(&data, &[5]);
        assert_eq!(array.as_slice::<i32>(), Some(&data[..]));
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 5);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 20);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 5);
        assert_eq!(array.shape(), &[5]);
        assert_eq!(array.dtype(), Kind::Int32);
    }

    #[test]
    fn new_2d_array_from_slice() {
        let data = [1i32, 2, 3, 4, 5, 6];
        let array = Array::from_slice(&data, &[2, 3]);
        assert_eq!(array.as_slice::<i32>(), Some(&data[..]));
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
        assert_eq!(array.dtype(), Kind::Int32);
    }

    // // TODO: fatal runtime error: Rust cannot catch foreign exceptions
    // #[test]
    // #[should_panic]
    // fn get_item_from_multi_element_array_should_panic() {
    //     let data = [1, 2, 3, 4, 5];
    //     let array = Array::from_slice(&data, &[5], 1);
    //     array.item::<i32>();
    // }
}