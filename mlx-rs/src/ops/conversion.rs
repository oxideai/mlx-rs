use mlx_macros::default_device;

use crate::{Array, ArrayElement, Dtype, Stream, StreamOrDevice};

impl Array {
    /// Create a new array with the contents converted to the given [ArrayElement] type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, Dtype};
    ///
    /// let array = Array::from_slice(&[1i16,2,3], &[3]);
    /// let mut new_array = array.as_type::<f32>();
    ///
    /// assert_eq!(new_array.dtype(), Dtype::Float32);
    /// assert_eq!(new_array.shape(), &[3]);
    /// assert_eq!(new_array.item_size(), 4);
    /// assert_eq!(new_array.as_slice::<f32>(), &[1.0,2.0,3.0]);
    /// ```
    #[default_device]
    pub fn as_type_device<T: ArrayElement>(&self, stream: impl AsRef<Stream>) -> Array {
        self.as_dtype_device(T::DTYPE, stream)
    }

    #[default_device]
    pub fn as_dtype_device(&self, dtype: Dtype, stream: impl AsRef<Stream>) -> Array {
        unsafe {
            let new_array =
                mlx_sys::mlx_astype(self.c_array, dtype.into(), stream.as_ref().as_ptr());
            Array::from_ptr(new_array)
        }
    }

    /// View the array as a different type.
    ///
    /// The output array will change along the last axis if the input array's
    /// type and the output array's type do not have the same size.
    ///
    /// _Note: the view op does not imply that the input and output arrays share
    /// their underlying data. The view only guarantees that the binary
    /// representation of each element (or group of elements) is the same._
    ///
    #[default_device]
    pub fn view_device<T: ArrayElement>(&self, stream: impl AsRef<Stream>) -> Array {
        self.view_dtype_device(T::DTYPE, stream)
    }

    #[default_device]
    pub fn view_dtype_device(&self, dtype: Dtype, stream: impl AsRef<Stream>) -> Array {
        unsafe {
            let new_array =
                mlx_sys::mlx_view(self.c_array, dtype.into(), stream.as_ref().as_ptr());
            Array::from_ptr(new_array)
        }
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use crate::complex64;
    use pretty_assertions::assert_eq;
    use super::*;

    macro_rules! test_as_type {
        ($src_type:ty, $src_val:expr, $dst_type:ty, $dst_val:expr, $len:expr) => {
            paste::paste! {
                #[test]
                fn [<test_as_type_ $src_type _ $dst_type>]() {
                    let array = Array::from_slice(&[$src_val; $len], &[$len as i32]);
                    let mut new_array = array.as_type::<$dst_type>();

                    assert_eq!(new_array.dtype(), $dst_type::DTYPE);
                    assert_eq!(new_array.shape(), &[3]);
                    assert_eq!(new_array.item_size(), std::mem::size_of::<$dst_type>());
                    assert_eq!(new_array.as_slice::<$dst_type>(), &[$dst_val; $len]);
                }
            }
        };
    }

    test_as_type!(bool, true, i8, 1, 3);
    test_as_type!(bool, true, i16, 1, 3);
    test_as_type!(bool, true, i32, 1, 3);
    test_as_type!(bool, true, i64, 1, 3);
    test_as_type!(bool, true, u8, 1, 3);
    test_as_type!(bool, true, u16, 1, 3);
    test_as_type!(bool, true, u32, 1, 3);
    test_as_type!(bool, true, u64, 1, 3);
    test_as_type!(bool, true, f32, 1.0, 3);
    test_as_type!(bool, true, f16, f16::from_f32(1.0), 3);
    test_as_type!(bool, true, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(bool, true, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i8, 1, bool, true, 3);
    test_as_type!(i8, 1, i16, 1, 3);
    test_as_type!(i8, 1, i32, 1, 3);
    test_as_type!(i8, 1, i64, 1, 3);
    test_as_type!(i8, 1, u8, 1, 3);
    test_as_type!(i8, 1, u16, 1, 3);
    test_as_type!(i8, 1, u32, 1, 3);
    test_as_type!(i8, 1, u64, 1, 3);
    test_as_type!(i8, 1, f32, 1.0, 3);
    test_as_type!(i8, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i8, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i8, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i16, 1, bool, true, 3);
    test_as_type!(i16, 1, i8, 1, 3);
    test_as_type!(i16, 1, i32, 1, 3);
    test_as_type!(i16, 1, i64, 1, 3);
    test_as_type!(i16, 1, u8, 1, 3);
    test_as_type!(i16, 1, u16, 1, 3);
    test_as_type!(i16, 1, u32, 1, 3);
    test_as_type!(i16, 1, u64, 1, 3);
    test_as_type!(i16, 1, f32, 1.0, 3);
    test_as_type!(i16, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i16, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i16, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i32, 1, bool, true, 3);
    test_as_type!(i32, 1, i8, 1, 3);
    test_as_type!(i32, 1, i16, 1, 3);
    test_as_type!(i32, 1, i64, 1, 3);
    test_as_type!(i32, 1, u8, 1, 3);
    test_as_type!(i32, 1, u16, 1, 3);
    test_as_type!(i32, 1, u32, 1, 3);
    test_as_type!(i32, 1, u64, 1, 3);
    test_as_type!(i32, 1, f32, 1.0, 3);
    test_as_type!(i32, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i32, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i32, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i64, 1, bool, true, 3);
    test_as_type!(i64, 1, i8, 1, 3);
    test_as_type!(i64, 1, i16, 1, 3);
    test_as_type!(i64, 1, i32, 1, 3);
    test_as_type!(i64, 1, u8, 1, 3);
    test_as_type!(i64, 1, u16, 1, 3);
    test_as_type!(i64, 1, u32, 1, 3);
    test_as_type!(i64, 1, u64, 1, 3);
    test_as_type!(i64, 1, f32, 1.0, 3);
    test_as_type!(i64, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i64, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i64, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u8, 1, bool, true, 3);
    test_as_type!(u8, 1, i8, 1, 3);
    test_as_type!(u8, 1, i16, 1, 3);
    test_as_type!(u8, 1, i32, 1, 3);
    test_as_type!(u8, 1, i64, 1, 3);
    test_as_type!(u8, 1, u16, 1, 3);
    test_as_type!(u8, 1, u32, 1, 3);
    test_as_type!(u8, 1, u64, 1, 3);
    test_as_type!(u8, 1, f32, 1.0, 3);
    test_as_type!(u8, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u8, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u8, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u16, 1, bool, true, 3);
    test_as_type!(u16, 1, i8, 1, 3);
    test_as_type!(u16, 1, i16, 1, 3);
    test_as_type!(u16, 1, i32, 1, 3);
    test_as_type!(u16, 1, i64, 1, 3);
    test_as_type!(u16, 1, u8, 1, 3);
    test_as_type!(u16, 1, u32, 1, 3);
    test_as_type!(u16, 1, u64, 1, 3);
    test_as_type!(u16, 1, f32, 1.0, 3);
    test_as_type!(u16, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u16, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u16, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u32, 1, bool, true, 3);
    test_as_type!(u32, 1, i8, 1, 3);
    test_as_type!(u32, 1, i16, 1, 3);
    test_as_type!(u32, 1, i32, 1, 3);
    test_as_type!(u32, 1, i64, 1, 3);
    test_as_type!(u32, 1, u8, 1, 3);
    test_as_type!(u32, 1, u16, 1, 3);
    test_as_type!(u32, 1, u64, 1, 3);
    test_as_type!(u32, 1, f32, 1.0, 3);
    test_as_type!(u32, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u32, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u32, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u64, 1, bool, true, 3);
    test_as_type!(u64, 1, i8, 1, 3);
    test_as_type!(u64, 1, i16, 1, 3);
    test_as_type!(u64, 1, i32, 1, 3);
    test_as_type!(u64, 1, i64, 1, 3);
    test_as_type!(u64, 1, u8, 1, 3);
    test_as_type!(u64, 1, u16, 1, 3);
    test_as_type!(u64, 1, u32, 1, 3);
    test_as_type!(u64, 1, f32, 1.0, 3);
    test_as_type!(u64, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u64, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u64, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(f32, 1.0, bool, true, 3);
    test_as_type!(f32, 1.0, i8, 1, 3);
    test_as_type!(f32, 1.0, i16, 1, 3);
    test_as_type!(f32, 1.0, i32, 1, 3);
    test_as_type!(f32, 1.0, i64, 1, 3);
    test_as_type!(f32, 1.0, u8, 1, 3);
    test_as_type!(f32, 1.0, u16, 1, 3);
    test_as_type!(f32, 1.0, u32, 1, 3);
    test_as_type!(f32, 1.0, u64, 1, 3);
    test_as_type!(f32, 1.0, f16, f16::from_f32(1.0), 3);
    test_as_type!(f32, 1.0, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(f32, 1.0, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(f16, f16::from_f32(1.0), bool, true, 3);
    test_as_type!(f16, f16::from_f32(1.0), i8, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i16, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i32, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i64, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u8, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u16, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u32, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u64, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), f32, 1.0, 3);
    test_as_type!(f16, f16::from_f32(1.0), bf16, bf16::from_f32(1.0), 3);
    test_as_type!(
        f16,
        f16::from_f32(1.0),
        complex64,
        complex64::new(1.0, 0.0),
        3
    );

    test_as_type!(bf16, bf16::from_f32(1.0), bool, true, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i8, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i16, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i32, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i64, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u8, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u16, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u32, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u64, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), f32, 1.0, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), f16, f16::from_f32(1.0), 3);

    test_as_type!(complex64, complex64::new(1.0, 0.0), bool, true, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i8, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i16, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i32, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i64, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u8, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u16, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u32, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u64, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), f32, 1.0, 3);
    test_as_type!(
        complex64,
        complex64::new(1.0, 0.0),
        f16,
        f16::from_f32(1.0),
        3
    );
    test_as_type!(
        complex64,
        complex64::new(1.0, 0.0),
        bf16,
        bf16::from_f32(1.0),
        3
    );

    #[test]
    fn test_view() {
        let array = Array::from_slice(&[1i16, 2, 3], &[3]);
        let mut new_array = array.view::<i8>();

        assert_eq!(new_array.dtype(), Dtype::Int8);
        assert_eq!(new_array.shape(), &[6]);
        assert_eq!(new_array.item_size(), 1);
        assert_eq!(new_array.as_slice::<i8>(), &[1, 0, 2, 0, 3, 0]);
    }
}
