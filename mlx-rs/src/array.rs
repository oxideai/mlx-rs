use std::pin::Pin;

use cxx::{CxxVector, Exception, UniquePtr};
use mlx_sys::{array::ffi::{self, array}, dtype::ffi::Dtype, types::{bfloat16::bfloat16_t, complex64::complex64_t, float16::float16_t}};

pub struct Array {
    inner: UniquePtr<ffi::array>,
}

impl Array {
    pub fn inner(&self) -> &ffi::array {
        &self.inner
    }

    pub fn into_inner(self) -> UniquePtr<ffi::array> {
        self.inner
    }

    pub fn as_pin_mut(&mut self) -> Pin<&mut ffi::array> {
        self.inner.pin_mut()
    }

    pub fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }

    // The number of elements in the array.
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    pub fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn shape(&self) -> &CxxVector<i32> {
        self.inner.shape()
    }

    pub fn shape_of_dim(&self, dim: i32) -> i32 {
        self.inner.shape_of_dim(dim)
    }

    pub fn strides(&self) -> &CxxVector<usize> {
        self.inner.strides()
    }

    pub fn dtype(&self) -> Dtype {
        self.inner.dtype()
    }

    pub fn eval(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().eval()
    }

    pub fn is_evaled(&self) -> bool {
        self.inner.is_evaled()
    }

    pub fn is_tracer(&self) -> bool {
        self.inner.is_tracer()
    }

    pub fn set_tracer(&mut self, is_tracer: bool) {
        self.inner.pin_mut().set_tracer(is_tracer)
    }

    pub fn overwrite_descriptor(&mut self, other: &Self) {
        self.inner.pin_mut().overwrite_descriptor(other.as_ref())
    }

    pub fn as_slice<T>(&self) -> &[T] 
    where
        Self: Data<T>,
    {
        unsafe {
            std::slice::from_raw_parts(self.data(), self.size())
        }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] 
    where
        Self: DataMut<T>,
    {
        unsafe {
            std::slice::from_raw_parts_mut(self.data_mut(), self.size())
        }
    }
}

pub trait FromScalar<T> {
    fn from_scalar(value: T) -> Self;
}

impl FromScalar<bool> for Array {
    fn from_scalar(value: bool) -> Self {
        Self {
            inner: ffi::array_new_bool(value),
        }
    }
}

impl FromScalar<i8> for Array {
    fn from_scalar(value: i8) -> Self {
        Self {
            inner: ffi::array_new_int8(value),
        }
    }
}

impl FromScalar<i16> for Array {
    fn from_scalar(value: i16) -> Self {
        Self {
            inner: ffi::array_new_int16(value),
        }
    }
}

impl FromScalar<i32> for Array {
    fn from_scalar(value: i32) -> Self {
        Self {
            inner: ffi::array_new_int32(value),
        }
    }
}

impl FromScalar<i64> for Array {
    fn from_scalar(value: i64) -> Self {
        Self {
            inner: ffi::array_new_int64(value),
        }
    }
}

impl FromScalar<u8> for Array {
    fn from_scalar(value: u8) -> Self {
        Self {
            inner: ffi::array_new_uint8(value),
        }
    }
}

impl FromScalar<u16> for Array {
    fn from_scalar(value: u16) -> Self {
        Self {
            inner: ffi::array_new_uint16(value),
        }
    }
}

impl FromScalar<u32> for Array {
    fn from_scalar(value: u32) -> Self {
        Self {
            inner: ffi::array_new_uint32(value),
        }
    }
}

impl FromScalar<u64> for Array {
    fn from_scalar(value: u64) -> Self {
        Self {
            inner: ffi::array_new_uint64(value),
        }
    }
}

impl FromScalar<f32> for Array {
    fn from_scalar(value: f32) -> Self {
        Self {
            inner: ffi::array_new_float32(value),
        }
    }
}

impl FromScalar<float16_t> for Array {
    fn from_scalar(value: float16_t) -> Self {
        Self {
            inner: ffi::array_new_float16(value),
        }
    }
}

impl FromScalar<bfloat16_t> for Array {
    fn from_scalar(value: bfloat16_t) -> Self {
        Self {
            inner: ffi::array_new_bfloat16(value),
        }
    }
}

impl FromScalar<complex64_t> for Array {
    fn from_scalar(value: complex64_t) -> Self {
        Self {
            inner: ffi::array_new_complex64(value),
        }
    }
}

impl<T> From<T> for Array
where
    Array: FromScalar<T>,
{
    fn from(value: T) -> Self {
        Self::from_scalar(value)
    }
}

pub trait FromSlice<T> {
    fn from_slice(slice: &[T], shape: &CxxVector<i32>) -> Self;
}

impl FromSlice<bool> for Array {
    fn from_slice(slice: &[bool], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_bool(slice, shape),
        }
    }
}

impl FromSlice<i8> for Array {
    fn from_slice(slice: &[i8], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_int8(slice, shape),
        }
    }
}

impl FromSlice<i16> for Array {
    fn from_slice(slice: &[i16], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_int16(slice, shape),
        }
    }
}

impl FromSlice<i32> for Array {
    fn from_slice(slice: &[i32], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_int32(slice, shape),
        }
    }
}

impl FromSlice<i64> for Array {
    fn from_slice(slice: &[i64], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_int64(slice, shape),
        }
    }
}

impl FromSlice<u8> for Array {
    fn from_slice(slice: &[u8], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_uint8(slice, shape),
        }
    }
}

impl FromSlice<u16> for Array {
    fn from_slice(slice: &[u16], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_uint16(slice, shape),
        }
    }
}

impl FromSlice<u32> for Array {
    fn from_slice(slice: &[u32], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_uint32(slice, shape),
        }
    }
}

impl FromSlice<u64> for Array {
    fn from_slice(slice: &[u64], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_uint64(slice, shape),
        }
    }
}

impl FromSlice<float16_t> for Array {
    fn from_slice(slice: &[float16_t], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_float16(slice, shape),
        }
    }
}

impl FromSlice<bfloat16_t> for Array {
    fn from_slice(slice: &[bfloat16_t], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_bfloat16(slice, shape),
        }
    }
}

impl FromSlice<f32> for Array {
    fn from_slice(slice: &[f32], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_float32(slice, shape),
        }
    }
}

impl FromSlice<complex64_t> for Array {
    fn from_slice(slice: &[complex64_t], shape: &CxxVector<i32>) -> Self {
        Self {
            inner: ffi::array_from_slice_complex64(slice, shape),
        }
    }
}

impl<T> From<(&[T], &CxxVector<i32>)> for Array
where
    Array: FromSlice<T>,
{
    fn from((slice, shape): (&[T], &CxxVector<i32>)) -> Self {
        Self::from_slice(slice, shape)
    }
}

impl AsRef<array> for Array {
    fn as_ref(&self) -> &array {
        &self.inner
    }
}

/* -------------------------------------------------------------------------- */
/*                                    Item                                    */
/* -------------------------------------------------------------------------- */

pub trait Item<T> {
    fn item(&self) -> Result<T, Exception>;
}

impl Item<bool> for Array {
    fn item(&self) -> Result<bool, Exception> {
        self.inner.item_bool()
    }
}

impl Item<i8> for Array {
    fn item(&self) -> Result<i8, Exception> {
        self.inner.item_int8()
    }
}

impl Item<i16> for Array {
    fn item(&self) -> Result<i16, Exception> {
        self.inner.item_int16()
    }
}

impl Item<i32> for Array {
    fn item(&self) -> Result<i32, Exception> {
        self.inner.item_int32()
    }
}

impl Item<i64> for Array {
    fn item(&self) -> Result<i64, Exception> {
        self.inner.item_int64()
    }
}

impl Item<u8> for Array {
    fn item(&self) -> Result<u8, Exception> {
        self.inner.item_uint8()
    }
}

impl Item<u16> for Array {
    fn item(&self) -> Result<u16, Exception> {
        self.inner.item_uint16()
    }
}

impl Item<u32> for Array {
    fn item(&self) -> Result<u32, Exception> {
        self.inner.item_uint32()
    }
}

impl Item<u64> for Array {
    fn item(&self) -> Result<u64, Exception> {
        self.inner.item_uint64()
    }
}

impl Item<f32> for Array {
    fn item(&self) -> Result<f32, Exception> {
        self.inner.item_float32()
    }
}

impl Item<float16_t> for Array {
    fn item(&self) -> Result<float16_t, Exception> {
        self.inner.item_float16()
    }
}

impl Item<bfloat16_t> for Array {
    fn item(&self) -> Result<bfloat16_t, Exception> {
        self.inner.item_bfloat16()
    }
}

impl Item<complex64_t> for Array {
    fn item(&self) -> Result<complex64_t, Exception> {
        self.inner.item_complex64()
    }
}

/* -------------------------------------------------------------------------- */
/*                                    Data                                    */
/* -------------------------------------------------------------------------- */

pub trait Data<T> {
    fn data(&self) -> *const T;
}

impl Data<bool> for Array {
    fn data(&self) -> *const bool {
        self.inner.data_bool()
    }
}

impl Data<i8> for Array {
    fn data(&self) -> *const i8 {
        self.inner.data_int8()
    }
}

impl Data<i16> for Array {
    fn data(&self) -> *const i16 {
        self.inner.data_int16()
    }
}

impl Data<i32> for Array {
    fn data(&self) -> *const i32 {
        self.inner.data_int32()
    }
}

impl Data<i64> for Array {
    fn data(&self) -> *const i64 {
        self.inner.data_int64()
    }
}

impl Data<u8> for Array {
    fn data(&self) -> *const u8 {
        self.inner.data_uint8()
    }
}

impl Data<u16> for Array {
    fn data(&self) -> *const u16 {
        self.inner.data_uint16()
    }
}

impl Data<u32> for Array {
    fn data(&self) -> *const u32 {
        self.inner.data_uint32()
    }
}

impl Data<u64> for Array {
    fn data(&self) -> *const u64 {
        self.inner.data_uint64()
    }
}

impl Data<f32> for Array {
    fn data(&self) -> *const f32 {
        self.inner.data_float32()
    }
}

impl Data<float16_t> for Array {
    fn data(&self) -> *const float16_t {
        self.inner.data_float16()
    }
}

impl Data<bfloat16_t> for Array {
    fn data(&self) -> *const bfloat16_t {
        self.inner.data_bfloat16()
    }
}

impl Data<complex64_t> for Array {
    fn data(&self) -> *const complex64_t {
        self.inner.data_complex64()
    }
}

/* -------------------------------------------------------------------------- */
/*                                   DataMut                                  */
/* -------------------------------------------------------------------------- */

pub trait DataMut<T>: Data<T> {
    fn data_mut(&mut self) -> *mut T;
}

impl DataMut<bool> for Array {
    fn data_mut(&mut self) -> *mut bool {
        self.inner.pin_mut().data_mut_bool()
    }
}

impl DataMut<i8> for Array {
    fn data_mut(&mut self) -> *mut i8 {
        self.inner.pin_mut().data_mut_int8()
    }
}

impl DataMut<i16> for Array {
    fn data_mut(&mut self) -> *mut i16 {
        self.inner.pin_mut().data_mut_int16()
    }
}

impl DataMut<i32> for Array {
    fn data_mut(&mut self) -> *mut i32 {
        self.inner.pin_mut().data_mut_int32()
    }
}

impl DataMut<i64> for Array {
    fn data_mut(&mut self) -> *mut i64 {
        self.inner.pin_mut().data_mut_int64()
    }
}

impl DataMut<u8> for Array {
    fn data_mut(&mut self) -> *mut u8 {
        self.inner.pin_mut().data_mut_uint8()
    }
}

impl DataMut<u16> for Array {
    fn data_mut(&mut self) -> *mut u16 {
        self.inner.pin_mut().data_mut_uint16()
    }
}

impl DataMut<u32> for Array {
    fn data_mut(&mut self) -> *mut u32 {
        self.inner.pin_mut().data_mut_uint32()
    }
}

impl DataMut<u64> for Array {
    fn data_mut(&mut self) -> *mut u64 {
        self.inner.pin_mut().data_mut_uint64()
    }
}

impl DataMut<f32> for Array {
    fn data_mut(&mut self) -> *mut f32 {
        self.inner.pin_mut().data_mut_float32()
    }
}

impl DataMut<float16_t> for Array {
    fn data_mut(&mut self) -> *mut float16_t {
        self.inner.pin_mut().data_mut_float16()
    }
}

impl DataMut<bfloat16_t> for Array {
    fn data_mut(&mut self) -> *mut bfloat16_t {
        self.inner.pin_mut().data_mut_bfloat16()
    }
}

impl DataMut<complex64_t> for Array {
    fn data_mut(&mut self) -> *mut complex64_t {
        self.inner.pin_mut().data_mut_complex64()
    }
}
