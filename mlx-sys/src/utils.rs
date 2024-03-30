use std::os::raw::{c_char, c_void};

use cxx::{kind::Trivial, vector::VectorElement, ExternType};

#[derive(Clone, Copy)]
#[repr(C, u8)]
pub enum StreamOrDevice {
    Default,
    Stream(ffi::Stream),
    Device(ffi::Device),
}

impl Default for StreamOrDevice {
    fn default() -> Self {
        Self::Default
    }
}

unsafe impl cxx::ExternType for StreamOrDevice {
    type Id = cxx::type_id!("mlx_cxx::StreamOrDevice");
    type Kind = cxx::kind::Trivial;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringView<'a> {
    repr: core::mem::MaybeUninit<[*const c_void; 2]>,
    borrow: core::marker::PhantomData<&'a [c_char]>,
}

unsafe impl<'a> ExternType for StringView<'a> {
    type Id = cxx::type_id!("std::string_view");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/utils.h");
        include!("mlx-cxx/utils.hpp");
        include!("mlx/stream.h");
        include!("mlx/device.h");
        include!("mlx-cxx/mlx_cxx.hpp");

        #[namespace = "mlx::core"]
        type Stream = crate::stream::ffi::Stream;

        #[namespace = "mlx::core"]
        type Device = crate::device::ffi::Device;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::utils::StreamOrDevice;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "std"]
        #[cxx_name = "string_view"]
        type StringView<'a> = crate::utils::StringView<'a>;

        #[namespace = "mlx_cxx"]
        fn result_type(arrays: &[UniquePtr<array>]) -> Dtype;

        /// The type from promoting the arrays' types with one another.
        #[namespace = "mlx::core"]
        #[cxx_name = "result_type"]
        fn result_type2(a: &array, b: &array) -> Dtype;

        #[namespace = "mlx::core"]
        #[cxx_name = "result_type"]
        fn result_type3(a: &array, b: &array, c: &array) -> Dtype;

        #[namespace = "mlx_cxx"]
        fn broadcast_shapes(
            s1: &CxxVector<i32>,
            s2: &CxxVector<i32>,
        ) -> Result<UniquePtr<CxxVector<i32>>>;

        #[namespace = "mlx_cxx"]
        fn is_same_shape(arrays: &[UniquePtr<array>]) -> bool;

        #[namespace = "mlx::core"]
        fn is_big_endian() -> bool;

        #[namespace = "mlx::core"]
        fn normalize_axis(axis: i32, ndim: i32) -> Result<i32>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "push_opaque"]
        fn push_array(vec: Pin<&mut CxxVector<array>>, array: UniquePtr<array>);

        #[namespace = "mlx_cxx"]
        #[cxx_name = "std_vec_from_slice"]
        fn new_cxx_vec_array_from_slice(slice: &[UniquePtr<array>]) -> UniquePtr<CxxVector<array>>;

        #[namespace = "mlx_cxx"]
        fn string_view_from_str(s: &str) -> StringView;

        #[namespace = "mlx_cxx"]
        fn string_view_as_bytes(s: StringView) -> &[c_char];
    }
}

impl<'a> StringView<'a> {
    pub fn new(s: &'a str) -> Self {
        ffi::string_view_from_str(s)
    }

    pub fn as_bytes(self) -> &'a [c_char] {
        ffi::string_view_as_bytes(self)
    }
}

impl<'a> core::ops::Deref for StringView<'a> {
    type Target = [c_char];
    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

pub trait IntoCxxVector<T> 
where
    T: ExternType<Kind = Trivial> + VectorElement,
{
    fn into_cxx_vector(self) -> cxx::UniquePtr<cxx::CxxVector<T>>;
}

impl<T> IntoCxxVector<T> for cxx::UniquePtr<cxx::CxxVector<T>>
where
    T: ExternType<Kind = Trivial> + VectorElement,
{
    fn into_cxx_vector(self) -> cxx::UniquePtr<cxx::CxxVector<T>> {
        self
    }
}

impl<T> IntoCxxVector<T> for Vec<T> 
where
    T: ExternType<Kind = Trivial> + VectorElement,
{
    fn into_cxx_vector(self) -> cxx::UniquePtr<cxx::CxxVector<T>> {
        let mut v = cxx::CxxVector::new();
        for x in self {
            v.pin_mut().push(x);
        }
        v
    }
}

impl<T, const N: usize> IntoCxxVector<T> for [T; N] 
where
    T: ExternType<Kind = Trivial> + VectorElement,
{
    fn into_cxx_vector(self) -> cxx::UniquePtr<cxx::CxxVector<T>> {
        let mut v = cxx::CxxVector::new();
        for x in self {
            v.pin_mut().push(x);
        }
        v
    }
}

impl<'a, T> IntoCxxVector<T> for &'a [T] 
where
    T: ExternType<Kind = Trivial> + VectorElement + Clone,
{
    fn into_cxx_vector(self) -> cxx::UniquePtr<cxx::CxxVector<T>> {
        let mut v = cxx::CxxVector::new();
        for x in self {
            v.pin_mut().push(x.clone());
        }
        v
    }
}

pub trait CloneCxxVector<T>
where
    T: ExternType<Kind = Trivial> + VectorElement + Clone,
{
    fn clone(&self) -> cxx::UniquePtr<cxx::CxxVector<T>>;
}

impl<T> CloneCxxVector<T> for cxx::CxxVector<T> 
where
    T: ExternType<Kind = Trivial> + VectorElement + Clone,
{
    fn clone(&self) -> cxx::UniquePtr<cxx::CxxVector<T>> {
        let mut v = cxx::CxxVector::new();
        for x in self.iter() {
            v.pin_mut().push(x.clone());
        }
        v
    }
}