use cxx::UniquePtr;
use mlx_sys::{array::ffi::array, dtype::ffi::Dtype};
pub use mlx_sys::utils::{
    ffi::{new_cxx_vec_array_from_slice, normalize_axis, push_array},
    StreamOrDevice, IntoCxxVector, CloneCxxVector,
};
pub use mlx_sys::Optional;

use crate::array::Array;

pub fn result_type(arrays: &[Array]) -> Dtype {
    let slice: &[UniquePtr<array>] = unsafe { 
        // SAFETY: `Array` is a transparent wrapper around `UniquePtr<array>`.
        // so the memory layout of `Array` and `UniquePtr<array>` is the same.
        std::mem::transmute(arrays) 
    };
    mlx_sys::utils::ffi::result_type(slice)
}

pub fn is_same_shape(arrays: &[Array]) -> bool {
    let slice: &[UniquePtr<array>] = unsafe { 
        // SAFETY: `Array` is a transparent wrapper around `UniquePtr<array>`.
        // so the memory layout of `Array` and `UniquePtr<array>` is the same.
        std::mem::transmute(arrays) 
    };
    mlx_sys::utils::ffi::is_same_shape(slice)
}