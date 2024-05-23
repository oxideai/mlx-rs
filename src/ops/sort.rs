use mlx_macros::default_device;

use crate::{Array, StreamOrDevice};

#[default_device]
pub unsafe fn sort_device_unchecked(a: &Array, axis: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_sort(a.as_ptr(), axis, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn sort_all_device_unchecked(a: &Array, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_sort_all(a.as_ptr(), stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn argsort_device_unchecked(a: &Array, axis: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argsort(a.as_ptr(), axis, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn argsort_all_device_unchecked(a: &Array, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argsort_all(a.as_ptr(), stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn partition_device_unchecked(a: &Array, kth: i32, axis: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_partition(a.as_ptr(), kth, axis, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn partition_all_device_unchecked(a: &Array, kth: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_partition_all(a.as_ptr(), kth, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn argpartition_device_unchecked(a: &Array, kth: i32, axis: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argpartition(a.as_ptr(), kth, axis, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn argpartition_all_device_unchecked(a: &Array, kth: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argpartition_all(a.as_ptr(), kth, stream.as_ptr());
        Array::from_ptr(c_array)
    }
}
