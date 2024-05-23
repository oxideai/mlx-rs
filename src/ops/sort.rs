//! Implements bindings for the sorting ops.

use mlx_macros::default_device;

use crate::{
    error::{
        ArrayTooLargeForGpuError, InvalidAxisError, InvalidKthError, PartitionAllError,
        PartitionError, SortAllError, SortError,
    },
    utils::resolve_index,
    Array, Stream, StreamOrDevice,
};

/// Returns a sorted copy of the array.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = unsafe { sort_unchecked(&a, axis) };
/// ```
#[default_device]
pub unsafe fn sort_device_unchecked(a: &Array, axis: i32, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_sort(a.as_ptr(), axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns a sorted copy of the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = try_sort(&a, axis);
/// ```
#[default_device]
pub fn try_sort_device(
    a: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, SortError> {
    let resolved_axis = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: a.ndim(),
    })?;

    if a.shape()[resolved_axis] as usize >= (1usize << 21)
        // TODO: mlx-c doesn't support getting the device type yet
        && stream.as_ref() == &Stream::gpu()
    {
        return Err(ArrayTooLargeForGpuError {
            size: a.shape()[resolved_axis] as usize,
        }
        .into());
    }

    Ok(unsafe { sort_device_unchecked(a, axis, stream) })
}

/// Returns a sorted copy of the array. Panics if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_sort_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = sort(&a, axis);
/// ```
#[default_device]
pub fn sort_device(a: &Array, axis: i32, stream: impl AsRef<Stream>) -> Array {
    try_sort_device(a, axis, stream).unwrap()
}

/// Returns a sorted copy of the flattened array.
///
/// # Params
///
/// - `array`: input array
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = unsafe { sort_all_unchecked(&a) };
/// ```
#[default_device]
pub unsafe fn sort_all_device_unchecked(a: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_sort_all(a.as_ptr(), stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns a sorted copy of the flattened array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = try_sort_all(&a);
/// ```
#[default_device]
pub fn try_sort_all_device(a: &Array, stream: impl AsRef<Stream>) -> Result<Array, SortAllError> {
    if a.size() as u32 >= (1u32 << 21) && stream.as_ref() == &Stream::gpu() {
        return Err(ArrayTooLargeForGpuError { size: a.size() }.into());
    }

    Ok(unsafe { sort_all_device_unchecked(a, stream) })
}

/// Returns a sorted copy of the flattened array. Panics if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_sort_all_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = sort_all(&a);
/// ```
#[default_device]
pub fn sort_all_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    try_sort_all_device(a, stream).unwrap()
}

/// Returns the indices that sort the array.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = unsafe { argsort_unchecked(&a, axis) };
/// ```
#[default_device]
pub unsafe fn argsort_device_unchecked(a: &Array, axis: i32, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argsort(a.as_ptr(), axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns the indices that sort the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = try_argsort(&a, axis);
/// ```
#[default_device]
pub fn try_argsort_device(
    a: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, SortError> {
    let resolved_axis = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: a.ndim(),
    })?;

    if a.shape()[resolved_axis] as usize >= (1usize << 21) && stream.as_ref() == &Stream::gpu() {
        return Err(ArrayTooLargeForGpuError {
            size: a.shape()[resolved_axis] as usize,
        }
        .into());
    }

    Ok(unsafe { argsort_device_unchecked(a, axis, stream) })
}

/// Returns the indices that sort the array. Panics if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_argsort_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = argsort(&a, axis);
/// ```
#[default_device]
pub fn argsort_device(a: &Array, axis: i32, stream: impl AsRef<Stream>) -> Array {
    try_argsort_device(a, axis, stream).unwrap()
}

/// Returns the indices that sort the flattened array.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = unsafe { argsort_all_unchecked(&a) };
/// ```
#[default_device]
pub unsafe fn argsort_all_device_unchecked(a: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argsort_all(a.as_ptr(), stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns the indices that sort the flattened array. Returns an error if the arguments are
/// invalid.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = try_argsort_all(&a);
/// ```
#[default_device]
pub fn try_argsort_all_device(
    a: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Array, SortAllError> {
    if a.size() as u32 >= (1u32 << 21) && stream.as_ref() == &Stream::gpu() {
        return Err(ArrayTooLargeForGpuError { size: a.size() }.into());
    }

    Ok(unsafe { argsort_all_device_unchecked(a, stream) })
}

/// Returns the indices that sort the flattened array. Panics if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_argsort_all_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = argsort_all(&a);
/// ```
#[default_device]
pub fn argsort_all_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    try_argsort_all_device(a, stream).unwrap()
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = unsafe { partition_unchecked(&a, kth, axis) };
/// ```
#[default_device]
pub unsafe fn partition_device_unchecked(
    a: &Array,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_partition(a.as_ptr(), kth, axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
/// Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = try_partition(&a, kth, axis);
/// ```
#[default_device]
pub fn try_partition_device(
    a: &Array,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, PartitionError> {
    let resolved_axis = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: a.ndim(),
    })?;
    resolve_kth(kth, Some(resolved_axis), a)?;

    Ok(unsafe { partition_device_unchecked(a, kth, axis, stream) })
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
/// Panics if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_partition_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = partition(&a, kth, axis);
/// ```
#[default_device]
pub fn partition_device(a: &Array, kth: i32, axis: i32, stream: impl AsRef<Stream>) -> Array {
    try_partition_device(a, kth, axis, stream).unwrap()
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params:
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = unsafe { partition_all_unchecked(&a, kth) };
/// ```
#[default_device]
pub unsafe fn partition_all_device_unchecked(
    a: &Array,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_partition_all(a.as_ptr(), kth, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first. Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = try_partition_all(&a, kth);
/// ```
#[default_device]
pub fn try_partition_all_device(
    a: &Array,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, PartitionAllError> {
    resolve_kth(kth, None, a)?;
    Ok(unsafe { partition_all_device_unchecked(a, kth, stream) })
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first. Panics if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_partition_all_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = partition_all(&a, kth);
/// ```
#[default_device]
pub fn partition_all_device(a: &Array, kth: i32, stream: impl AsRef<Stream>) -> Array {
    try_partition_all_device(a, kth, stream).unwrap()
}

/// Returns the indices that partition the array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = unsafe { argpartition_unchecked(&a, kth, axis) };
/// ```
#[default_device]
pub unsafe fn argpartition_device_unchecked(
    a: &Array,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argpartition(a.as_ptr(), kth, axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns the indices that partition the array. Returns an error if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = try_argpartition(&a, kth, axis);
/// ```
#[default_device]
pub fn try_argpartition_device(
    a: &Array,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, PartitionError> {
    let resolved_axis = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: a.ndim(),
    })?;
    resolve_kth(kth, Some(resolved_axis), a)?;

    Ok(unsafe { argpartition_device_unchecked(a, kth, axis, stream) })
}

/// Returns the indices that partition the array. Panics if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_argpartition_device`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = argpartition(&a, kth, axis);
/// ```
#[default_device]
pub fn argpartition_device(a: &Array, kth: i32, axis: i32, stream: impl AsRef<Stream>) -> Array {
    try_argpartition_device(a, kth, axis, stream).unwrap()
}

/// Returns the indices that partition the flattened array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Safety
///
/// This is unsafe because it doesn't check if the arguments are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = unsafe { argpartition_all_unchecked(&a, kth) };
/// ```
#[default_device]
pub unsafe fn argpartition_all_device_unchecked(
    a: &Array,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_argpartition_all(a.as_ptr(), kth, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Returns the indices that partition the flattened array. Returns an error if the arguments are
/// invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = try_argpartition_all(&a, kth);
/// ```
#[default_device]
pub fn try_argpartition_all_device(
    a: &Array,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, PartitionAllError> {
    resolve_kth(kth, None, a)?;
    Ok(unsafe { argpartition_all_device_unchecked(a, kth, stream) })
}

/// Returns the indices that partition the flattened array. Panics if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Panics
///
/// This panics if the arguments are invalid. See [`try_argpartition_all_device`] for more
/// information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = argpartition_all(&a, kth);
/// ```
#[default_device]
pub fn argpartition_all_device(a: &Array, kth: i32, stream: impl AsRef<Stream>) -> Array {
    try_argpartition_all_device(a, kth, stream).unwrap()
}

fn resolve_kth(kth: i32, resolved_axis: Option<usize>, a: &Array) -> Result<i32, InvalidKthError> {
    match resolved_axis {
        Some(axis) => {
            let resolved_kth = if kth < 0 { a.shape()[axis] + kth } else { kth };

            if resolved_kth < 0 || resolved_kth >= a.shape()[axis] {
                return Err(InvalidKthError {
                    kth,
                    axis: axis as i32,
                    shape: a.shape().to_vec(),
                });
            }

            Ok(resolved_kth)
        }
        None => {
            let resolved_kth = if kth < 0 { a.size() as i32 + kth } else { kth };

            if resolved_kth < 0 || resolved_kth >= a.size() as i32 {
                return Err(InvalidKthError {
                    kth,
                    axis: 0,
                    shape: a.shape().to_vec(),
                });
            }

            Ok(resolved_kth)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Array, StreamOrDevice};

    #[test]
    fn test_sort_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let axis = 1;
        let result = super::try_sort(&a, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_with_large_arrays_on_gpu() {
        let a = Array::ones::<i32>(&[1 << 21]);
        let s = StreamOrDevice::gpu();
        let result = super::try_sort_device(&a, 0, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_all_with_large_arrays_on_gpu() {
        let a = Array::ones::<i32>(&[1 << 21]);
        let s = StreamOrDevice::gpu();
        let result = super::try_sort_all_device(&a, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_axis() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 2;
        let axis = 1;
        let result = super::try_partition(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let axis = 0;
        let result = super::try_partition(&a, kth, axis);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_all_with_invalid_kth() {
        let a = Array::from_slice(&[1, 2, 3, 4, 5], &[5]);
        let kth = 5;
        let result = super::try_partition_all(&a, kth);
        assert!(result.is_err());
    }
}
