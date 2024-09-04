//! Indexing Arrays
//!
//! # Overview
//!
//! Due to limitations in the `std::ops::Index` and `std::ops::IndexMut` traits (only references can
//! be returned), the indexing is achieved with the [`IndexOp`] and [`IndexMutOp`] traits where
//! arrays can be indexed with [`IndexOp::index()`] and [`IndexMutOp::index_mut()`] respectively.
//!
//! The following types can be used as indices:
//!
//! | Type | Description |
//! |------|-------------|
//! | `i32` | An integer index |
//! | `Array` | Use an array to index another array |
//! | `Rc<Array>` | Use an array to index another array |
//! | `std::ops::Range<i32>` | A range index |
//! | `std::ops::RangeFrom<i32>` | A range index |
//! | `std::ops::RangeFull` | A range index |
//! | `std::ops::RangeInclusive<i32>` | A range index |
//! | `std::ops::RangeTo<i32>` | A range index |
//! | `std::ops::RangeToInclusive<i32>` | A range index |
//! | [`StrideBy`] | A range index with stride |
//! | `NewAxis` | Add a new axis |
//! | `Ellipsis` | Consume all axes |
//!
//! # Single axis indexing
//!
//! | Indexing Operation | `mlx` (python) | `mlx-swift` | `mlx-rs` |
//! |--------------------|--------|-------|------|
//! | integer | `arr[1]` | `arr[1]` | `arr.index(1)` |
//! | range expression | `arr[1:3]` | `arr[1..<3]` | `arr.index(1..3)` |
//! | full range | `arr[:]` | `arr[0...]` | `arr.index(..)` |
//! | range with stride | `arr[::2]` | `arr[.stride(by: 2)]` | `arr.index((..).stride_by(2))` |
//! | ellipsis (consuming all axes) | `arr[...]` | `arr[.ellipsis]` | `arr.index(Ellipsis)` |
//! | newaxis | `arr[None]` | `arr[.newAxis]` | `arr.index(NewAxis)` |
//! | mlx array `i` | `arr[i]` | `arr[i]` | `arr.index(i)` |
//!
//! # Multi-axes indexing
//!
//! Multi-axes indexing with combinations of the above operations is also supported by combining the
//! operations in a tuple with the restriction that `Ellipsis` can only be used once.
//!
//! ## Examples
//!
//! ```rust
//! // See the multi-dimensional example code for mlx python https://ml-explore.github.io/mlx/build/html/usage/indexing.html
//!
//! use mlx_rs::prelude::*;
//!
//! let a = Array::from_iter(0..8, &[2, 2, 2]);
//!
//! // a[:, :, 0]
//! let mut s1 = a.index((.., .., 0));
//!
//! let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
//! assert_eq!(s1, expected);
//!
//! // a[..., 0]
//! let mut s2 = a.index((Ellipsis, 0));
//!
//! let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
//! assert_eq!(s1, expected);
//! ```
//!
//! # Set values with indexing
//!
//! The same indexing operations (single or multiple) can be used to set values in an array using
//! the [`IndexMutOp`] trait.
//!
//! ## Example
//!
//! ```rust
//! use mlx_rs::prelude::*;
//!
//! let mut a = Array::from_slice(&[1, 2, 3], &[3]);
//! a.index_mut(2, Array::from_int(0));
//!
//! let expected = Array::from_slice(&[1, 2, 0], &[3]);
//! assert_eq!(a, expected);
//! ```
//!
//! ```rust
//! use mlx_rs::prelude::*;
//!
//! let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);
//!
//! // writing using slices -- this ends up covering two elements
//! a.index_mut((0..1, 1..2, 2..4), Array::from_int(88));
//!
//! let expected = Array::from_slice(
//!     &[
//!         0, 1, 2, 3, 4, 5, 6, 88, 88, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//!     ],
//!     &[2, 2, 5],
//! );
//! assert_eq!(a, expected);
//! ```

use std::{
    borrow::Cow,
    ops::{Bound, RangeBounds},
};

use mlx_macros::default_device;

use crate::{error::Exception, Array, Stream, StreamOrDevice};

mod index_impl;
mod indexmut_impl;

/* -------------------------------------------------------------------------- */
/*                                Custom types                                */
/* -------------------------------------------------------------------------- */

#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

#[derive(Debug, Clone, Copy)]
pub struct Ellipsis;

#[derive(Debug, Clone, Copy)]
pub struct StrideBy<I> {
    pub inner: I,
    pub stride: i32,
}

pub trait IntoStrideBy: Sized {
    fn stride_by(self, stride: i32) -> StrideBy<Self>;
}

impl<T> IntoStrideBy for T {
    fn stride_by(self, stride: i32) -> StrideBy<Self> {
        StrideBy {
            inner: self,
            stride,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeIndex {
    start: Bound<i32>,
    stop: Bound<i32>,
    stride: i32,
}

impl RangeIndex {
    pub(crate) fn new(start: Bound<i32>, stop: Bound<i32>, stride: Option<i32>) -> Self {
        let stride = stride.unwrap_or(1);
        Self {
            start,
            stop,
            stride,
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        matches!(self.start, Bound::Unbounded)
            && matches!(self.stop, Bound::Unbounded)
            && self.stride == 1
    }

    pub(crate) fn stride(&self) -> i32 {
        self.stride
    }

    pub(crate) fn start(&self, size: i32) -> i32 {
        match self.start {
            Bound::Included(start) => start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => {
                // ref swift binding
                // _start ?? (stride < 0 ? size - 1 : 0)

                if self.stride.is_negative() {
                    size - 1
                } else {
                    0
                }
            }
        }
    }

    pub(crate) fn absolute_start(&self, size: i32) -> i32 {
        // ref swift binding
        // return start < 0 ? start + size : start

        let start = self.start(size);
        if start.is_negative() {
            start + size
        } else {
            start
        }
    }

    pub(crate) fn end(&self, size: i32) -> i32 {
        match self.stop {
            Bound::Included(stop) => stop + 1,
            Bound::Excluded(stop) => stop,
            Bound::Unbounded => {
                // ref swift binding
                // _end ?? (stride < 0 ? -size - 1 : size)

                if self.stride.is_negative() {
                    -size - 1
                } else {
                    size
                }
            }
        }
    }

    pub(crate) fn absolute_end(&self, size: i32) -> i32 {
        // ref swift binding
        // return end < 0 ? end + size : end

        let end = self.end(size);
        if end.is_negative() {
            end + size
        } else {
            end
        }
    }
}

#[derive(Debug, Clone)]
pub enum ArrayIndexOp {
    /// An `Ellipsis` is used to consume all axes
    ///
    /// This is equivalent to `...` in python
    Ellipsis,

    /// A single index operation
    ///
    /// This is equivalent to `arr[1]` in python
    TakeIndex { index: i32 },

    /// Indexing with an array
    TakeArray { indices: Array },

    /// Indexing with a range
    ///
    /// This is equivalent to `arr[1:3]` in python
    Slice(RangeIndex),

    /// New axis operation
    ///
    /// This is equivalent to `arr[None]` in python
    ExpandDims,
}

impl ArrayIndexOp {
    fn is_array_or_index(&self) -> bool {
        matches!(
            self,
            ArrayIndexOp::TakeIndex { .. } | ArrayIndexOp::TakeArray { .. }
        )
    }

    fn is_array(&self) -> bool {
        matches!(self, ArrayIndexOp::TakeArray { .. })
    }
}

/* -------------------------------------------------------------------------- */
/*                                Custom traits                               */
/* -------------------------------------------------------------------------- */

pub trait IndexOp<Idx> {
    fn index_device(&self, i: Idx, stream: impl AsRef<Stream>) -> Array;

    fn index(&self, i: Idx) -> Array {
        self.index_device(i, StreamOrDevice::default())
    }
}

// TODO: should `Val` impl `AsRef<Array>` or `Into<Array>`?
pub trait IndexMutOp<Idx, Val> {
    fn index_mut_device(&mut self, i: Idx, val: Val, stream: impl AsRef<Stream>);

    fn index_mut(&mut self, i: Idx, val: Val) {
        self.index_mut_device(i, val, StreamOrDevice::default())
    }
}

/// A marker trait for range bounds that are `i32`.
pub trait IndexBounds: RangeBounds<i32> {}

impl IndexBounds for std::ops::Range<i32> {}

impl IndexBounds for std::ops::RangeFrom<i32> {}

// impl IndexBounds for std::ops::RangeFull {}

impl IndexBounds for std::ops::RangeInclusive<i32> {}

impl IndexBounds for std::ops::RangeTo<i32> {}

impl IndexBounds for std::ops::RangeToInclusive<i32> {}

/// Trait for custom indexing operations.
pub trait ArrayIndex {
    /// `mlx` allows out of bounds indexing.
    fn index_op(self) -> ArrayIndexOp;
}

/* -------------------------------------------------------------------------- */
/*                               Implementation                               */
/* -------------------------------------------------------------------------- */

// Implement public bindings
impl Array {
    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis. If the axis is not specified
    /// the array is treated as a flattened 1-D array prior to performing the take.
    ///
    /// See [`Array::take_all`] for the flattened array.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: The axis along which to take the elements.
    #[default_device]
    pub fn take_device(
        &self,
        indices: &Array,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_take(self.c_array, indices.c_array, axis, stream.as_ref().as_ptr())
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    /// Take elements from flattened 1-D array.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    #[default_device]
    pub fn take_all_device(
        &self,
        indices: &Array,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_take_all(self.c_array, indices.c_array, stream.as_ref().as_ptr())
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    // NOTE: take and take_long_axis are two separate functions in the c++ code. They don't call
    // each other.

    /// Take values along an axis at the specified indices.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    #[default_device]
    pub fn take_along_axis_device(
        &self,
        indices: &Array,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_take_along_axis(
                    self.c_array,
                    indices.c_array,
                    axis,
                    stream.as_ref().as_ptr(),
                )
            };

            Ok(Array::from_ptr(c_array))
        }
    }
}

/// Indices of the maximum values along the axis.
///
/// See [`argmax_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis`: Axis to reduce over
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[default_device]
pub fn argmax_device(
    a: &Array,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argmax(a.as_ptr(), axis, keep_dims, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Indices of the maximum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[default_device]
pub fn argmax_all_device(
    a: &Array,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argmax_all(a.as_ptr(), keep_dims, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Indices of the minimum values along the axis.
///
/// See [`argmin_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis`: Axis to reduce over.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[default_device]
pub fn argmin_device(
    a: &Array,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argmin(a.as_ptr(), axis, keep_dims, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Indices of the minimum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[default_device]
pub fn argmin_all_device(
    a: &Array,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argmin_all(a.as_ptr(), keep_dims, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Returns the indices that partition the array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// See [`argpartition_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `kth`: Element index at the `kth` position in the output will give the sorted position. All
///   indices before the `kth` position will be of elements less or equal to the element at the
///   `kth` index and all indices after will be of elements greater or equal to the element at the
///   `kth` index.
/// - `axis`: Axis to partition over
#[default_device]
pub fn argpartition_device(
    a: &Array,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argpartition(a.as_ptr(), kth, axis, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Returns the indices that partition the flattened array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The input array.
/// - `kth`: Element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
#[default_device]
pub fn argpartition_all_device(
    a: &Array,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argpartition_all(a.as_ptr(), kth, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Returns the indices that sort the array.
///
/// See [`argsort_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis`: Axis to sort over.
#[default_device]
pub fn argsort_device(
    a: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argsort(a.as_ptr(), axis, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Returns the indices that sort the flattened array.
#[default_device]
pub fn argsort_all_device(a: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_argsort_all(a.as_ptr(), stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// See [`Array::take_along_axis`]
#[default_device]
pub fn take_along_axis_device(
    a: &Array,
    indices: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.take_along_axis_device(indices, axis, stream)
}

/// See [`Array::take`]
#[default_device]
pub fn take_device(
    a: &Array,
    indices: &Array,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.take_device(indices, axis, stream)
}

/// See [`Array::take_all`]
#[default_device]
pub fn take_all_device(
    a: &Array,
    indices: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    a.take_all_device(indices, stream)
}

/// Returns the `k` largest elements from the input along a given axis.
///
/// The elements will not necessarily be in sorted order.
///
/// See [`topk_all`] for the flattened array.
///
/// # Params
///
/// - `a`: The input array.
/// - `k`: The number of elements to return.
/// - `axis`: Axis to sort over. Default to `-1` if not specified.
#[default_device]
pub fn topk_device(
    a: &Array,
    k: i32,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let axis = axis.into().unwrap_or(-1);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_topk(a.as_ptr(), k, axis, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Returns the `k` largest elements from the flattened input array.
#[default_device]
pub fn topk_all_device(a: &Array, k: i32, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_topk_all(a.as_ptr(), k, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

fn count_non_new_axis_operations(operations: &[ArrayIndexOp]) -> usize {
    operations
        .iter()
        .filter(|op| !matches!(op, ArrayIndexOp::ExpandDims))
        .count()
}

fn expand_ellipsis_operations(ndim: usize, operations: &[ArrayIndexOp]) -> Cow<'_, [ArrayIndexOp]> {
    let ellipsis_count = operations
        .iter()
        .filter(|op| matches!(op, ArrayIndexOp::Ellipsis))
        .count();
    if ellipsis_count == 0 {
        return Cow::Borrowed(operations);
    }

    if ellipsis_count > 1 {
        panic!("Indexing with multiple ellipsis is not supported");
    }

    let ellipsis_pos = operations
        .iter()
        .position(|op| matches!(op, ArrayIndexOp::Ellipsis))
        .unwrap();
    let prefix = &operations[..ellipsis_pos];
    let suffix = &operations[(ellipsis_pos + 1)..];
    let expand_range =
        count_non_new_axis_operations(prefix)..(ndim - count_non_new_axis_operations(suffix));
    let expand = expand_range.map(|_| (..).index_op());

    let mut expanded = Vec::with_capacity(ndim);
    expanded.extend_from_slice(prefix);
    expanded.extend(expand);
    expanded.extend_from_slice(suffix);

    Cow::Owned(expanded)
}
