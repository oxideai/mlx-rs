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
//! | [`i32`] | An integer index |
//! | [`Array`] | Use an array to index another array |
//! | `&Array` | Use a reference to an array to index another array |
//! | [`std::ops::Range<i32>`] | A range index |
//! | [`std::ops::RangeFrom<i32>`] | A range index |
//! | [`std::ops::RangeFull`] | A range index |
//! | [`std::ops::RangeInclusive<i32>`] | A range index |
//! | [`std::ops::RangeTo<i32>`] | A range index |
//! | [`std::ops::RangeToInclusive<i32>`] | A range index |
//! | [`StrideBy`] | A range index with stride |
//! | [`NewAxis`] | Add a new axis |
//! | [`Ellipsis`] | Consume all axes |
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
//! use mlx_rs::{Array, ops::indexing::*};
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
//! use mlx_rs::{Array, ops::indexing::*};
//!
//! let mut a = Array::from_slice(&[1, 2, 3], &[3]);
//! a.index_mut(2, Array::from_int(0));
//!
//! let expected = Array::from_slice(&[1, 2, 0], &[3]);
//! assert_eq!(a, expected);
//! ```
//!
//! ```rust
//! use mlx_rs::{Array, ops::indexing::*};
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

use std::{borrow::Cow, ops::Bound, rc::Rc};

use mlx_internal_macros::{default_device, generate_macro};

use crate::{error::Result, utils::guard::Guarded, Array, Stream, StreamOrDevice};

pub(crate) mod index_impl;
pub(crate) mod indexmut_impl;

/* -------------------------------------------------------------------------- */
/*                                Custom types                                */
/* -------------------------------------------------------------------------- */

/// New axis indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

/// Ellipsis indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsis;

/// Stride indexing operation.
///
/// See the module level documentation for more information.
#[derive(Debug, Clone, Copy)]
pub struct StrideBy<I> {
    /// The inner iterator
    pub inner: I,

    /// The stride
    pub stride: i32,
}

/// Helper trait for creating a stride indexing operation.
pub trait IntoStrideBy: Sized {
    /// Create a stride indexing operation.
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

/// Range indexing operation.
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

/// Indexing operation for arrays.
#[derive(Debug, Clone)]
pub enum ArrayIndexOp<'a> {
    /// An `Ellipsis` is used to consume all axes
    ///
    /// This is equivalent to `...` in python
    Ellipsis,

    /// A single index operation
    ///
    /// This is equivalent to `arr[1]` in python
    TakeIndex {
        /// The index to take
        index: i32,
    },

    /// Indexing with an array
    TakeArray {
        /// The indices to take
        indices: Rc<Array>, // TODO: remove `Rc` because `Array` is `Clone`
    },

    /// Indexing with an array reference
    TakeArrayRef {
        /// The indices to take
        indices: &'a Array,
    },

    /// Indexing with a range
    ///
    /// This is equivalent to `arr[1:3]` in python
    Slice(RangeIndex),

    /// New axis operation
    ///
    /// This is equivalent to `arr[None]` in python
    ExpandDims,
}

impl ArrayIndexOp<'_> {
    fn is_array_or_index(&self) -> bool {
        // Using the full match syntax to avoid forgetting to add new variants
        match self {
            ArrayIndexOp::TakeIndex { .. }
            | ArrayIndexOp::TakeArrayRef { .. }
            | ArrayIndexOp::TakeArray { .. } => true,
            ArrayIndexOp::Ellipsis | ArrayIndexOp::Slice(_) | ArrayIndexOp::ExpandDims => false,
        }
    }

    fn is_array(&self) -> bool {
        // Using the full match syntax to avoid forgetting to add new variants
        match self {
            ArrayIndexOp::TakeArray { .. } | ArrayIndexOp::TakeArrayRef { .. } => true,
            ArrayIndexOp::TakeIndex { .. }
            | ArrayIndexOp::Ellipsis
            | ArrayIndexOp::Slice(_)
            | ArrayIndexOp::ExpandDims => false,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                Custom traits                               */
/* -------------------------------------------------------------------------- */

/// Trait for custom indexing operations.
///
/// Out of bounds indexing is allowed and wouldn't return an error.
pub trait TryIndexOp<Idx> {
    /// Try to index the array with the given index.
    fn try_index_device(&self, i: Idx, stream: impl AsRef<Stream>) -> Result<Array>;

    /// Try to index the array with the given index.
    fn try_index(&self, i: Idx) -> Result<Array> {
        self.try_index_device(i, StreamOrDevice::default())
    }
}

/// Trait for custom indexing operations.
///
/// This is implemented for all types that implement `TryIndexOp`.
pub trait IndexOp<Idx>: TryIndexOp<Idx> {
    /// Index the array with the given index.
    fn index_device(&self, i: Idx, stream: impl AsRef<Stream>) -> Array {
        self.try_index_device(i, stream).unwrap()
    }

    /// Index the array with the given index.
    fn index(&self, i: Idx) -> Array {
        self.try_index(i).unwrap()
    }
}

impl<T, Idx> IndexOp<Idx> for T where T: TryIndexOp<Idx> {}

/// Trait for custom mutable indexing operations.
pub trait TryIndexMutOp<Idx, Val> {
    /// Try to index the array with the given index and set the value.
    fn try_index_mut_device(&mut self, i: Idx, val: Val, stream: impl AsRef<Stream>) -> Result<()>;

    /// Try to index the array with the given index and set the value.
    fn try_index_mut(&mut self, i: Idx, val: Val) -> Result<()> {
        self.try_index_mut_device(i, val, StreamOrDevice::default())
    }
}

// TODO: should `Val` impl `AsRef<Array>` or `Into<Array>`?

/// Trait for custom mutable indexing operations.
pub trait IndexMutOp<Idx, Val>: TryIndexMutOp<Idx, Val> {
    /// Index the array with the given index and set the value.
    fn index_mut_device(&mut self, i: Idx, val: Val, stream: impl AsRef<Stream>) {
        self.try_index_mut_device(i, val, stream).unwrap()
    }

    /// Index the array with the given index and set the value.
    fn index_mut(&mut self, i: Idx, val: Val) {
        self.try_index_mut(i, val).unwrap()
    }
}

impl<T, Idx, Val> IndexMutOp<Idx, Val> for T where T: TryIndexMutOp<Idx, Val> {}

/// Trait for custom indexing operations.
pub trait ArrayIndex<'a> {
    /// `mlx` allows out of bounds indexing.
    fn index_op(self) -> ArrayIndexOp<'a>;
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
    pub fn take_axis_device(
        &self,
        indices: impl AsRef<Array>,
        axis: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take_axis(
                res,
                self.as_ptr(),
                indices.as_ref().as_ptr(),
                axis,
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Take elements from flattened 1-D array.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    #[default_device]
    pub fn take_device(
        &self,
        indices: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take(
                res,
                self.as_ptr(),
                indices.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Take values along an axis at the specified indices.
    ///
    /// If no axis is specified, the array is flattened to 1D prior to the indexing operation.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    #[default_device]
    pub fn take_along_axis_device(
        &self,
        indices: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let (input, axis) = match axis.into() {
            None => (Cow::Owned(self.reshape_device(&[-1], &stream)?), 0),
            Some(ax) => (Cow::Borrowed(self), ax),
        };

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_take_along_axis(
                res,
                input.as_ptr(),
                indices.as_ref().as_ptr(),
                axis,
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Put values along an axis at the specified indices.
    ///
    /// If no axis is specified, the array is flattened to 1D prior to the indexing operation.
    ///
    /// # Params
    /// - indices: Indices array. These should be broadcastable with the input array excluding the `axis` dimension.
    /// - values: Values array. These should be broadcastable with the indices.
    /// - axis: Axis in the destination to put the values to.
    /// - stream: stream or device to evaluate on.
    #[default_device]
    pub fn put_along_axis_device(
        &self,
        indices: impl AsRef<Array>,
        values: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        match axis.into() {
            None => {
                let input = self.reshape_device(&[-1], &stream)?;
                let array = Array::try_from_op(|res| unsafe {
                    mlx_sys::mlx_put_along_axis(
                        res,
                        input.as_ptr(),
                        indices.as_ref().as_ptr(),
                        values.as_ref().as_ptr(),
                        0,
                        stream.as_ref().as_ptr(),
                    )
                })?;
                let array = array.reshape_device(self.shape(), &stream)?;
                Ok(array)
            }
            Some(ax) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_put_along_axis(
                    res,
                    self.as_ptr(),
                    indices.as_ref().as_ptr(),
                    values.as_ref().as_ptr(),
                    ax,
                    stream.as_ref().as_ptr(),
                )
            }),
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
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn argmax_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmax_axis(
            res,
            a.as_ref().as_ptr(),
            axis,
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Indices of the maximum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn argmax_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmax(
            res,
            a.as_ref().as_ptr(),
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
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
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn argmin_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmin_axis(
            res,
            a.as_ref().as_ptr(),
            axis,
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Indices of the minimum value over the entire array.
///
/// # Params
///
/// - `a`: The input array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn argmin_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_argmin(
            res,
            a.as_ref().as_ptr(),
            keep_dims,
            stream.as_ref().as_ptr(),
        )
    })
}

/// See [`Array::take_along_axis`]
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn take_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().take_along_axis_device(indices, axis, stream)
}

/// See [`Array::put_along_axis`]
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn put_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    values: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref()
        .put_along_axis_device(indices, values, axis, stream)
}

/// See [`Array::take`]
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn take_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().take_axis_device(indices, axis, stream)
}

/// See [`Array::take_all`]
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn take_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().take_device(indices, stream)
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
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn topk_axis_device(
    a: impl AsRef<Array>,
    k: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_topk_axis(res, a.as_ref().as_ptr(), k, axis, stream.as_ref().as_ptr())
    })
}

/// Returns the `k` largest elements from the flattened input array.
#[generate_macro(customize(root = "$crate::ops::indexing"))]
#[default_device]
pub fn topk_device(
    a: impl AsRef<Array>,
    k: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_topk(res, a.as_ref().as_ptr(), k, stream.as_ref().as_ptr())
    })
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

fn expand_ellipsis_operations<'a>(
    ndim: usize,
    operations: &'a [ArrayIndexOp<'a>],
) -> Cow<'a, [ArrayIndexOp<'a>]> {
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
