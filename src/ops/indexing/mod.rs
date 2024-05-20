//! Indexing Arrays
//!
//! # Overview
//!
//! The following types can be used for indexing:
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
//! // TODO: assert_eq!(s1, expected);
//!
//! // a[..., 0]
//! let mut s2 = a.index((Ellipsis, 0));
//!
//! let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
//! // TODO: assert_eq!(s1, expected);
//! ```

use std::{
    borrow::Cow,
    ops::{Bound, RangeBounds},
    rc::Rc,
};

use mlx_macros::default_device;

use crate::{
    error::{InvalidAxisError, SliceError, TakeAlongAxisError, TakeError},
    Array, StreamOrDevice,
};

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
    ///
    /// The reason an `Rc` is used instead of `Cow` is that even with `Cow`, the compiler will infer
    /// an `'static` lifetime due to current limitations in the borrow checker.
    TakeArray { indices: Rc<Array> },

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

    // pub fn range_full() -> Self {
    //     ArrayIndexOp::Slice(RangeIndex::new(Bound::Unbounded, Bound::Unbounded, Some(1)))
    // }
}

/* -------------------------------------------------------------------------- */
/*                                Custom traits                               */
/* -------------------------------------------------------------------------- */

pub trait IndexOp<Idx> {
    fn index_device(&self, i: Idx, stream: StreamOrDevice) -> Array;

    fn index(&self, i: Idx) -> Array {
        self.index_device(i, StreamOrDevice::default())
    }
}

// TODO: should `Val` impl `AsRef<Array>` or `Into<Array>`?
pub trait IndexMutOp<Idx, Val> {
    fn index_mut_device(&mut self, i: Idx, val: Val, stream: StreamOrDevice);

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

/* -------------------------------------------------------------------------- */
/*                               Implementation                               */
/* -------------------------------------------------------------------------- */

pub trait ArrayIndex {
    /// `mlx` allows out of bounds indexing.
    fn index_op(self) -> ArrayIndexOp;
}

impl ArrayIndex for i32 {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeIndex { index: self }
    }
}

impl ArrayIndex for NewAxis {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::ExpandDims
    }
}

impl ArrayIndex for Ellipsis {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Ellipsis
    }
}

impl ArrayIndex for Array {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeArray {
            indices: Rc::new(self),
        }
    }
}

impl ArrayIndex for Rc<Array> {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeArray { indices: self }
    }
}

impl ArrayIndex for ArrayIndexOp {
    fn index_op(self) -> ArrayIndexOp {
        self
    }
}

impl<T> ArrayIndex for T
where
    T: IndexBounds,
{
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice(RangeIndex::new(
            self.start_bound().cloned(),
            self.end_bound().cloned(),
            Some(1),
        ))
    }
}

impl ArrayIndex for std::ops::RangeFull {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice(RangeIndex::new(Bound::Unbounded, Bound::Unbounded, Some(1)))
    }
}

impl<T> ArrayIndex for StrideBy<T>
where
    T: IndexBounds,
{
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice(RangeIndex::new(
            self.inner.start_bound().cloned(),
            self.inner.end_bound().cloned(),
            Some(self.stride),
        ))
    }
}

impl ArrayIndex for StrideBy<std::ops::RangeFull> {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice(RangeIndex::new(
            Bound::Unbounded,
            Bound::Unbounded,
            Some(self.stride),
        ))
    }
}

// Implement public bindings
impl Array {
    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis. If the axis is not specified
    /// the array is treated as a flattened 1-D array prior to performing the take.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: The axis along which to take the elements. If `None`, the array is treated as a
    /// flattened 1-D vector.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arguments are valid.
    #[default_device]
    pub unsafe fn take_device_unchecked(
        &self,
        indices: &Array,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = match axis.into() {
                Some(axis) => {
                    mlx_sys::mlx_take(self.c_array, indices.c_array, axis, stream.as_ptr())
                }
                None => {
                    let shape = &[-1];
                    // SAFETY: &[-1] is a valid shape
                    let reshaped = self.reshape_device_unchecked(shape, stream.clone());

                    mlx_sys::mlx_take(reshaped.c_array, indices.c_array, 0, stream.as_ptr())
                }
            };

            Array::from_ptr(c_array)
        }
    }

    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis. If the axis is not specified
    /// the array is treated as a flattened 1-D array prior to performing the take.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: The axis along which to take the elements. If `None`, the array is treated as a
    /// flattened 1-D vector.
    #[default_device]
    pub fn try_take_device(
        &self,
        indices: &Array,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Result<Array, TakeError> {
        let ndim = self.ndim().min(i32::MAX as usize) as i32;

        let axis = axis.into();
        if let Some(axis) = axis {
            // Check for valid axis
            if axis + ndim < 0 || axis >= ndim {
                return Err(InvalidAxisError {
                    axis,
                    ndim: self.ndim(),
                }
                .into());
            }
        }

        // Check for valid take
        if self.size() == 0 && indices.size() != 0 {
            return Err(TakeError::NonEmptyTakeFromEmptyArray);
        }

        unsafe { Ok(self.take_device_unchecked(indices, axis, stream)) }
    }

    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis. If the axis is not specified
    /// the array is treated as a flattened 1-D array prior to performing the take.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: The axis along which to take the elements. If `None`, the array is treated as a
    /// flattened 1-D vector.
    ///
    /// # Panics
    ///
    /// This function panics if the arguments are invalid.
    #[default_device]
    pub fn take_device(
        &self,
        indices: &Array,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_take_device(indices, axis, stream).unwrap()
    }

    // NOTE: take and take_long_axis are two separate functions in the c++ code. They don't call
    // each other.

    /// Take values along an axis at the specified indices.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the arguments are valid.
    #[default_device]
    pub unsafe fn take_along_axis_device_unchecked(
        &self,
        indices: &Array,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array =
                mlx_sys::mlx_take_along_axis(self.c_array, indices.c_array, axis, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    /// Take values along an axis at the specified indices.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    #[default_device]
    pub fn try_take_along_axis_device(
        &self,
        indices: &Array,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, TakeAlongAxisError> {
        let ndim = self.ndim().min(i32::MAX as usize) as i32;

        // Check for valid axis
        if axis + ndim < 0 || axis >= ndim {
            return Err(InvalidAxisError {
                axis,
                ndim: self.ndim(),
            }
            .into());
        }

        // Check for dimension mismatch
        if indices.ndim() != self.ndim() {
            return Err(TakeAlongAxisError::IndicesDimensionMismatch {
                array_ndim: self.ndim(),
                indices_ndim: indices.ndim(),
            });
        }

        unsafe { Ok(self.take_along_axis_device_unchecked(indices, axis, stream)) }
    }

    /// Take values along an axis at the specified indices.
    ///
    /// # Params
    ///
    /// - `indices`: The indices to take from the array.
    /// - `axis`: Axis in the input to take the values from.
    ///
    /// # Panics
    ///
    /// This function panics if the arguments are invalid.
    #[default_device]
    pub fn take_along_axis_device(
        &self,
        indices: &Array,
        axis: i32,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_take_along_axis_device(indices, axis, stream)
            .unwrap()
    }
}

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

impl Array {
    #[inline]
    pub(self) fn check_slice_index_dimensions(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<(), SliceError> {
        if start.len() != self.ndim() || stop.len() != self.ndim() || strides.len() != self.ndim() {
            return Err(SliceError { ndim: self.ndim() });
        }

        Ok(())
    }
}

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
