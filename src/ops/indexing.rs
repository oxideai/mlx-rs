//! Indexing Arrays
//!
//! Arrays can be indexed in the following ways:
//!
//! 1. indexing with a single integer`i32`
//! 2. indexing with a slice `&[i32]`
//! 3. indexing with an iterator `impl Iterator<Item=i32>`

use std::ops::{Bound, RangeBounds};

use mlx_macros::default_device;
use smallvec::{smallvec, SmallVec};

use crate::{
    error::{
        DuplicateAxisError, ExpandDimsError, InvalidAxisError, SliceError, TakeAlongAxisError,
        TakeError,
    },
    utils::{all_unique, resolve_index, resolve_index_unchecked},
    Array, StreamOrDevice,
};

/* -------------------------------------------------------------------------- */
/*                                Custom types                                */
/* -------------------------------------------------------------------------- */

#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

#[derive(Debug, Clone, Copy)]
pub struct Stride<I> {
    pub iter: I,
    pub stride: i32,
}

pub trait IntoStride: Sized {
    fn stride(self, stride: i32) -> Stride<Self>;
}

impl<I> IntoStride for I
where
    I: Iterator<Item = i32>,
{
    fn stride(self, stride: i32) -> Stride<Self> {
        Stride { iter: self, stride }
    }
}

#[derive(Debug)]
pub enum ArrayIndexOp {
    TakeIndex {
        index: i32,
        axis: i32,
    },
    TakeArrayOwned {
        indices: Array,
        axis: i32,
    },
    Slice {
        start: Bound<i32>,
        stop: Bound<i32>,
        stride: i32,
    },
    ExpandDims,
}

impl ArrayIndexOp {
    fn is_array_or_index(&self) -> bool {
        matches!(
            self,
            ArrayIndexOp::TakeIndex { .. }
                | ArrayIndexOp::TakeArrayOwned { .. }
        )
    }

    fn is_array(&self) -> bool {
        matches!(
            self,
            ArrayIndexOp::TakeArrayOwned { .. }
        )
    }
}

/* -------------------------------------------------------------------------- */
/*                                Custom traits                               */
/* -------------------------------------------------------------------------- */

/// A marker trait for range bounds that are `i32`.
pub trait IndexBounds: RangeBounds<i32> {}

impl IndexBounds for std::ops::Range<i32> {}

impl IndexBounds for std::ops::RangeFrom<i32> {}

impl IndexBounds for std::ops::RangeFull {}

impl IndexBounds for std::ops::RangeInclusive<i32> {}

impl IndexBounds for std::ops::RangeTo<i32> {}

impl IndexBounds for std::ops::RangeToInclusive<i32> {}

/// A marker trait for iterators in `std` that yield `i32`.
pub trait IndexIterator: Iterator<Item = i32> {}

/* -------------------------------------------------------------------------- */
/*                               Implementation                               */
/* -------------------------------------------------------------------------- */

// Implement public bindings
impl Array {
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
                    let reshaped = Array::from_ptr(mlx_sys::mlx_reshape(
                        self.c_array,
                        shape as *const _,
                        shape.len(),
                        stream.as_ptr(),
                    ));

                    mlx_sys::mlx_take(reshaped.c_array, indices.c_array, 0, stream.as_ptr())
                }
            };

            Array::from_ptr(c_array)
        }
    }

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

    #[default_device]
    pub unsafe fn expand_dims_device_unchecked(
        &self,
        axes: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array =
                mlx_sys::mlx_expand_dims(self.c_array, axes.as_ptr(), axes.len(), stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_expand_dims_device(
        &self,
        axes: &[i32],
        stream: StreamOrDevice,
    ) -> Result<Array, ExpandDimsError> {
        // Check for valid axes
        // TODO: what is a good default capacity for SmallVec?
        let out_ndim = self.size() + axes.len();
        let mut out_axes = SmallVec::<[i32; 4]>::with_capacity(out_ndim as usize);
        for axis in axes {
            let valid_axis = resolve_index(*axis, out_ndim).ok_or_else(|| InvalidAxisError {
                axis: *axis,
                ndim: out_ndim,
            })?;
            if valid_axis > i32::MAX as usize {
                // TODO: return a different error type?
                return Err(InvalidAxisError {
                    axis: *axis,
                    ndim: out_ndim,
                }
                .into());
            }
            out_axes.push(valid_axis as i32);
        }

        // Check for duplicate axes
        all_unique(&out_axes).map_err(|axis| DuplicateAxisError { axis })?;

        unsafe { Ok(self.expand_dims_device_unchecked(&out_axes, stream)) }
    }

    #[default_device]
    pub fn expand_dims_device(&self, axes: &[i32], stream: StreamOrDevice) -> Array {
        self.try_expand_dims_device(axes, stream).unwrap()
    }
}

// Implement private bindings
impl Array {
    // This is exposed in the c api but not found in the swift or python api
    //
    // Thie is not the same as rust slice. Slice in python is more like `StepBy` iterator in rust
    #[default_device]
    pub(crate) unsafe fn slice_device_unchecked(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_slice(
                self.c_array,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_ptr(),
            );

            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub(crate) fn try_slice_device(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Result<Array, SliceError> {
        if start.len() != self.ndim() || stop.len() != self.ndim() || strides.len() != self.ndim() {
            return Err(SliceError { ndim: self.ndim() });
        }

        unsafe { Ok(self.slice_device_unchecked(start, stop, strides, stream)) }
    }

    #[default_device]
    pub(crate) fn slice_device(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        self.try_slice_device(start, stop, strides, stream).unwrap()
    }
}

// Implement additional public APIs
//
// TODO: rewrite this in a more rusty way
#[inline]
fn gather_nd(
    src: &Array,
    operations: impl Iterator<Item = ArrayIndexOp>,
    gather_first: bool,
) -> (i32, Array) {
    let mut max_dims = 0;
    let mut slice_count = 0;
    let mut is_slice: Vec<bool> = Vec::new();
    let mut gather_indices: Vec<Array> = Vec::new();

    let shape = src.shape();

    // prepare the gather indices
    for (i, op) in operations.enumerate() {
        match op {
            ArrayIndexOp::TakeIndex { index, axis } => {
                let item = Array::from_int(resolve_index_unchecked(index, src.dim(axis) as usize) as i32);
                gather_indices.push(item);
                is_slice.push(false);
            },
            ArrayIndexOp::Slice { start, stop, stride } => {
                slice_count += 1;
                is_slice.push(true);

                let size = shape[i];

                let absolute_start = {
                    let start = match start {
                        Bound::Included(start) => start,
                        Bound::Excluded(start) => start + 1,
                        Bound::Unbounded => if stride.is_negative() {
                            size - 1
                        } else {
                            0
                        },
                    };
                    if start < 0 {
                        start + size
                    } else {
                        start
                    }
                };

                let absolute_end = {
                    let end = match stop {
                        Bound::Included(end) => end + 1,
                        Bound::Excluded(end) => end,
                        Bound::Unbounded => if stride.is_negative() {
                            -size-1
                        } else {
                            size
                        },
                    };
                    if end < 0 {
                        end + size
                    } else {
                        end
                    }
                };

                // TODO: check if this is correct when stride is negative
                let indices: Vec<i32> = (absolute_start..absolute_end).step_by(stride.abs() as usize).collect();
                let item = Array::from_slice(&indices, &[indices.len() as i32]);

                gather_indices.push(item);
            },
            ArrayIndexOp::TakeArrayOwned { indices, axis: _ } => {
                is_slice.push(false);
                max_dims = max_dims.max(indices.ndim());
                gather_indices.push(indices);
            },
            ArrayIndexOp::ExpandDims => unreachable!("ExpandDims(ie. NewAxis) is already filtered out"),
        }
    }

    // reshape them so that the int/array indices are first
    if gather_first {
        if slice_count > 0 {
            let mut slice_indx = 0;
            for (i, item) in gather_indices.iter().enumerate() {
                if is_slice[i] {
                    let mut new_shape = vec![1; max_dims + slice_count];
                    new_shape[max_dims + slice_indx] = item.dim(0);
                    // gather_indices[i] = item.reshape(&new_shape);
                    todo!("need to implement reshape")
                } else {

                }
            }
        }
    }

    todo!()
}

#[inline]
fn get_item_int(src: &Array, index: i32, axis: i32) -> Array {
    src.take(&index.into(), axis)
}

#[inline]
fn get_item_array(src: &Array, indices: Array, axis: i32) -> Array {
    src.take(&indices, axis)
}

#[inline]
fn get_item_slice(src: &Array, start: Bound<i32>, stop: Bound<i32>, stride: i32) -> Array {
    let start_i = match start {
        Bound::Included(start) => start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    };
    let starts: SmallVec<[i32; 4]> = smallvec![start_i; src.ndim()];

    let mut stops = SmallVec::<[i32; 4]>::with_capacity(src.ndim());
    for i in 0..src.ndim() {
        let stop_i = match stop {
            Bound::Included(stop) => stop + 1,
            Bound::Excluded(stop) => stop,
            Bound::Unbounded => src.shape()[i],
        };

        stops.push(stop_i);
    }

    let strides: SmallVec<[i32; 4]> = smallvec![stride; src.ndim()];

    src.slice(&starts, &stops, &strides)
}

// See `mlx_get_item` in python/src/indexing.cpp and `getItem` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item(src: &Array, index: impl ArrayIndex) -> Array {
    use ArrayIndexOp::*;

    match index.index_op() {
        TakeIndex { index, axis } => get_item_int(src, index, axis),
        TakeArrayOwned { indices, axis } => get_item_array(src, indices, axis),
        Slice {
            start,
            stop,
            stride,
        } => get_item_slice(src, start, stop, stride),
        ExpandDims => unsafe {
            // SAFETY: 0 is always a valid axis
            src.expand_dims_unchecked(&[0])
        },
    }
}

// See `mlx_get_item_nd` in python/src/indexing.cpp and `getItemNd` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item_nd(src: &Array, indices: &[ArrayIndexOp]) -> Array {
    // Gather handling

    // compute gatherFirst -- this will be true if there is:
    // - a leading array or index operation followed by
    // - a non index/array (e.g. a slice)
    // - an int/array operation
    //
    // - and there is at least one array operation (hanled below with haveArray)
    let mut gather_first = false;
    let mut have_array_or_index = false; // This is `haveArrayOrIndex` in the swift binding
    let mut have_non_array = false;
    for item in indices {
        if item.is_array_or_index() {
            if have_array_or_index && have_non_array {
                gather_first = true;
                break;
            }
            have_array_or_index = true;
        } else {
            have_non_array = have_non_array || have_array_or_index;
        }
    }

    let array_count = indices.iter().filter(|op| op.is_array()).count();
    let have_array = array_count > 0;

    let mut remaining_indices: Vec<ArrayIndexOp> = Vec::new();
    if have_array {
        // apply all the operations (except for .newAxis) up to and including the
        // final .array operation (array operations are implemented via gather)
        let last_array_or_index = indices
            .iter()
            .rposition(|op| op.is_array_or_index())
            .unwrap(); // safe because we know there is at least one array operation
        let gather_indices = indices[..=last_array_or_index]
            .iter()
            .filter(|op| !matches!(op, ArrayIndexOp::ExpandDims));
    }

    todo!()
}

pub trait ArrayIndex {
    /// `mlx` allows out of bounds indexing.
    fn index_op(self) -> ArrayIndexOp;
}

impl ArrayIndex for i32 {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeIndex {
            index: self,
            axis: 0,
        }
    }
}

impl ArrayIndex for NewAxis {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::ExpandDims
    }
}

impl ArrayIndex for Array {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeArrayOwned {
            indices: self,
            axis: 0,
        }
    }
}

impl<T> ArrayIndex for T
where
    T: IndexBounds,
{
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice {
            start: self.start_bound().cloned(),
            stop: self.end_bound().cloned(),
            stride: 1,
        }
    }
}

impl<T> ArrayIndex for Stride<T>
where
    T: IndexBounds,
{
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice {
            start: self.iter.start_bound().cloned(),
            stop: self.iter.end_bound().cloned(),
            stride: self.stride,
        }
    }
}

// impl ArrayIndex for i32 {
//     fn get(self, array: &Array) -> Array {
//         let indices = self.into();
//         array.take(&indices, 0)
//     }
// }

// impl ArrayIndex for NewAxis {
//     fn get(self, array: &Array) -> Array {
//         let axes = &[0];

//         // SAFETY: 0 is always a valid axis
//         unsafe { array.expand_dims_unchecked(axes) }
//     }
// }

// impl ArrayIndex for Array {
//     fn get(self, array: &Array) -> Array {
//         array.take(&self, 0)
//     }
// }

// impl<'a> ArrayIndex for &'a Array {
//     fn get(self, array: &Array) -> Array {
//         array.take(self, 0)
//     }
// }

// impl<T> ArrayIndex for T
// where
//     T: IndexBounds,
// {
//     fn get(self, array: &Array) -> Array {
//         let mut start = SmallVec::<[i32; 4]>::with_capacity(array.ndim());
//         let mut stop = SmallVec::<[i32; 4]>::with_capacity(array.ndim());
//         let strides: SmallVec<[i32; 4]> = smallvec![1; array.ndim()];

//         for i in 0..array.ndim() {
//             let start_i = match self.start_bound() {
//                 std::ops::Bound::Included(&start) => start,
//                 std::ops::Bound::Excluded(&start) => start + 1,
//                 std::ops::Bound::Unbounded => 0,
//             };

//             let stop_i = match self.end_bound() {
//                 std::ops::Bound::Included(&stop) => stop + 1,
//                 std::ops::Bound::Excluded(&stop) => stop,
//                 std::ops::Bound::Unbounded => array.shape()[i],
//             };

//             start.push(start_i);
//             stop.push(stop_i);
//         }

//         array.slice(&start, &stop, &strides)
//     }
// }

// impl<T> ArrayIndex for Stride<T>
// where
//     T: IndexBounds,
// {
//     fn get(self, array: &Array) -> Array {
//         let mut start = SmallVec::<[i32; 4]>::with_capacity(array.ndim());
//         let mut stop = SmallVec::<[i32; 4]>::with_capacity(array.ndim());
//         let strides: SmallVec<[i32; 4]> = smallvec![self.stride; array.ndim()];

//         for i in 0..array.ndim() {
//             let start_i = match self.iter.start_bound() {
//                 std::ops::Bound::Included(&start) => start,
//                 std::ops::Bound::Excluded(&start) => start + 1,
//                 std::ops::Bound::Unbounded => 0,
//             };

//             let stop_i = match self.iter.end_bound() {
//                 std::ops::Bound::Included(&stop) => stop + 1,
//                 std::ops::Bound::Excluded(&stop) => stop,
//                 std::ops::Bound::Unbounded => array.shape()[i],
//             };

//             start.push(start_i);
//             stop.push(stop_i);
//         }

//         array.slice(&start, &stop, &strides)
//     }
// }

/* -------------------------------------------------------------------------- */
/*                                 Unit tests                                 */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_item() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let mut b = get_item(&a, 1);
        b.eval();

        assert_eq!(b.item::<f32>(), 2.0);
    }
}
