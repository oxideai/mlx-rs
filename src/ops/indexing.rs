//! Indexing Arrays
//!
//! Arrays can be indexed in the following ways:
//!
//! 1. indexing with a single integer`i32`
//! 2. indexing with a slice `&[i32]`
//! 3. indexing with an iterator `impl Iterator<Item=i32>`

use std::{
    borrow::Cow,
    ops::{Bound, RangeBounds},
    rc::Rc,
};

use mlx_macros::default_device;
use smallvec::{smallvec, SmallVec};

use crate::{
    error::{
        DuplicateAxisError, ExpandDimsError, InvalidAxisError, SliceError, TakeAlongAxisError,
        TakeError,
    },
    ops::{expand_dims_device_unchecked, expand_dims_unchecked, reshape},
    utils::{all_unique, resolve_index, resolve_index_unchecked},
    Array, StreamOrDevice,
};

use super::reshape_unchecked;

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

#[derive(Debug, Clone)]
pub enum ArrayIndexOp {
    TakeIndex {
        index: i32,
    },
    TakeArray {
        indices: Rc<Array>,
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
            ArrayIndexOp::TakeIndex { .. } | ArrayIndexOp::TakeArray { .. }
        )
    }

    fn is_array(&self) -> bool {
        matches!(self, ArrayIndexOp::TakeArray { .. })
    }

    pub fn range_full() -> Self {
        ArrayIndexOp::Slice {
            start: Bound::Unbounded,
            stop: Bound::Unbounded,
            stride: 1,
        }
    }
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
                    // SAFETY: &[-1] is a valid shape
                    let reshaped = self.reshape_device_unchecked(shape, stream.clone());

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
}

impl<T> IndexOp<T> for Array 
where
    T: ArrayIndex,
{
    fn index_device(&self, i: T, stream: StreamOrDevice) -> Array {
        get_item(self, i, stream)
    }
}

impl<A> IndexOp<(A,)> for Array
where
    A: ArrayIndex,
{
    fn index_device(&self, i: (A,), stream: StreamOrDevice) -> Array {
        let i = [i.0.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<A, B> IndexOp<(A, B)> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
{
    fn index_device(&self, i: (A, B), stream: StreamOrDevice) -> Array {
        let i = [i.0.index_op(), i.1.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<A, B, C> IndexOp<(A, B, C)> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
{
    fn index_device(&self, i: (A, B, C), stream: StreamOrDevice) -> Array {
        let i = [i.0.index_op(), i.1.index_op(), i.2.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<A, B, C, D> IndexOp<(A, B, C, D)> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
{
    fn index_device(&self, i: (A, B, C, D), stream: StreamOrDevice) -> Array {
        let i = [i.0.index_op(), i.1.index_op(), i.2.index_op(), i.3.index_op()];
        get_item_nd(self, &i, stream)
    }
}

impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
{
    fn index_device(&self, i: (A, B, C, D, E), stream: StreamOrDevice) -> Array {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
        ];
        get_item_nd(self, &i, stream)
    }
}

impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
{
    fn index_device(&self, i: (A, B, C, D, E, F), stream: StreamOrDevice) -> Array {
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
        ];
        get_item_nd(self, &i, stream)
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

#[inline]
fn slice_start(start: Bound<i32>, stride: i32, size: i32) -> i32 {
    match start {
        Bound::Included(start) => start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => {
            if stride.is_negative() {
                size - 1
            } else {
                0
            }
        }
    }
}

#[inline]
fn absolute_start(start: Bound<i32>, stride: i32, size: i32) -> i32 {
    let start = slice_start(start, stride, size);
    if start < 0 {
        start + size
    } else {
        start
    }
}

#[inline]
fn slice_end(end: Bound<i32>, stride: i32, size: i32) -> i32 {
    match end {
        Bound::Included(end) => end + 1,
        Bound::Excluded(end) => end,
        Bound::Unbounded => {
            if stride.is_negative() {
                -size - 1
            } else {
                size
            }
        }
    }
}

#[inline]
fn absolute_end(end: Bound<i32>, stride: i32, size: i32) -> i32 {
    let end = slice_end(end, stride, size);
    if end < 0 {
        end + size
    } else {
        end
    }
}

// Implement additional public APIs
//
// TODO: rewrite this in a more rusty way
#[inline]
fn gather_nd<'a>(
    src: &'a Array,
    operations: impl Iterator<Item = &'a ArrayIndexOp>,
    gather_first: bool,
    stream: StreamOrDevice,
) -> (usize, Array) {
    use ArrayIndexOp::*;

    let mut max_dims = 0;
    let mut slice_count = 0;
    let mut is_slice: Vec<bool> = Vec::new();
    let mut gather_indices: Vec<Rc<Array>> = Vec::new();

    let shape = src.shape();

    // prepare the gather indices
    let mut axes = Vec::new();
    let mut operation_len: usize = 0;
    for (i, op) in operations.enumerate() {
        axes.push(i as i32);
        operation_len += 1;
        match op {
            TakeIndex { index } => {
                let item = Array::from_int(resolve_index_unchecked(
                    *index,
                    src.dim(i as i32) as usize,
                ) as i32);
                gather_indices.push(Rc::new(item));
                is_slice.push(false);
            }
            Slice {
                start,
                stop,
                stride,
            } => {
                slice_count += 1;
                is_slice.push(true);

                let size = shape[i];
                let absolute_start = absolute_start(*start, *stride, size);
                let absolute_end = absolute_end(*stop, *stride, size);

                // TODO: check if this is correct when stride is negative
                let indices: Vec<i32> = (absolute_start..absolute_end)
                    .step_by(stride.abs() as usize)
                    .collect();
                let item = Array::from_slice(&indices, &[indices.len() as i32]);

                gather_indices.push(Rc::new(item));
            }
            TakeArray { indices } => {
                is_slice.push(false);
                max_dims = max_dims.max(indices.ndim());
                gather_indices.push(Rc::clone(indices));
            }
            ExpandDims => {
                unreachable!("ExpandDims(ie. NewAxis) is already filtered out")
            }
        }
    }

    // reshape them so that the int/array indices are first
    if gather_first {
        if slice_count > 0 {
            let mut slice_index = 0;
            for (i, item) in gather_indices.iter_mut().enumerate() {
                if is_slice[i] {
                    let mut new_shape = vec![1; max_dims + slice_count];
                    new_shape[max_dims + slice_index] = item.dim(0);
                    *item = Rc::new(reshape(&item, &new_shape));
                    slice_index += 1;
                } else {
                    let mut new_shape = item.shape().to_vec();
                    new_shape.extend((0..slice_count).map(|_| 1));
                    *item = Rc::new(reshape(&item, &new_shape));
                }
            }
        }
    } else {
        // reshape them so that the int/array indices are last
        for (i, item) in gather_indices[..slice_count].iter_mut().enumerate() {
            let mut new_shape = vec![1; max_dims + slice_count];
            new_shape[i] = item.dim(0);
            *item = Rc::new(reshape(&item, &new_shape));
        }
    }

    // Do the gather
    // let indices = new_mlx_vector_array(gather_indices);
    // SAFETY: indices will be freed at the end of this function. The lifetime of the items in
    // `gather_indices` is managed by the `gather_indices` vector.
    let indices = unsafe {
        let c_vec_array = mlx_sys::mlx_vector_array_new();
        for item in gather_indices.iter() {
            mlx_sys::mlx_vector_array_add(c_vec_array, item.c_array);
        }
        c_vec_array
    };
    let mut slice_sizes = shape.to_vec();
    (0..operation_len).for_each(|i| slice_sizes[i] = 1);

    let gathered = unsafe {
        let c_array = mlx_sys::mlx_gather(
            src.c_array,
            indices,
            axes.as_ptr(),
            axes.len(),
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_ptr(),
        );

        Array::from_ptr(c_array)
    };
    let gathered_shape = gathered.shape();

    // Squeeze the dims
    let output_shape: Vec<i32> = gathered_shape[0..(max_dims + slice_count)]
        .iter()
        .chain(gathered_shape[(max_dims + slice_count + operation_len)..].iter())
        .map(|i| *i)
        .collect();
    let result = gathered.reshape(&output_shape);

    unsafe {
        mlx_sys::mlx_free(indices as *mut _);
    }

    (max_dims, result)
}

#[inline]
fn get_item_int(src: &Array, index: i32, axis: i32, stream: StreamOrDevice) -> Array {
    src.take_device(&index.into(), axis, stream)
}

#[inline]
fn get_item_array(src: &Array, indices: &Array, axis: i32, stream: StreamOrDevice) -> Array {
    src.take_device(indices, axis, stream)
}

#[inline]
fn get_item_slice(src: &Array, start: Bound<i32>, stop: Bound<i32>, stride: i32, stream: StreamOrDevice) -> Array {
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

    src.slice_device(&starts, &stops, &strides, stream)
}

// See `mlx_get_item` in python/src/indexing.cpp and `getItem` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item(src: &Array, index: impl ArrayIndex, stream: StreamOrDevice) -> Array {
    use ArrayIndexOp::*;

    match index.index_op() {
        TakeIndex { index } => get_item_int(src, index, 0, stream),
        TakeArray { indices } => get_item_array(src, &indices, 0, stream),
        Slice {
            start,
            stop,
            stride,
        } => get_item_slice(src, start, stop, stride, stream),
        ExpandDims => unsafe {
            // SAFETY: 0 is always a valid axis
            expand_dims_device_unchecked(src, &[0], stream)
        },
    }
}

// See `mlx_get_item_nd` in python/src/indexing.cpp and `getItemNd` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item_nd(src: &Array, operations: &[ArrayIndexOp], stream: StreamOrDevice) -> Array {
    use ArrayIndexOp::*;

    let mut src = Cow::Borrowed(src);

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
    for item in operations.iter() {
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

    let array_count = operations.iter().filter(|op| op.is_array()).count();
    let have_array = array_count > 0;

    let mut remaining_indices: Vec<ArrayIndexOp> = Vec::new();
    if have_array {
        // apply all the operations (except for .newAxis) up to and including the
        // final .array operation (array operations are implemented via gather)
        let last_array_or_index = operations
            .iter()
            .rposition(|op| op.is_array_or_index())
            .unwrap(); // safe because we know there is at least one array operation
                       // let remaining = indices.split_off(last_array_or_index + 1);
        let gather_indices = operations[..=last_array_or_index]
            .iter()
            .filter(|op| !matches!(op, ArrayIndexOp::ExpandDims));
        let (max_dims, gathered) = gather_nd(&*src, gather_indices, gather_first, stream.clone());

        src = Cow::Owned(gathered);

        // Reassemble the indices for the slicing or reshaping if there are any
        if gather_first {
            remaining_indices.extend((0..max_dims).map(|_| ArrayIndexOp::Slice {
                start: Bound::Unbounded,
                stop: Bound::Unbounded,
                stride: 1,
            }));

            // copy any newAxis in the gatherIndices through.  any slices get
            // copied in as full range (already applied)
            for item in &operations[..=last_array_or_index] {
                match item {
                    ExpandDims => remaining_indices.push(item.clone()),
                    Slice { .. } => remaining_indices.push(ArrayIndexOp::range_full()),
                    _ => {}
                }
            }

            // append the remaining operations
            remaining_indices.extend(operations[(last_array_or_index + 1)..].iter().cloned());
        } else {
            // !gather_first
            for item in operations {
                match item {
                    TakeIndex { .. } | TakeArray { .. } => break,
                    ExpandDims => remaining_indices.push(item.clone()),
                    _ => remaining_indices.push(ArrayIndexOp::range_full()),
                }
            }

            // handle the trailing gathers
            remaining_indices.extend((0..max_dims).map(|_| ArrayIndexOp::range_full()));

            // and the remaining operations
            remaining_indices.extend(operations[(last_array_or_index + 1)..].iter().cloned());
        }
    }

    if have_array && remaining_indices.is_empty() {
        // `clone` returns a new array with the same shape and data
        match src {
            Cow::Borrowed(src) => return src.clone(),
            Cow::Owned(src) => return src,
        }
    }

    if remaining_indices.is_empty() {
        remaining_indices = operations.to_vec();
    }

    // Slice handling
    let ndim = src.ndim();
    let mut starts: SmallVec<[i32; 4]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; 4]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; 4]> = smallvec![1; ndim];
    let mut squeeze_needed = false;
    let mut axis = 0;

    for item in &remaining_indices {
        match item {
            ExpandDims => continue,
            TakeIndex { mut index } => {
                if !have_array {
                    index = resolve_index_unchecked(index, src.dim(axis) as usize) as i32;
                    starts[axis as usize] = index;
                    ends[axis as usize] = index + 1;
                    squeeze_needed = true;
                }
            }
            Slice {
                start,
                stop,
                stride,
            } => {
                let size = src.dim(axis);
                starts[axis as usize] = slice_start(*start, *stride, size);
                ends[axis as usize] = slice_end(*stop, *stride, size);
                strides[axis as usize] = *stride;
            }
            _ => unreachable!("Unexpected item in remaining_indices: {:?}", item),
        }

        axis += 1;
    }

    src = unsafe {
        let c_array = mlx_sys::mlx_slice(
            src.c_array,
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ptr(),
        );

        Cow::Owned(Array::from_ptr(c_array))
    };

    // Unsqueeze handling
    if remaining_indices.len() > ndim || squeeze_needed {
        let mut new_shape = SmallVec::<[i32; 4]>::new();
        let mut axis_ = 0;
        for item in remaining_indices {
            match item {
                ExpandDims => new_shape.push(1),
                TakeIndex { .. } => {
                    if squeeze_needed {
                        axis_ += 1;
                    }
                }
                _ => {
                    new_shape.push(src.dim(axis_));
                    axis_ += 1;
                }
            }
        }
        new_shape.extend(src.shape()[(axis_ as usize)..].iter().cloned());

        src = Cow::Owned(src.reshape(&new_shape));
    }

    match src {
        Cow::Borrowed(src) => src.clone(),
        Cow::Owned(src) => src,
    }
}

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

impl ArrayIndex for Array {
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::TakeArray {
            indices: Rc::new(self),
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

/* -------------------------------------------------------------------------- */
/*                                 Unit tests                                 */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_item() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let mut b = a.index(1);
        b.eval();

        assert_eq!(b.item::<f32>(), 2.0);
    }
}
