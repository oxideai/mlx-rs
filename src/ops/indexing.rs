//! Indexing Arrays
//!
//! TODO: fix documentation

use std::{
    borrow::Cow,
    ops::{Bound, RangeBounds},
    rc::Rc,
};

use mlx_macros::default_device;
use smallvec::{smallvec, SmallVec};

use crate::{
    error::{InvalidAxisError, SliceError, TakeAlongAxisError, TakeError},
    ops::{expand_dims_device_unchecked, reshape},
    utils::resolve_index_unchecked,
    Array, StreamOrDevice,
};

/* -------------------------------------------------------------------------- */
/*                                Custom types                                */
/* -------------------------------------------------------------------------- */

#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

#[derive(Debug, Clone, Copy)]
pub struct Ellipsis;

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
pub struct RangeIndex {
    start: i32,
    stop: Option<i32>,
    stride: i32,
}

impl RangeIndex {
    pub(crate) fn new(start: Bound<i32>, stop: Bound<i32>, stride: Option<i32>) -> Self {
        let start = match start {
            Bound::Included(start) => start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };
        let stop = match stop {
            Bound::Included(stop) => Some(stop + 1),
            Bound::Excluded(stop) => Some(stop),
            Bound::Unbounded => None,
        };
        let stride = stride.unwrap_or(1);
        Self {
            start,
            stop,
            stride,
        }
    }

    pub(crate) fn stride(&self) -> i32 {
        self.stride
    }

    pub(crate) fn start(&self, size: i32) -> i32 {
        // _start ?? (stride < 0 ? size - 1 : 0)

        if self.stride.is_negative() {
            self.start + size
        } else {
            self.start
        }
    }

    pub(crate) fn absolute_start(&self, size: i32) -> i32 {
        // let start = self.start(size)
        // return start < 0 ? start + size : start

        let start = self.start(size);
        if start.is_negative() {
            start + size
        } else {
            start
        }
    }

    pub(crate) fn end(&self, size: i32) -> i32 {
        // _end ?? (stride < 0 ? -size - 1 : size)

        self.stop.unwrap_or_else(|| {
            if self.stride.is_negative() {
                -size - 1
            } else {
                size
            }
        })
    }

    pub(crate) fn absolute_end(&self, size: i32) -> i32 {
        // let end = self.end(size)
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
    Ellipsis,
    TakeIndex { index: i32 },
    TakeArray { indices: Rc<Array> },
    Slice(RangeIndex),
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
        ArrayIndexOp::Slice(RangeIndex::new(Bound::Unbounded, Bound::Unbounded, Some(1)))
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

impl<T> ArrayIndex for Stride<T>
where
    T: IndexBounds,
{
    fn index_op(self) -> ArrayIndexOp {
        ArrayIndexOp::Slice(RangeIndex::new(
            self.iter.start_bound().cloned(),
            self.iter.end_bound().cloned(),
            Some(self.stride),
        ))
    }
}

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
        let i = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
        ];
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

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */

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
            Slice(range) => {
                slice_count += 1;
                is_slice.push(true);

                let size = shape[i];
                let absolute_start = range.absolute_start(size);
                let absolute_end = range.absolute_end(size);

                // TODO: check if this is correct when stride is negative
                let indices: Vec<i32> = (absolute_start..absolute_end)
                    .step_by(range.stride().abs() as usize)
                    .collect();
                let item = Array::from_slice(&indices, &[indices.len() as i32]);

                gather_indices.push(Rc::new(item));
            }
            TakeArray { indices } => {
                is_slice.push(false);
                max_dims = max_dims.max(indices.ndim());
                gather_indices.push(Rc::clone(indices));
            }
            Ellipsis | ExpandDims => {
                unreachable!("Unexpected operation in gather_nd")
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
fn get_item_slice(src: &Array, range: RangeIndex, stream: StreamOrDevice) -> Array {
    let ndim = src.ndim();
    let mut starts: SmallVec<[i32; 4]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; 4]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; 4]> = smallvec![1; ndim];

    let size = ends[0];
    starts[0] = range.start(size);
    ends[0] = range.end(size);
    strides[0] = range.stride();

    src.slice_device(&starts, &ends, &strides, stream)
}

fn count_non_new_axis_operations(operations: &[ArrayIndexOp]) -> usize {
    operations
        .iter()
        .filter(|op| !matches!(op, ArrayIndexOp::ExpandDims))
        .count()
}

fn expand_ellipsis_operations<'a>(
    ndim: usize,
    operations: &'a [ArrayIndexOp],
) -> Cow<'a, [ArrayIndexOp]> {
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
    let expand = expand_range.map(|_| ArrayIndexOp::range_full());

    let mut expanded = Vec::with_capacity(ndim);
    expanded.extend_from_slice(prefix);
    expanded.extend(expand);
    expanded.extend_from_slice(suffix);

    Cow::Owned(expanded)
}

// See `mlx_get_item` in python/src/indexing.cpp and `getItem` in
// mlx-swift/Sources/MLX/MLXArray+Indexing.swift
fn get_item(src: &Array, index: impl ArrayIndex, stream: StreamOrDevice) -> Array {
    use ArrayIndexOp::*;

    match index.index_op() {
        Ellipsis => src.clone(),
        TakeIndex { index } => get_item_int(src, index, 0, stream),
        TakeArray { indices } => get_item_array(src, &indices, 0, stream),
        Slice(range) => get_item_slice(src, range, stream),
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

    // The plan is as follows:
    // 1. Replace the ellipsis with a series of slice(None)
    // 2. Loop over the indices and calculate the gather indices
    // 3. Calculate the remaining slices and reshapes

    let operations = expand_ellipsis_operations(src.ndim(), operations);

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
            .filter(|op| match op {
                Ellipsis | ExpandDims => false,
                _ => true,
            });
        let (max_dims, gathered) = gather_nd(&*src, gather_indices, gather_first, stream.clone());

        src = Cow::Owned(gathered);

        // Reassemble the indices for the slicing or reshaping if there are any
        if gather_first {
            remaining_indices.extend((0..max_dims).map(|_| ArrayIndexOp::range_full()));

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
            for item in operations.iter() {
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

    for (axis, item) in remaining_indices.iter().enumerate() {
        match item {
            ExpandDims => continue,
            TakeIndex { mut index } => {
                if !have_array {
                    index = resolve_index_unchecked(index, src.dim(axis as i32) as usize) as i32;
                    starts[axis] = index;
                    ends[axis] = index + 1;
                    squeeze_needed = true;
                }
            }
            Slice(range) => {
                let size = src.dim(axis as i32);
                starts[axis] = range.start(size);
                ends[axis] = range.end(size);
                strides[axis] = range.stride();
            }
            _ => unreachable!("Unexpected item in remaining_indices: {:?}", item),
        }
    }

    src = Cow::Owned(src.slice_device(&starts, &ends, &strides, stream));

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

/* -------------------------------------------------------------------------- */
/*                                 Unit tests                                 */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_array_all_close {
        ($a:tt, $b:tt) => {
            let _b: Array = $b.into();
            let mut assert = $a.all_close(&_b, None, None, None);
            assert.eval();
            assert!(assert.item::<bool>());
        };
    }

    #[test]
    fn test_array_index_new_axis() {
        let a = Array::from_iter(0..60, &[3, 4, 5]);
        let mut s = a.index(NewAxis);
        s.eval();

        assert_eq!(s.ndim(), 4);
        assert_eq!(s.shape(), &[1, 3, 4, 5]);

        let expected = Array::from_iter(0..60, &[1, 3, 4, 5]);
        assert_array_all_close!(s, expected);
    }

    #[test]
    fn test_array_index_ellipsis() {
        let a = Array::from_iter(0i32..8, &[2, 2, 2]);

        let mut s1 = a.index((.., .., 0));
        s1.eval();
        let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
        assert_array_all_close!(s1, expected);

        let mut s2 = a.index((Ellipsis, .., 0));
        s2.eval();
        let expected = Array::from_slice(&[0, 2, 4, 6], &[2, 2]);
        assert_array_all_close!(s2, expected);
    }

    // The unit tests below are ported from the swift binding.
    // See `mlx-swift/Tests/MLXTests/MLXArray+IndexingTests.swift`

    #[test]
    fn test_array_subscript_int() {
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let mut s = a.index(1);
        s.eval();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.shape(), &[8, 8]);

        let expected = Array::from_iter(64..128, &[8, 8]);
        assert_array_all_close!(s, expected);
    }

    #[test]
    fn test_array_subscript_int_array() {
        // squeeze output dimensions as needed
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let mut s1 = a.index((1, 2));
        s1.eval();
        assert_eq!(s1.ndim(), 1);
        assert_eq!(s1.shape(), &[8]);

        let expected = Array::from_iter(80..88, &[8]);
        assert_array_all_close!(s1, expected);

        let mut s2 = a.index((1, 2, 3));
        s2.eval();
        assert_eq!(s2.ndim(), 0);
        assert!(s2.shape().is_empty());
        assert_eq!(s2.item::<i32>(), 64 + 2 * 8 + 3);
    }

    #[test]
    fn test_array_subscript_int_array_2() {
        // last dimension should not be squeezed
        let a = Array::from_iter(0i32..512, &[8, 8, 8, 1]);

        let mut s = a.index(1);
        s.eval();
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.shape(), &[8, 8, 1]);

        let mut s1 = a.index((1, 2));
        s1.eval();
        assert_eq!(s1.ndim(), 2);
        assert_eq!(s1.shape(), &[8, 1]);

        let mut s2 = a.index((1, 2, 3));
        s2.eval();
        assert_eq!(s2.ndim(), 1);
        assert_eq!(s2.shape(), &[1]);
    }

    #[test]
    fn test_array_subscript_from_end() {
        let a = Array::from_iter(0i32..12, &[3, 4]);

        let mut s = a.index((-1, -2));
        s.eval();

        assert_eq!(s.ndim(), 0);
        assert_eq!(s.item::<i32>(), 10);
    }

    #[test]
    fn test_array_subscript_range() {
        let a = Array::from_iter(0i32..512, &[8, 8, 8]);

        let mut s1 = a.index(1..3);
        s1.eval();
        assert_eq!(s1.ndim(), 3);
        assert_eq!(s1.shape(), &[2, 8, 8]);
        let expected = Array::from_iter(64..192, &[2, 8, 8]);
        assert_array_all_close!(s1, expected);

        // even though the first dimension is 1 we do not squeeze it
        let mut s2 = a.index(1..=1);
        s2.eval();
        assert_eq!(s2.ndim(), 3);
        assert_eq!(s2.shape(), &[1, 8, 8]);
        let expected = Array::from_iter(64..128, &[1, 8, 8]);
        assert_array_all_close!(s2, expected);

        // multiple ranges, resolving RangeExpressions vs the dimensions
        let mut s3 = a.index((1..2, ..3, 3..));
        s3.eval();
        assert_eq!(s3.ndim(), 3);
        assert_eq!(s3.shape(), &[1, 3, 5]);
        let expected = Array::from_slice(
            &[67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87],
            &[1, 3, 5],
        );
        assert_array_all_close!(s3, expected);

        let mut s4 = a.index((-2..-1, ..-3, -3..));
        s4.eval();
        assert_eq!(s4.ndim(), 3);
        assert_eq!(s4.shape(), &[1, 5, 3]);
        let expected = Array::from_slice(
            &[
                389, 390, 391, 397, 398, 399, 405, 406, 407, 413, 414, 415, 421, 422, 423,
            ],
            &[1, 5, 3],
        );
        assert_array_all_close!(s4, expected);
    }

    #[test]
    fn test_array_subscript_advanced() {
        // advanced subscript examples taken from
        // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

        let a = Array::from_iter(0..35, &[5, 7]).as_type::<i32>();

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let i2 = Array::from_slice(&[0, 1, 2], &[3]);

        let mut s1 = a.index((i1, i2));
        s1.eval();

        assert_eq!(s1.ndim(), 1);
        assert_eq!(s1.shape(), &[3]);

        let expected = Array::from_slice(&[0i32, 15, 30], &[3]);
        assert_array_all_close!(s1, expected);
    }

    #[test]
    fn test_array_subscript_advanced_2() {
        let a = Array::from_iter(0..12, &[6, 2]).as_type::<i32>();

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let mut s2 = a.index(i1);
        s2.eval();

        let expected = Array::from_slice(&[0i32, 1, 4, 5, 8, 9], &[3, 2]);
        assert_array_all_close!(s2, expected);
    }

    #[test]
    fn test_array_subscript_advanced_2d() {
        let a = Array::from_iter(0..12, &[4, 3]).as_type::<i32>();

        let rows = Array::from_slice(&[0, 0, 3, 3], &[2, 2]);
        let cols = Array::from_slice(&[0, 2, 0, 2], &[2, 2]);

        let mut s = a.index((rows, cols));
        s.eval();

        let expected = Array::from_slice(&[0, 2, 9, 11], &[2, 2]);
        assert_array_all_close!(s, expected);
    }

    #[test]
    fn test_array_subscript_advanced_2d_2() {
        let a = Array::from_iter(0..12, &[4, 3]).as_type::<i32>();

        let rows = Array::from_slice(&[0, 3], &[2, 1]);
        let cols = Array::from_slice(&[0, 2], &[2]);

        let mut s = a.index((rows, cols));
        s.eval();

        let expected = Array::from_slice(&[0, 2, 9, 11], &[2, 2]);
        assert_array_all_close!(s, expected);
    }
}
