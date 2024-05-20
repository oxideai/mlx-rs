use std::rc::Rc;

use smallvec::{smallvec, SmallVec};

use crate::{
    error::SliceError,
    ops::{
        broadcast_arrays_device_unchecked, broadcast_to_device,
        indexing::{count_non_new_axis_operations, expand_ellipsis_operations, IntoStrideBy},
    },
    utils::{resolve_index_signed_unchecked, OwnedOrRef},
    Array, StreamOrDevice,
};

use super::{ArrayIndexOp, RangeIndex};

impl Array {
    pub(crate) unsafe fn slice_update_device_unchecked(
        &self,
        update: &Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_slice_update(
                self.as_ptr(),
                update.as_ptr(),
                starts.as_ptr(),
                starts.len(),
                ends.as_ptr(),
                ends.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_ptr(),
            );
            Array::from_ptr(c_array)
        }
    }

    pub(crate) fn try_slice_update_device(
        &self,
        update: &Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Result<Array, SliceError> {
        self.check_slice_index_dimensions(starts, ends, strides)?;
        unsafe { Ok(self.slice_update_device_unchecked(update, starts, ends, strides, stream)) }
    }

    pub(crate) fn slice_update_device(
        &self,
        update: &Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        stream: StreamOrDevice,
    ) -> Array {
        self.try_slice_update_device(update, starts, ends, strides, stream)
            .unwrap()
    }
}

// See `updateSlice` in the swift binding or `mlx_slice_update` in the python binding
fn update_slice(
    src: &Array, // TODO: should this be &mut?
    operations: &[ArrayIndexOp],
    update: &Array,
    stream: StreamOrDevice,
) -> Option<Array> {
    use OwnedOrRef::*;

    let ndim = src.ndim();
    if ndim == 0 || operations.is_empty() {
        return None;
    }

    // Remove leading singletons dimensions from the update
    let mut update = remove_leading_singleton_dimensions(update, stream.clone());

    // Build slice update params
    let mut starts: SmallVec<[i32; 4]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; 4]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; 4]> = smallvec![1; ndim];

    // If it's just a simple slice, just do a slice update and return
    if operations.len() == 1 {
        if let ArrayIndexOp::Slice(range_index) = &operations[0] {
            let size = src.dim(0);
            starts[0] = range_index.start(size);
            ends[0] = range_index.end(size);
            strides[0] = range_index.stride();

            return Some(src.slice_update_device(
                &update,
                &starts,
                &ends,
                &strides,
                stream.clone(),
            ));
        }
    }

    // Can't route to slice update if any arrays are present
    if operations.iter().any(|op| op.is_array()) {
        return None;
    }

    // Expand ellipses into a series of ':' (range full) slices
    let operations = expand_ellipsis_operations(ndim, operations);

    // If no non-None indices return the broadcasted update
    if count_non_new_axis_operations(&operations) == 0 {
        return Some(broadcast_to_device(&update, src.shape(), stream.clone()));
    }

    // Process entries
    let mut update_expand_dims = Vec::new(); // TODO: pre-allcate
    let mut axis = 0i32;
    for item in operations.iter() {
        use ArrayIndexOp::*;

        match item {
            TakeIndex { index } => {
                let size = src.dim(axis);
                let index = if index.is_negative() {
                    size + index
                } else {
                    *index
                };
                // SAFETY: axis is always non-negative
                starts[axis as usize] = index;
                ends[axis as usize] = index + 1;
                if ndim - (axis as usize) < update.ndim() {
                    update_expand_dims.push(axis.saturating_sub_unsigned(ndim as u32));
                }

                axis = axis.saturating_add(1);
            }
            Slice(range_index) => {
                let size = src.dim(axis);
                // SAFETY: axis is always non-negative
                starts[axis as usize] = range_index.start(size);
                ends[axis as usize] = range_index.end(size);
                strides[axis as usize] = range_index.stride();
                axis = axis.saturating_add(1);
            }
            ExpandDims => break,
            Ellipsis | TakeArray { indices: _ } => panic!("unexpected item in operations"),
        }
    }

    if !update_expand_dims.is_empty() {
        update = Owned(update.expand_dims_device(&update_expand_dims, stream.clone()));
    }

    Some(src.slice_update_device(&update, &starts, &ends, &strides, stream.clone()))
}

// See `leadingSingletonDimensionsRemoved` in the swift binding
fn remove_leading_singleton_dimensions<'a>(
    a: &'a Array,
    stream: StreamOrDevice,
) -> OwnedOrRef<'a, Array> {
    let shape = a.shape();
    let mut new_shape: Vec<_> = shape.iter().skip_while(|&&dim| dim == 1).cloned().collect();
    if shape != new_shape {
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        // TODO: should we use `reshape_device_unchecked`?
        OwnedOrRef::Owned(a.reshape_device(&new_shape, stream))
    } else {
        OwnedOrRef::Ref(a)
    }
}

/// See `scatterArguments` in the swift binding
fn scatter_args<'a>(
    src: &'a Array,
    operations: &'a [ArrayIndexOp],
    update: &Array,
    stream: StreamOrDevice,
) -> (
    SmallVec<[OwnedOrRef<'a, Array>; 4]>,
    Array,
    SmallVec<[i32; 4]>,
) {
    use ArrayIndexOp::*;
    use OwnedOrRef::*;

    if operations.len() == 1 {
        return match &operations[0] {
            TakeIndex { index } => scatter_args_index(src, *index, update, stream),
            TakeArray { indices } => scatter_args_array(src, Ref(indices), update, stream),
            Slice(range_index) => scatter_args_slice(src, range_index, update, stream),
            ExpandDims => (
                smallvec![],
                broadcast_to_device(&update, src.shape(), stream.clone()),
                smallvec![],
            ),
            Ellipsis => panic!("Unable to update array with ellipsis argument"),
        };
    }

    scatter_args_nd(src, operations, update, stream)
}

fn scatter_args_index<'a>(
    src: &'a Array,
    index: i32,
    update: &Array,
    stream: StreamOrDevice,
) -> (
    SmallVec<[OwnedOrRef<'a, Array>; 4]>,
    Array,
    SmallVec<[i32; 4]>,
) {
    // mlx_scatter_args_index

    // Remove any leading singleton dimensions from the update
    // and then broadcast update to shape of src[0, ...]
    let update = remove_leading_singleton_dimensions(update, stream.clone());

    let mut shape: SmallVec<[i32; 4]> = SmallVec::from_slice(src.shape());
    shape[0] = 1;

    (
        smallvec![OwnedOrRef::Owned(Array::from_int(
            resolve_index_signed_unchecked(index, src.dim(0))
        ))],
        broadcast_to_device(&update, &shape, stream.clone()),
        smallvec![0],
    )
}

fn scatter_args_array<'a>(
    src: &'a Array,
    a: OwnedOrRef<'a, Array>,
    update: &Array,
    stream: StreamOrDevice,
) -> (
    SmallVec<[OwnedOrRef<'a, Array>; 4]>,
    Array,
    SmallVec<[i32; 4]>,
) {
    // mlx_scatter_args_array

    // trim leading singleton dimensions
    let update = remove_leading_singleton_dimensions(update, stream.clone());

    // The update shape must broadcast with indices.shape + [1] + src.shape[1:]
    let mut update_shape: SmallVec<[i32; 4]> = a
        .shape()
        .iter()
        .chain(src.shape().iter().skip(1))
        .cloned()
        .collect();
    let update = broadcast_to_device(&update, &update_shape, stream.clone());

    update_shape.insert(a.ndim(), 1);
    let update = update.reshape_device(&update_shape, stream.clone());

    (smallvec![a], update, smallvec![0])
}

fn scatter_args_slice<'a>(
    src: &'a Array,
    range_index: &'a RangeIndex,
    update: &Array,
    stream: StreamOrDevice,
) -> (
    SmallVec<[OwnedOrRef<'a, Array>; 4]>,
    Array,
    SmallVec<[i32; 4]>,
) {
    use OwnedOrRef::*;

    // mlx_scatter_args_slice

    // if none slice is requested braodcast the update to the src size and return it
    if range_index.is_full() {
        let update = remove_leading_singleton_dimensions(&update, stream.clone());

        return (
            smallvec![],
            broadcast_to_device(&update, src.shape(), stream.clone()),
            smallvec![],
        );
    }

    let size = src.dim(0);
    let start = range_index.start(size);
    let end = range_index.end(size);
    let stride = range_index.stride();

    // If simple stride
    if stride == 1 {
        let update = remove_leading_singleton_dimensions(&update, stream.clone());

        // Broadcast update to slice size
        let update_broadcast_shape: SmallVec<[i32; 4]> = (1..end - start)
            .chain(src.shape().iter().skip(1).cloned())
            .collect();
        let update = broadcast_to_device(&update, &update_broadcast_shape, stream.clone());

        let indices = Array::from_int(start).reshape_device(&[1], stream.clone()); // TODO: is reshape really needed?
        (smallvec![OwnedOrRef::Owned(indices)], update, smallvec![0])
    } else {
        // stride != 1, convert the slice to an array
        let a_vals = strided_range_to_vec(start, end, stride);
        let a = Array::from_slice(&a_vals, &[a_vals.len() as i32]);

        scatter_args_array(src, Owned(a), update, stream)
    }
}

fn scatter_args_nd<'a>(
    src: &'a Array,
    operations: &'a [ArrayIndexOp],
    update: &Array,
    stream: StreamOrDevice,
) -> (
    SmallVec<[OwnedOrRef<'a, Array>; 4]>,
    Array,
    SmallVec<[i32; 4]>,
) {
    use ArrayIndexOp::*;

    // mlx_scatter_args_nd

    let shape = src.shape();

    let operations = expand_ellipsis_operations(src.ndim(), operations);
    let update = remove_leading_singleton_dimensions(&update, stream.clone());

    // If no non-newAxis indices return the broadcasted update
    let non_new_axis_operation_count = count_non_new_axis_operations(&operations);
    if non_new_axis_operation_count == 0 {
        return (
            smallvec![],
            broadcast_to_device(&update, shape, stream.clone()),
            smallvec![],
        );
    }

    // Analyse the types of the indices
    let mut max_dims = 0;
    let mut arrays_first = false;
    let mut count_new_axis: i32 = 0;
    let mut count_slices: i32 = 0;
    let mut count_arrays: i32 = 0;
    let mut count_strided_slices: i32 = 0;
    let mut count_simple_slices_post: i32 = 0;

    let mut have_array = false;
    let mut have_non_array = false;

    for item in operations.iter() {
        match item {
            TakeIndex { index: _ } => {
                // ignore
                break;
            }
            Slice(range_index) => {
                have_non_array = have_array;
                count_slices = count_slices.saturating_add(1);
                if range_index.stride() != 1 {
                    count_strided_slices = count_strided_slices.saturating_add(1);
                    count_simple_slices_post = 0;
                } else {
                    count_simple_slices_post = count_simple_slices_post.saturating_add(1);
                }
            }
            TakeArray { indices } => {
                have_array = true;
                if have_array && have_non_array {
                    arrays_first = true;
                }
                max_dims = indices.ndim().max(max_dims);
                count_arrays = count_arrays.saturating_add(1);
                count_simple_slices_post = 0;
            }
            ExpandDims => {
                have_non_array = true;
                count_new_axis = count_new_axis.saturating_add(1);
            }
            Ellipsis => panic!("Unexpected item ellipsis in scatter_args_nd"),
        }
    }

    // We have index dims for the arrays, strided slices (implemented as arrays), none
    let mut index_dims = (max_dims + count_new_axis as usize + count_slices as usize)
        .saturating_sub(count_simple_slices_post as usize);

    // If we have simple non-strided slices, we also attach an index for that
    if index_dims == 0 {
        index_dims = 1;
    }

    // Go over each index type and translate to the needed scatter args
    let mut array_indices: Vec<Array> = Vec::new(); // TODO: pre-allocate
    let mut slice_number: i32 = 0;
    let mut array_number: i32 = 0;
    let mut axis: i32 = 0;

    // We collect the shapes of the slices and updates during this process
    let mut update_shape = vec![1; non_new_axis_operation_count];
    let mut slice_shapes: Vec<i32> = Vec::new(); // TODO: pre-allocate

    for item in operations.iter() {
        match item {
            TakeIndex { index } => {
                let resolved_index = resolve_index_signed_unchecked(*index, src.dim(axis));
                array_indices.push(Array::from_int(resolved_index));
                // SAFETY: axis is always non-negative
                update_shape[axis as usize] = 1;
                axis = axis.saturating_add(1);
            }
            Slice(range_index) => {
                let size = src.dim(axis);
                let start = range_index.absolute_start(size);
                let end = range_index.absolute_end(size);
                let stride = range_index.stride();

                let mut index_shape = vec![1; index_dims];

                // If it's a simple slice, we only need to add the start index
                if array_number >= count_arrays && count_strided_slices <= 0 && stride == 1 {
                    let index = Array::from_int(start).reshape_device(&index_shape, stream.clone());
                    let slice_shape_entry = end - start;
                    slice_shapes.push(slice_shape_entry);
                    array_indices.push(index);

                    // Add the shape to the update
                    update_shape[axis as usize] = slice_shape_entry;
                } else {
                    // Otherwise we expand the slice into indices using arange
                    let index_vals = strided_range_to_vec(start, end, stride);
                    let index = Array::from_slice(&index_vals, &[index_vals.len() as i32]);
                    let location = if arrays_first {
                        slice_number.saturating_add(max_dims as i32)
                    } else {
                        slice_number
                    };
                    index_shape[location as usize] = index.size() as i32;
                    array_indices.push(index.reshape_device(&index_shape, stream.clone()));

                    slice_number = slice_number.saturating_add(1);
                    count_strided_slices = count_strided_slices.saturating_sub(1);

                    // Add the shape to the update
                    update_shape[axis as usize] = 1;
                }

                axis = axis.saturating_add(1);
            }
            TakeArray { indices } => {
                // Place the arrays in the correct dimension
                let start = if arrays_first {
                    max_dims - indices.ndim()
                } else {
                    // SAFETY: slice_number is never decremented and should be non-negative
                    slice_number as usize + max_dims - indices.ndim()
                };
                let mut new_shape = vec![1; index_dims];

                for j in 0..indices.ndim() {
                    new_shape[start + j] = indices.dim(j as i32);
                }

                array_indices.push(indices.reshape_device(&new_shape, stream.clone()));
                array_number = array_number.saturating_add(1);

                if !arrays_first && array_number == count_arrays {
                    slice_number = slice_number.saturating_add_unsigned(max_dims as u32);
                }

                // Add the shape to the update
                update_shape[axis as usize] = 1;
                axis = axis.saturating_add(1);
            }
            ExpandDims => slice_number = slice_number.saturating_add(1),
            Ellipsis => panic!("Unexpected item ellipsis in scatter_args_nd"),
        }
    }

    // Broadcast the update to the indices and slices
    let array_indices =
        unsafe { broadcast_arrays_device_unchecked(&array_indices, stream.clone()) };
    let update_shape_broadcast: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(slice_shapes.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();
    let update = broadcast_to_device(&update, &update_shape_broadcast, stream.clone());

    // Reshape the update with the size-1 dims for the int and array indices
    let update_reshape: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(update_shape.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();

    let update = update.reshape_device(&update_reshape, stream.clone());

    let array_indices_len = array_indices.len();
    (
        array_indices
            .into_iter()
            .map(|i| OwnedOrRef::Owned(i))
            .collect(),
        update,
        (0..array_indices_len as i32).collect(),
    )
}

fn strided_range_to_vec(start: i32, end: i32, stride: i32) -> Vec<i32> {
    let estimated_capacity = (end - start).abs() / stride.abs();
    let mut vec = Vec::with_capacity(estimated_capacity as usize);
    let mut current = start;

    assert_ne!(stride, 0, "Stride cannot be zero");

    if stride.is_negative() {
        while current >= end {
            vec.push(current);
            current += stride;
        }
    } else {
        while current <= end {
            vec.push(current);
            current += stride;
        }
    }

    vec
}
