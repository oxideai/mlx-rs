use smallvec::{smallvec, SmallVec};

use crate::{
    constants::DEFAULT_STACK_VEC_LEN,
    error::Exception,
    ops::{
        broadcast_arrays_device, broadcast_to_device,
        indexing::{count_non_new_axis_operations, expand_ellipsis_operations},
    },
    utils::{resolve_index_signed_unchecked, OwnedOrRef, VectorArray},
    Array, Stream,
};

use super::{ArrayIndex, ArrayIndexOp, IndexMutOp, RangeIndex};

impl Array {
    pub(crate) fn slice_update_device(
        &self,
        update: &Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array, Exception> {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_slice_update(
                    self.as_ptr(),
                    update.as_ptr(),
                    starts.as_ptr(),
                    starts.len(),
                    ends.as_ptr(),
                    ends.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }
}

// See `updateSlice` in the swift binding or `mlx_slice_update` in the python binding
fn update_slice(
    src: &Array,
    operations: &[ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<Option<Array>, Exception> {
    use OwnedOrRef::*;

    let ndim = src.ndim();
    if ndim == 0 || operations.is_empty() {
        return Ok(None);
    }

    // Remove leading singletons dimensions from the update
    let mut update = remove_leading_singleton_dimensions(update, &stream)?;

    // Build slice update params
    let mut starts: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![0; ndim];
    let mut ends: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    let mut strides: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = smallvec![1; ndim];

    // If it's just a simple slice, just do a slice update and return
    if operations.len() == 1 {
        if let ArrayIndexOp::Slice(range_index) = &operations[0] {
            let size = src.dim(0);
            starts[0] = range_index.start(size);
            ends[0] = range_index.end(size);
            strides[0] = range_index.stride();

            return Ok(Some(src.slice_update_device(
                &update, &starts, &ends, &strides, &stream,
            )?));
        }
    }

    // Can't route to slice update if any arrays are present
    if operations.iter().any(|op| op.is_array()) {
        return Ok(None);
    }

    // Expand ellipses into a series of ':' (range full) slices
    let operations = expand_ellipsis_operations(ndim, operations);

    // If no non-None indices return the broadcasted update
    if count_non_new_axis_operations(&operations) == 0 {
        return Ok(Some(broadcast_to_device(&update, src.shape(), &stream)?));
    }

    // Process entries
    let mut update_expand_dims: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::new();
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
            ExpandDims => {}
            Ellipsis | TakeArray { indices: _ } => panic!("unexpected item in operations"),
        }
    }

    if !update_expand_dims.is_empty() {
        update = Owned(update.expand_dims_device(&update_expand_dims, &stream)?);
    }

    Ok(Some(src.slice_update_device(
        &update, &starts, &ends, &strides, &stream,
    )?))
}

// See `leadingSingletonDimensionsRemoved` in the swift binding
fn remove_leading_singleton_dimensions(
    a: &Array,
    stream: impl AsRef<Stream>,
) -> Result<OwnedOrRef<'_, Array>, Exception> {
    let shape = a.shape();
    let mut new_shape: Vec<_> = shape.iter().skip_while(|&&dim| dim == 1).cloned().collect();
    if shape != new_shape {
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        Ok(OwnedOrRef::Owned(a.reshape_device(&new_shape, stream)?))
    } else {
        Ok(OwnedOrRef::Ref(a))
    }
}

struct ScatterArgs<'a> {
    indices: SmallVec<[OwnedOrRef<'a, Array>; DEFAULT_STACK_VEC_LEN]>,
    update: Array,
    axes: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]>,
}

/// See `scatterArguments` in the swift binding
fn scatter_args<'a>(
    src: &Array,
    operations: &'a [ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>, Exception> {
    use ArrayIndexOp::*;
    use OwnedOrRef::*;

    if operations.len() == 1 {
        return match &operations[0] {
            TakeIndex { index } => scatter_args_index(src, *index, update, stream),
            TakeArray { indices } => scatter_args_array(src, Ref(indices), update, stream),
            Slice(range_index) => scatter_args_slice(src, range_index, update, stream),
            ExpandDims => Ok(ScatterArgs {
                indices: smallvec![],
                update: broadcast_to_device(update, src.shape(), &stream)?,
                axes: smallvec![],
            }),
            Ellipsis => panic!("Unable to update array with ellipsis argument"),
        };
    }

    scatter_args_nd(src, operations, update, stream)
}

fn scatter_args_index<'a>(
    src: &Array,
    index: i32,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>, Exception> {
    // mlx_scatter_args_index

    // Remove any leading singleton dimensions from the update
    // and then broadcast update to shape of src[0, ...]
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    let mut shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::from_slice(src.shape());
    shape[0] = 1;

    Ok(ScatterArgs {
        indices: smallvec![OwnedOrRef::Owned(Array::from_int(
            resolve_index_signed_unchecked(index, src.dim(0))
        ))],
        update: broadcast_to_device(&update, &shape, &stream)?,
        axes: smallvec![0],
    })
}

fn scatter_args_array<'a>(
    src: &Array,
    a: OwnedOrRef<'a, Array>,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>, Exception> {
    // mlx_scatter_args_array

    // trim leading singleton dimensions
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    // The update shape must broadcast with indices.shape + [1] + src.shape[1:]
    let mut update_shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = a
        .shape()
        .iter()
        .chain(src.shape().iter().skip(1))
        .cloned()
        .collect();
    let update = broadcast_to_device(&update, &update_shape, &stream)?;

    update_shape.insert(a.ndim(), 1);
    let update = update.reshape_device(&update_shape, &stream)?;

    Ok(ScatterArgs {
        indices: smallvec![a],
        update,
        axes: smallvec![0],
    })
}

fn scatter_args_slice<'a>(
    src: &Array,
    range_index: &'a RangeIndex,
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>, Exception> {
    use OwnedOrRef::*;

    // mlx_scatter_args_slice

    // if none slice is requested braodcast the update to the src size and return it
    if range_index.is_full() {
        let update = remove_leading_singleton_dimensions(update, &stream)?;

        return Ok(ScatterArgs {
            indices: smallvec![],
            update: broadcast_to_device(&update, src.shape(), &stream)?,
            axes: smallvec![],
        });
    }

    let size = src.dim(0);
    let start = range_index.start(size);
    let end = range_index.end(size);
    let stride = range_index.stride();

    // If simple stride
    if stride == 1 {
        let update = remove_leading_singleton_dimensions(update, &stream)?;

        // Broadcast update to slice size
        let update_broadcast_shape: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = (1..end - start)
            .chain(src.shape().iter().skip(1).cloned())
            .collect();
        let update = broadcast_to_device(&update, &update_broadcast_shape, &stream)?;

        let indices = Array::from_slice(&[start], &[1]);
        Ok(ScatterArgs {
            indices: smallvec![Owned(indices)],
            update,
            axes: smallvec![0],
        })
    } else {
        // stride != 1, convert the slice to an array
        let a_vals = strided_range_to_vec(start, end, stride);
        let a = Array::from_slice(&a_vals, &[a_vals.len() as i32]);

        scatter_args_array(src, Owned(a), update, stream)
    }
}

fn scatter_args_nd<'a>(
    src: &Array,
    operations: &'a [ArrayIndexOp],
    update: &Array,
    stream: impl AsRef<Stream>,
) -> Result<ScatterArgs<'a>, Exception> {
    use ArrayIndexOp::*;

    // mlx_scatter_args_nd

    let shape = src.shape();

    let operations = expand_ellipsis_operations(src.ndim(), operations);
    let update = remove_leading_singleton_dimensions(update, &stream)?;

    // If no non-newAxis indices return the broadcasted update
    let non_new_axis_operation_count = count_non_new_axis_operations(&operations);
    if non_new_axis_operation_count == 0 {
        return Ok(ScatterArgs {
            indices: smallvec![],
            update: broadcast_to_device(&update, shape, &stream)?,
            axes: smallvec![],
        });
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
    let mut array_indices: SmallVec<[Array; DEFAULT_STACK_VEC_LEN]> =
        SmallVec::with_capacity(operations.len());
    let mut slice_number: i32 = 0;
    let mut array_number: i32 = 0;
    let mut axis: i32 = 0;

    // We collect the shapes of the slices and updates during this process
    let mut update_shape = vec![1; non_new_axis_operation_count];
    let mut slice_shapes: SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> = SmallVec::new();

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
                    let index = Array::from_int(start).reshape_device(&index_shape, &stream)?;
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
                    array_indices.push(index.reshape_device(&index_shape, &stream)?);

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

                array_indices.push(indices.reshape_device(&new_shape, &stream)?);
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
    let array_indices = broadcast_arrays_device(&array_indices, &stream)?;
    let update_shape_broadcast: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(slice_shapes.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();
    let update = broadcast_to_device(&update, &update_shape_broadcast, &stream)?;

    // Reshape the update with the size-1 dims for the int and array indices
    let update_reshape: Vec<_> = array_indices[0]
        .shape()
        .iter()
        .chain(update_shape.iter())
        .chain(src.shape().iter().skip(non_new_axis_operation_count))
        .cloned()
        .collect();

    let update = update.reshape_device(&update_reshape, &stream)?;

    let array_indices_len = array_indices.len();

    Ok(ScatterArgs {
        indices: array_indices.into_iter().map(OwnedOrRef::Owned).collect(),
        update,
        axes: (0..array_indices_len as i32).collect(),
    })
}

fn strided_range_to_vec(start: i32, exclusive_end: i32, stride: i32) -> Vec<i32> {
    let estimated_capacity = (exclusive_end - start).abs() / stride.abs();
    let mut vec = Vec::with_capacity(estimated_capacity as usize);
    let mut current = start;

    assert_ne!(stride, 0, "Stride cannot be zero");

    if stride.is_negative() {
        while current > exclusive_end {
            vec.push(current);
            current += stride;
        }
    } else {
        while current < exclusive_end {
            vec.push(current);
            current += stride;
        }
    }

    vec
}

unsafe fn scatter_device_unchecked(
    a: &Array,
    indices: &[impl AsRef<Array>],
    updates: &Array,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Array {
    let indices_vector = VectorArray::from_iter(indices.iter());

    unsafe {
        let result = mlx_sys::mlx_scatter(
            a.as_ptr(),
            indices_vector.as_ptr(),
            updates.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(result)
    }
}

impl Array {
    fn index_mut_device_inner(
        &mut self,
        operations: &[ArrayIndexOp],
        update: &Array,
        stream: impl AsRef<Stream>,
    ) {
        if let Some(result) = update_slice(self, operations, update, &stream).unwrap() {
            *self = result;
            return;
        }

        let ScatterArgs {
            indices,
            update,
            axes,
        } = scatter_args(self, operations, update, &stream).unwrap();
        if !indices.is_empty() {
            let result =
                unsafe { scatter_device_unchecked(self, &indices, &update, &axes, stream) };
            *self = result;
        } else {
            *self = update;
        }
    }
}

impl<A, Val> IndexMutOp<A, Val> for Array
where
    A: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: A, val: Val, stream: impl AsRef<Stream>) {
        let operations = [i.index_op()];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, Val> IndexMutOp<(A,), Val> for Array
where
    A: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, (i,): (A,), val: Val, stream: impl AsRef<Stream>) {
        let operations = [i.index_op()];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, Val> IndexMutOp<(A, B), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B), val: Val, stream: impl AsRef<Stream>) {
        let operations = [i.0.index_op(), i.1.index_op()];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, Val> IndexMutOp<(A, B, C), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B, C), val: Val, stream: impl AsRef<Stream>) {
        let operations = [i.0.index_op(), i.1.index_op(), i.2.index_op()];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, Val> IndexMutOp<(A, B, C, D), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B, C, D), val: Val, stream: impl AsRef<Stream>) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, Val> IndexMutOp<(A, B, C, D, E), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B, C, D, E), val: Val, stream: impl AsRef<Stream>) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, Val> IndexMutOp<(A, B, C, D, E, F), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B, C, D, E, F), val: Val, stream: impl AsRef<Stream>) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, Val> IndexMutOp<(A, B, C, D, E, F, G), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(&mut self, i: (A, B, C, D, E, F, G), val: Val, stream: impl AsRef<Stream>) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, Val> IndexMutOp<(A, B, C, D, E, F, G, H), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, Val> IndexMutOp<(A, B, C, D, E, F, G, H, I), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, Val> IndexMutOp<(A, B, C, D, E, F, G, H, I, J), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, Val> IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K), Val>
    for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, Val> IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L), Val>
    for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    L: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, M, Val>
    IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    L: ArrayIndex,
    M: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, Val>
    IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    L: ArrayIndex,
    M: ArrayIndex,
    N: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, Val>
    IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    L: ArrayIndex,
    M: ArrayIndex,
    N: ArrayIndex,
    O: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
            i.14.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Val>
    IndexMutOp<(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), Val> for Array
where
    A: ArrayIndex,
    B: ArrayIndex,
    C: ArrayIndex,
    D: ArrayIndex,
    E: ArrayIndex,
    F: ArrayIndex,
    G: ArrayIndex,
    H: ArrayIndex,
    I: ArrayIndex,
    J: ArrayIndex,
    K: ArrayIndex,
    L: ArrayIndex,
    M: ArrayIndex,
    N: ArrayIndex,
    O: ArrayIndex,
    P: ArrayIndex,
    Val: AsRef<Array>,
{
    fn index_mut_device(
        &mut self,
        i: (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
        val: Val,
        stream: impl AsRef<Stream>,
    ) {
        let operations = [
            i.0.index_op(),
            i.1.index_op(),
            i.2.index_op(),
            i.3.index_op(),
            i.4.index_op(),
            i.5.index_op(),
            i.6.index_op(),
            i.7.index_op(),
            i.8.index_op(),
            i.9.index_op(),
            i.10.index_op(),
            i.11.index_op(),
            i.12.index_op(),
            i.13.index_op(),
            i.14.index_op(),
            i.15.index_op(),
        ];
        let update = val.as_ref();
        self.index_mut_device_inner(&operations, update, stream);
    }
}

/// The unit tests below are adapted from the Swift binding tests
#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{
        ops::indexing::{ArrayIndex, IndexOp},
        prelude::*,
    };

    #[test]
    fn test_array_mutate_single_index() {
        let mut a = Array::from_iter(0i32..12, &[3, 4]);
        let new_value = Array::from_int(77);
        a.index_mut(1, new_value);

        let expected = Array::from_slice(&[0, 1, 2, 3, 77, 77, 77, 77, 8, 9, 10, 11], &[3, 4]);
        assert_eq!(a, expected);
    }

    #[test]
    fn test_array_mutate_broadcast_multi_index() {
        let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);

        // broadcast to a row
        a.index_mut((1, 0), Array::from_int(77));

        // assign to a row
        a.index_mut((0, 0), Array::from_slice(&[55i32, 66, 77, 88, 99], &[5]));

        // single element
        a.index_mut((0, 1, 3), Array::from_int(123));

        let expected = Array::from_slice(
            &[
                55, 66, 77, 88, 99, 5, 6, 7, 123, 9, 77, 77, 77, 77, 77, 15, 16, 17, 18, 19,
            ],
            &[2, 2, 5],
        );
        assert_eq!(a, expected);
    }

    #[test]
    fn test_array_mutate_broadcast_slice() {
        let mut a = Array::from_iter(0i32..20, &[2, 2, 5]);

        // writing using slices -- this ends up covering two elements
        a.index_mut((0..1, 1..2, 2..4), Array::from_int(88));

        let expected = Array::from_slice(
            &[
                0, 1, 2, 3, 4, 5, 6, 88, 88, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
            &[2, 2, 5],
        );
        assert_eq!(a, expected);
    }

    #[test]
    fn test_array_mutate_advanced() {
        let mut a = Array::from_iter(0i32..35, &[5, 7]);

        let i1 = Array::from_slice(&[0, 2, 4], &[3]);
        let i2 = Array::from_slice(&[0, 1, 2], &[3]);

        a.index_mut((i1, i2), Array::from_slice(&[100, 200, 300], &[3]));

        assert_eq!(a.index((0, 0)).item::<i32>(), 100i32);
        assert_eq!(a.index((2, 1)).item::<i32>(), 200i32);
        assert_eq!(a.index((4, 2)).item::<i32>(), 300i32);
    }

    #[test]
    fn test_full_index_write_single() {
        fn check(index: impl ArrayIndex, expected_sum: i32) {
            let mut a = Array::from_iter(0..60, &[3, 4, 5]);

            a.index_mut(index, Array::from_int(1));
            let sum = a.sum(None, None).unwrap().item::<i32>();
            assert_eq!(sum, expected_sum);
        }

        // a[...]
        // not valid

        // a[None]
        check(NewAxis, 60);

        // a[0]
        check(0, 1600);

        // a[1:3]
        check(1..3, 230);

        // i = mx.array([2, 1])
        let i = Array::from_slice(&[2, 1], &[2]);

        // a[i]
        check(i, 230);
    }

    #[test]
    fn test_full_index_write_no_array() {
        macro_rules! check {
            (($( $i:expr ),*), $sum:expr ) => {
                {
                    let mut a = Array::from_iter(0..360, &[2, 3, 4, 5, 3]);

                    a.index_mut(($($i),*), Array::from_int(1));
                    let sum = a.sum(None, None).unwrap().item::<i32>();
                    assert_eq!(sum, $sum);
                }
            };
        }

        // a[..., 0] = 1
        check!((Ellipsis, 0), 43320);

        // a[0, ...] = 1
        check!((0, Ellipsis), 48690);

        // a[0, ..., 0] = 1
        check!((0, Ellipsis, 0), 59370);

        // a[..., ::2, :] = 1
        check!((Ellipsis, (..).stride_by(2), ..), 26064);

        // a[..., None, ::2, -1]
        check!((Ellipsis, NewAxis, (..).stride_by(2), -1), 51696);

        // a[:, 2:, 0] = 1
        check!((.., 2.., 0), 58140);

        // a[::-1, :2, 2:, ..., None, ::2] = 1
        check!(
            (
                (..).stride_by(-1),
                ..2,
                2..,
                Ellipsis,
                NewAxis,
                (..).stride_by(2)
            ),
            51540
        );
    }

    #[test]
    fn test_full_index_write_array() {
        // these have an Array as a source of indices and go through the gather path

        macro_rules! check {
            (($( $i:expr ),*), $sum:expr ) => {
                {
                    let mut a = Array::from_iter(0..540, &[3, 3, 4, 5, 3]);

                    a.index_mut(($($i),*), Array::from_int(1));
                    let sum = a.sum(None, None).unwrap().item::<i32>();
                    assert_eq!(sum, $sum);
                }
            };
        }

        // i = mx.array([2, 1])
        let i = Rc::new(Array::from_slice(&[2, 1], &[2]));

        // a[0, i] = 1
        check!((0, i.clone()), 131310);

        // a[..., i, 0] = 1
        check!((Ellipsis, i.clone(), 0), 126378);

        // a[i, 0, ...] = 1
        check!((i.clone(), 0, Ellipsis), 109710);

        // a[i, ..., i] = 1
        check!((i.clone(), Ellipsis, i.clone()), 102450);

        // a[i, ..., ::2, :] = 1
        check!((i.clone(), Ellipsis, (..).stride_by(2), ..), 68094);

        // a[..., i, None, ::2, -1] = 1
        check!(
            (Ellipsis, i.clone(), NewAxis, (..).stride_by(2), -1),
            130977
        );

        // a[:, 2:, i] = 1
        check!((.., 2.., i.clone()), 115965);

        // a[::-1, :2, i, 2:, ..., None, ::2] = 1
        check!(
            (
                (..).stride_by(-1),
                ..2,
                i,
                2..,
                Ellipsis,
                NewAxis,
                (..).stride_by(2)
            ),
            128142
        );
    }
}
