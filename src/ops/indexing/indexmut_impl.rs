use smallvec::{smallvec, SmallVec};

use crate::{
    error::SliceError,
    ops::{
        broadcast_to_device,
        indexing::{count_non_new_axis_operations, expand_ellipsis_operations},
    },
    utils::OwnedOrRef,
    Array, StreamOrDevice,
};

use super::ArrayIndexOp;

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

            return Some(src.slice_update_device(&update, &starts, &ends, &strides, stream.clone()));
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
            Ellipsis | TakeArray { indices: _ } => unreachable!("unexpected item in operations"),
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
