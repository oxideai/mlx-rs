use std::{borrow::Cow, collections::HashSet};

use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    error::{
        BroadcastError, ConcatenateError, ExpandDimsError, FlattenError, InvalidAxisError,
        PadError, ReshapeError, SqueezeError, StackError, TransposeError,
    },
    utils::{all_unique, is_broadcastable, is_same_shape, resolve_index, VectorArray},
    Array, Stream, StreamOrDevice,
};

impl Array {
    /// See [`expand_dims_unchecked`].
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the axes are valid.
    #[default_device]
    pub unsafe fn expand_dims_device_unchecked(
        &self,
        axes: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { expand_dims_device_unchecked(self, axes, stream) }
    }

    /// See [`try_expand_dims`].
    #[default_device]
    pub fn try_expand_dims_device(
        &self,
        axes: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array, ExpandDimsError> {
        try_expand_dims_device(self, axes, stream)
    }

    /// See [`expand_dims`].
    #[default_device]
    pub fn expand_dims_device(&self, axes: &[i32], stream: impl AsRef<Stream>) -> Array {
        self.try_expand_dims_device(axes, stream).unwrap()
    }

    /// See [`flatten_unchecked`].
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the axes are valid.
    #[default_device]
    pub unsafe fn flatten_device_unchecked(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { flatten_device_unchecked(self, start_axis, end_axis, stream) }
    }

    /// See [`try_flatten`].
    #[default_device]
    pub fn try_flatten_device(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, FlattenError> {
        unsafe { Ok(flatten_device_unchecked(self, start_axis, end_axis, stream)) }
    }

    /// See [`flatten`].
    #[default_device]
    pub fn flatten_device(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_flatten_device(start_axis, end_axis, stream)
            .unwrap()
    }

    /// See [`reshape_unchecked`].
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the shapes are valid.
    #[default_device]
    pub unsafe fn reshape_device_unchecked(
        &self,
        shape: &[i32],
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { reshape_device_unchecked(self, shape, stream) }
    }

    /// See [`try_reshape`].
    #[default_device]
    pub fn try_reshape_device<'a>(
        &self,
        shape: &'a [i32],
        stream: impl AsRef<Stream>,
    ) -> Result<Array, ReshapeError<'a>> {
        try_reshape_device(self, shape, stream)
    }

    /// See [`reshape`].
    #[default_device]
    pub fn reshape_device(&self, shape: &[i32], stream: impl AsRef<Stream>) -> Array {
        self.try_reshape_device(shape, stream).unwrap()
    }

    /// See [`squeeze_unchecked`].
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the axes are valid.
    #[default_device]
    pub unsafe fn squeeze_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        squeeze_device_unchecked(self, axes, stream)
    }

    /// See [`try_squeeze`].
    #[default_device]
    pub fn try_squeeze_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, SqueezeError> {
        try_squeeze_device(self, axes, stream)
    }

    /// See [`squeeze`].
    #[default_device]
    pub fn squeeze_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_squeeze_device(axes, stream).unwrap()
    }

    /// See [`as_strided`]
    #[default_device]
    pub fn as_strided_device<'a>(
        &'a self,
        shape: impl Into<Option<&'a [i32]>>,
        strides: impl Into<Option<&'a [usize]>>,
        offset: impl Into<Option<usize>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        as_strided_device(self, shape, strides, offset, stream)
    }

    /// See [`at_least_1d`]
    #[default_device]
    pub fn at_least_1d_device(&self, stream: impl AsRef<Stream>) -> Array {
        at_least_1d_device(self, stream)
    }

    /// See [`at_least_2d`]
    #[default_device]
    pub fn at_least_2d_device(&self, stream: impl AsRef<Stream>) -> Array {
        at_least_2d_device(self, stream)
    }

    /// See [`at_least_3d`]
    #[default_device]
    pub fn at_least_3d_device(&self, stream: impl AsRef<Stream>) -> Array {
        at_least_3d_device(self, stream)
    }

    /// See [`move_axis_unchecked`]
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the axes are valid.
    #[default_device]
    pub unsafe fn move_axis_device_unchecked(
        &self,
        src: i32,
        dst: i32,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { move_axis_device_unchecked(self, src, dst, stream) }
    }

    /// See [`try_move_axis`]
    #[default_device]
    pub fn try_move_axis_device(
        &self,
        src: i32,
        dst: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, InvalidAxisError> {
        try_move_axis_device(self, src, dst, stream)
    }

    /// See [`move_axis`]
    #[default_device]
    pub fn move_axis_device(&self, src: i32, dst: i32, stream: impl AsRef<Stream>) -> Array {
        self.try_move_axis_device(src, dst, stream).unwrap()
    }

    /// See [`split_unchecked`]
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the indices are valid.
    #[default_device]
    pub unsafe fn split_device_unchecked(
        &self,
        indices: &[i32],
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Vec<Array> {
        split_device_unchecked(self, indices, axis, stream)
    }

    /// See [`try_split`]
    #[default_device]
    pub fn try_split_device(
        &self,
        indices: &[i32],
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>, InvalidAxisError> {
        try_split_device(self, indices, axis, stream)
    }

    /// See [`split`]
    #[default_device]
    pub fn split_device(
        &self,
        indices: &[i32],
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Vec<Array> {
        self.try_split_device(indices, axis, stream).unwrap()
    }

    /// See [`split_equal_unchecked`]
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the number of parts is valid.
    #[default_device]
    pub unsafe fn split_equal_device_unchecked(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Vec<Array> {
        unsafe { split_equal_device_unchecked(self, num_parts, axis, stream) }
    }

    /// See [`try_split_equal`]
    #[default_device]
    pub fn try_split_equal_device(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>, InvalidAxisError> {
        try_split_equal_device(self, num_parts, axis, stream)
    }

    /// See [`split_equal`]
    #[default_device]
    pub fn split_equal_device(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Vec<Array> {
        self.try_split_equal_device(num_parts, axis, stream)
            .unwrap()
    }

    /// See [`stack_unchecked`]
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the shapes are valid.
    #[default_device]
    pub unsafe fn swap_axes_device_unchecked(
        &self,
        axis1: i32,
        axis2: i32,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { swap_axes_device_unchecked(self, axis1, axis2, stream) }
    }

    /// See [`try_swap_axes`]
    #[default_device]
    pub fn try_swap_axes_device(
        &self,
        axis1: i32,
        axis2: i32,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, InvalidAxisError> {
        try_swap_axes_device(self, axis1, axis2, stream)
    }

    #[default_device]
    pub fn swap_axes_device(&self, axis1: i32, axis2: i32, stream: impl AsRef<Stream>) -> Array {
        self.try_swap_axes_device(axis1, axis2, stream).unwrap()
    }

    /// See [`transpose_unchecked`]
    ///
    /// # Safety
    ///
    /// The function is unsafe because it does not check if the axes are valid.
    #[default_device]
    pub unsafe fn transpose_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        unsafe { transpose_device_unchecked(self, axes, stream) }
    }

    /// See [`try_transpose`]
    #[default_device]
    pub fn try_transpose_device<'a>(
        &self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array, TransposeError> {
        try_transpose_device(self, axes, stream)
    }

    /// See [`transpose`]
    #[default_device]
    pub fn transpose_device<'a>(
        &self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: impl AsRef<Stream>,
    ) -> Array {
        self.try_transpose_device(axes, stream).unwrap()
    }

    /// See [`transpose`]
    pub fn t(&self) -> Array {
        self.transpose_device(None, StreamOrDevice::default())
    }
}

fn axes_or_default_to_all_size_one_axes<'a>(
    axes: impl Into<Option<&'a [i32]>>,
    shape: &[i32],
) -> Cow<'a, [i32]> {
    match axes.into() {
        Some(axes) => Cow::Borrowed(axes),
        None => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| if dim == 1 { Some(i as i32) } else { None })
            .collect(),
    }
}

fn resolve_strides(shape: &[i32], strides: Option<&[usize]>) -> SmallVec<[usize; 4]> {
    match strides {
        Some(strides) => SmallVec::from_slice(strides),
        None => {
            let result = shape
                .iter()
                .rev()
                .scan(1, |acc, &dim| {
                    let result = *acc;
                    *acc *= dim as usize;
                    Some(result)
                })
                .collect::<SmallVec<[usize; 4]>>();
            result.into_iter().rev().collect()
        }
    }
}

/// Create a view into the array with the given shape and strides.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_iter(0..10, &[10]);
/// let y = as_strided(&x, &[3, 3][..], &[1, 1][..], 0);
/// ```
#[default_device]
pub fn as_strided_device<'a>(
    a: &'a Array,
    shape: impl Into<Option<&'a [i32]>>,
    strides: impl Into<Option<&'a [usize]>>,
    offset: impl Into<Option<usize>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let shape = shape.into().unwrap_or(a.shape());
    let resolved_strides = resolve_strides(shape, strides.into());
    let offset = offset.into().unwrap_or(0);

    unsafe {
        let c_array = mlx_sys::mlx_as_strided(
            a.c_array,
            shape.as_ptr(),
            shape.len(),
            resolved_strides.as_ptr(),
            resolved_strides.len(),
            offset,
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Broadcast an array to the given shape.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: The shape to broadcast to.
///
/// # Safety
///
/// The function is unsafe because it does not check if the shapes are broadcastable.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_float(2.3);
/// let y = unsafe { broadcast_to_unchecked(&x, &[1, 1]) };
/// ```
#[default_device]
pub unsafe fn broadcast_to_device_unchecked(
    a: &Array,
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_broadcast_to(
            a.c_array,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Broadcast an array to the given shape. Returns an error if the shapes are not broadcastable.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: The shape to broadcast to.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_float(2.3);
/// let result = try_broadcast_to(&x, &[1, 1]);
/// ```
#[default_device]
pub fn try_broadcast_to_device<'a>(
    a: &'a Array,
    shape: &'a [i32],
    stream: impl AsRef<Stream>,
) -> Result<Array, BroadcastError<'a>> {
    if !is_broadcastable(a.shape(), shape) {
        return Err(BroadcastError {
            src_shape: a.shape(),
            dst_shape: shape,
        });
    }
    unsafe { Ok(broadcast_to_device_unchecked(a, shape, stream)) }
}

/// Broadcast an array to the given shape. Panics if the shapes are not broadcastable.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: The shape to broadcast to.
///
/// # Panics
///
/// Panics if the shapes are not broadcastable. See [`try_broadcast_to`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_float(2.3);
/// let y = broadcast_to(&x, &[1, 1]);
/// ```
#[default_device]
pub fn broadcast_to_device<'a>(
    a: &'a Array,
    shape: &'a [i32],
    stream: impl AsRef<Stream>,
) -> Array {
    try_broadcast_to_device(a, shape, stream).unwrap()
}

fn concatenate_inner(arrays: &[impl AsRef<Array>], axis: i32, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_arrays = VectorArray::from_iter(arrays.iter());
        let c_array = mlx_sys::mlx_concatenate(c_arrays.as_ptr(), axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Concatenate the arrays along the given axis.
///
/// # Params
///
/// - `arrays`: The arrays to concatenate.
/// - `axis`: The axis to concatenate along.
///
/// # Safety
///
/// The function is unsafe because it does not check if the shapes are valid for concatenation.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_iter(0..4, &[2, 2]);
/// let y = Array::from_iter(4..8, &[2, 2]);
/// let z = unsafe { concatenate_unchecked(&[x, y], 0) };
/// ```
#[default_device]
pub unsafe fn concatenate_device_unchecked(
    arrays: &[impl AsRef<Array>],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let axis = axis.into().unwrap_or(0);
    concatenate_inner(arrays, axis, stream)
}

/// Concatenate the arrays along the given axis. Returns an error if the shapes are invalid.
///
/// # Params
///
/// - `arrays`: The arrays to concatenate.
/// - `axis`: The axis to concatenate along.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_iter(0..4, &[2, 2]);
/// let y = Array::from_iter(4..8, &[2, 2]);
/// let result = try_concatenate(&[x, y], 0);
/// ```
#[default_device]
pub fn try_concatenate_device(
    arrays: &[impl AsRef<Array>],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, ConcatenateError> {
    let axis = axis.into().unwrap_or(0);

    if arrays.is_empty() {
        return Err(ConcatenateError::NoInputArray);
    }

    let resolved_axis =
        resolve_index(axis, arrays[0].as_ref().ndim()).ok_or_else(|| InvalidAxisError {
            axis,
            ndim: arrays[0].as_ref().ndim(),
        })? as i32;

    // validate shapes
    let shape = arrays[0].as_ref().shape();
    for array in arrays[1..].iter() {
        if array.as_ref().ndim() != shape.len() {
            return Err(ConcatenateError::InvalidAxis(InvalidAxisError {
                axis,
                ndim: array.as_ref().ndim(),
            }));
        }

        for (i, axis_shape) in array.as_ref().shape().iter().enumerate() {
            if i as i32 == resolved_axis {
                continue;
            }

            if axis_shape != &shape[i] {
                return Err(ConcatenateError::InvalidAxis(InvalidAxisError {
                    axis,
                    ndim: array.as_ref().ndim(),
                }));
            }
        }
    }

    Ok(concatenate_inner(arrays, resolved_axis, stream))
}

/// Concatenate the arrays along the given axis. Panics if the shapes are invalid.
///
/// # Params
///
/// - `arrays`: The arrays to concatenate.
/// - `axis`: The axis to concatenate along.
///
/// # Panics
///
/// Panics if the shapes are invalid. See [`try_concatenate`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_iter(0..4, &[2, 2]);
/// let y = Array::from_iter(4..8, &[2, 2]);
/// let z = concatenate(&[x, y], 0);
/// ```
#[default_device]
pub fn concatenate_device(
    arrays: &[impl AsRef<Array>],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_concatenate_device(arrays, axis, stream).unwrap()
}

/// Add a size one dimension at the given axis.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: The index of the inserted dimensions.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let y = unsafe { expand_dims_unchecked(&x, &[0]) };
/// ```
#[default_device]
pub unsafe fn expand_dims_device_unchecked(
    a: &Array,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_expand_dims(
            a.c_array,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Add a size one dimension at the given axis, returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: The index of the inserted dimensions.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let result = try_expand_dims(&x, &[0]);
/// ```
#[default_device]
pub fn try_expand_dims_device(
    a: &Array,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array, ExpandDimsError> {
    // Check for valid axes
    // TODO: what is a good default capacity for SmallVec?
    let out_ndim = a.ndim() + axes.len();
    let mut out_axes = SmallVec::<[i32; 4]>::with_capacity(out_ndim);
    for axis in axes {
        let resolved_axis = resolve_index(*axis, out_ndim).ok_or(InvalidAxisError {
            axis: *axis,
            ndim: out_ndim,
        })?;
        if resolved_axis > i32::MAX as usize {
            // TODO: return a different error type?
            return Err(InvalidAxisError {
                axis: *axis,
                ndim: out_ndim,
            }
            .into());
        }
        out_axes.push(resolved_axis as i32);
    }

    // Check for duplicate axes
    all_unique(&out_axes).map_err(|_axis| ExpandDimsError::DuplicateAxis)?;

    unsafe { Ok(expand_dims_device_unchecked(a, &out_axes, stream)) }
}

/// Add a size one dimension at the given axis.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: The index of the inserted dimensions.
///
/// # Panics
///
/// Panics if the axes are invalid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let y = expand_dims(&x, &[0]);
/// ```
#[default_device]
pub fn expand_dims_device(a: &Array, axes: &[i32], stream: impl AsRef<Stream>) -> Array {
    a.expand_dims_device(axes, stream)
}

/// Flatten an array.
///
/// The axes flattened will be between `start_axis` and `end_axis`, inclusive. Negative axes are
/// supported. After converting negative axis to positive, axes outside the valid range will be
/// clamped to a valid value, `start_axis` to `0` and `end_axis` to `ndim - 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `start_axis`: The first axis to flatten. Default is `0` if not provided.
/// - `end_axis`: The last axis to flatten. Default is `-1` if not provided.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2, 2]);
/// let y = unsafe { flatten_unchecked(&x, None, None) };
/// ```
#[default_device]
pub unsafe fn flatten_device_unchecked(
    a: &Array,
    start_axis: impl Into<Option<i32>>,
    end_axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let start_axis = start_axis.into().unwrap_or(0);
    let end_axis = end_axis.into().unwrap_or(-1);

    unsafe {
        let c_array =
            mlx_sys::mlx_flatten(a.c_array, start_axis, end_axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Flatten an array. Returns an error if the axes are invalid.
///
/// The axes flattened will be between `start_axis` and `end_axis`, inclusive. Negative axes are
/// supported. After converting negative axis to positive, axes outside the valid range will be
/// clamped to a valid value, `start_axis` to `0` and `end_axis` to `ndim - 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `start_axis`: The first axis to flatten. Default is `0` if not provided.
/// - `end_axis`: The last axis to flatten. Default is `-1` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2, 2]);
/// let y = try_flatten(&x, None, None);
/// ```
#[default_device]
pub fn try_flatten_device(
    a: &Array,
    start_axis: impl Into<Option<i32>>,
    end_axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, FlattenError> {
    let ndim = a.ndim();

    if ndim == 0 {
        return unsafe { Ok(flatten_device_unchecked(a, start_axis, end_axis, stream)) };
    }

    let mut start_axis = start_axis.into().unwrap_or(0);
    let mut end_axis = end_axis.into().unwrap_or(-1);

    if start_axis.is_negative() {
        start_axis += ndim as i32;
    }

    if end_axis.is_negative() {
        end_axis += ndim as i32;
    }

    let start_axis = start_axis.max(0);
    let end_axis = end_axis.min(ndim as i32 - 1);

    if end_axis < start_axis {
        return Err(FlattenError::StartAxisGreaterThanEndAxis {
            start: start_axis,
            end: end_axis,
        });
    }

    if start_axis >= ndim as i32 {
        return Err(FlattenError::InvalidStartAxis(InvalidAxisError {
            axis: start_axis,
            ndim,
        }));
    }

    if end_axis < 0 {
        return Err(FlattenError::InvalidStartAxis(InvalidAxisError {
            axis: end_axis,
            ndim,
        }));
    }

    unsafe { Ok(flatten_device_unchecked(a, start_axis, end_axis, stream)) }
}

/// Flatten an array.
///
/// The axes flattened will be between `start_axis` and `end_axis`, inclusive. Negative axes are
/// supported. After converting negative axis to positive, axes outside the valid range will be
/// clamped to a valid value, `start_axis` to `0` and `end_axis` to `ndim - 1`.
///
/// # Params
///
/// - `a`: The input array.
/// - `start_axis`: The first axis to flatten. Default is `0` if not provided.
/// - `end_axis`: The last axis to flatten. Default is `-1` if not provided.
///
/// # Panics
///
/// Panics if the axes are invalid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2, 2]);
/// let y = flatten(&x, None, None);
/// ```
#[default_device]
pub fn flatten_device(
    a: &Array,
    start_axis: impl Into<Option<i32>>,
    end_axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    a.flatten_device(start_axis, end_axis, stream)
}

/// Reshape an array while preserving the size.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: New shape.
///
/// # Safety
///
/// The function is unsafe because it does not check if the new shape is valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let y = unsafe { reshape_unchecked(&x, &[4]) };
/// ```
#[default_device]
pub unsafe fn reshape_device_unchecked(
    a: &Array,
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_reshape(
            a.c_array,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Reshape an array while preserving the size. Returns an error if the new shape is invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: New shape.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let result = try_reshape(&x, &[4]);
/// ```
#[default_device]
pub fn try_reshape_device<'a>(
    a: &Array,
    shape: &'a [i32],
    stream: impl AsRef<Stream>,
) -> Result<Array, ReshapeError<'a>> {
    a.can_reshape_to(shape)?;
    unsafe { Ok(reshape_device_unchecked(a, shape, stream)) }
}

/// Reshape an array while preserving the size. Panics if the new shape is invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `shape`: New shape.
///
/// # Panics
///
/// Panics if the new shape is invalid. See [`try_reshape`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[2, 2]);
/// let y = reshape(&x, &[4]);
/// ```
#[default_device]
pub fn reshape_device(a: &Array, shape: &[i32], stream: impl AsRef<Stream>) -> Array {
    a.reshape_device(shape, stream)
}

/// Remove length one axes from an array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes to remove. If `None`, all length one axes will be removed.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[1, 2, 1, 3]);
/// let y = unsafe { squeeze_unchecked(&x, None) };
/// ```
#[default_device]
pub unsafe fn squeeze_device_unchecked<'a>(
    a: &'a Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Array {
    // All size 1 axes are removed if axes is None
    let axes = axes_or_default_to_all_size_one_axes(axes, a.shape());
    unsafe {
        let c_array = mlx_sys::mlx_squeeze(
            a.c_array,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Remove length one axes from an array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes to remove. If `None`, all length one axes will be removed.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[1, 2, 1, 3]);
/// let result = try_squeeze(&x, None);
/// ```
#[default_device]
pub fn try_squeeze_device<'a>(
    a: &'a Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, SqueezeError> {
    let axes = axes_or_default_to_all_size_one_axes(axes, a.shape());
    let mut unique_axes = HashSet::new();

    for axis in axes.iter() {
        let resolved_axis = if axis.is_negative() {
            axis + a.ndim() as i32
        } else {
            *axis
        };
        if resolved_axis < 0 || resolved_axis >= a.ndim() as i32 {
            return Err(InvalidAxisError {
                axis: resolved_axis,
                ndim: a.ndim(),
            }
            .into());
        }

        let axis_size = a.shape()[resolved_axis as usize];
        if axis_size != 1 {
            return Err(SqueezeError::AxisSizeGreaterThanOne {
                axis: resolved_axis,
                size: axis_size,
            });
        }

        if !unique_axes.insert(resolved_axis) {
            return Err(SqueezeError::DuplicateAxis);
        }
    }

    unsafe {
        let c_array = mlx_sys::mlx_squeeze(
            a.c_array,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        );
        Ok(Array::from_ptr(c_array))
    }
}

/// Remove length one axes from an array. Panics if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes to remove. If `None`, all length one axes will be removed.
///
/// # Panics
///
/// Panics if the axes are invalid. See [`try_squeeze`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::zeros::<i32>(&[1, 2, 1, 3]);
/// let y = squeeze(&x, None);
/// ```
#[default_device]
pub fn squeeze_device<'a>(
    a: &'a Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Array {
    a.squeeze_device(axes, stream)
}

/// Convert array to have at least one dimension.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_1d(&x);
/// ```
#[default_device]
pub fn at_least_1d_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_atleast_1d(a.c_array, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Convert array to have at least two dimensions.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_2d(&x);
/// ```
#[default_device]
pub fn at_least_2d_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_atleast_2d(a.c_array, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Convert array to have at least three dimensions.
///
/// # Params
///
/// - `a`: The input array.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_int(1);
/// let out = at_least_3d(&x);
/// ```
#[default_device]
pub fn at_least_3d_device(a: &Array, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_atleast_3d(a.c_array, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Move an axis to a new position.
///
/// # Params
///
/// - `a`: The input array.
/// - `src`: Specifies the source axis.
/// - `dst`: Specifies the destination axis.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::zeros::<i32>(&[2, 3, 4]);
/// let result = unsafe { move_axis_unchecked(&a, 0, 2).shape() };
/// ```
#[default_device]
pub unsafe fn move_axis_device_unchecked(
    a: &Array,
    src: i32,
    dst: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_moveaxis(a.c_array, src, dst, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Move an axis to a new position. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `src`: Specifies the source axis.
/// - `dst`: Specifies the destination axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::zeros::<i32>(&[2, 3, 4]);
/// let result = try_move_axis(&a, 0, 2);
/// ```
#[default_device]
pub fn try_move_axis_device(
    a: &Array,
    src: i32,
    dst: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, InvalidAxisError> {
    let ndim = a.ndim();
    let src = resolve_index(src, ndim).ok_or(InvalidAxisError { axis: src, ndim })? as i32;
    let dst = resolve_index(dst, ndim).ok_or(InvalidAxisError { axis: dst, ndim })? as i32;

    unsafe { Ok(a.move_axis_device_unchecked(src, dst, stream)) }
}

/// Move an axis to a new position. Panics if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `src`: Specifies the source axis.
/// - `dst`: Specifies the destination axis.
///
/// # Panics
///
/// Panics if the axes are invalid. See [`try_move_axis`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::zeros::<i32>(&[2, 3, 4]);
/// let result = move_axis(&a, 0, 2);
/// ```
#[default_device]
pub fn move_axis_device(a: &Array, src: i32, dst: i32, stream: impl AsRef<Stream>) -> Array {
    a.move_axis_device(src, dst, stream)
}

/// Split an array along a given axis.
///
/// # Params
///
/// - `a`: The input array.
/// - `indices`: The indices to split at.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Safety
///
/// The function is unsafe because it does not check if the indices are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = unsafe { split_unchecked(&a, &[3, 7], 0) };
/// ```
#[default_device]
pub unsafe fn split_device_unchecked(
    a: &Array,
    indices: &[i32],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Vec<Array> {
    let axis = axis.into().unwrap_or(0);
    unsafe {
        let c_vec = VectorArray::from_op(|| {
            mlx_sys::mlx_split(
                a.c_array,
                indices.as_ptr(),
                indices.len(),
                axis,
                stream.as_ref().as_ptr(),
            )
        });
        c_vec.into_values()
    }
}

/// Split an array along a given axis. Returns an error if the indices are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `indices`: The indices to split at.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = try_split(&a, &[3, 7], 0);
/// ```
#[default_device]
pub fn try_split_device(
    a: &Array,
    indices: &[i32],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>, InvalidAxisError> {
    let axis = axis.into().unwrap_or(0);
    let ndim = a.ndim();
    let resolved_axis = resolve_index(axis, ndim).ok_or(InvalidAxisError { axis, ndim })? as i32;

    unsafe { Ok(a.split_device_unchecked(indices, resolved_axis, stream)) }
}

/// Split an array along a given axis. Panics if the indices are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `indices`: The indices to split at.
///
/// # Panics
///
/// Panics if the indices are invalid. See [`try_split`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = split(&a, &[3, 7], 0);
/// ```
#[default_device]
pub fn split_device(
    a: &Array,
    indices: &[i32],
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Vec<Array> {
    a.split_device(indices, axis, stream)
}

/// Split an array into equal parts along a given axis.
///
/// # Params
///
/// - `a`: The input array.
/// - `num_parts`: The number of parts to split into.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Safety
///
/// The function is unsafe because it does not check if the array can be split into equal parts.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = unsafe { split_equal_unchecked(&a, 2, 0) };
/// ```
#[default_device]
pub unsafe fn split_equal_device_unchecked(
    a: &Array,
    num_parts: i32,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Vec<Array> {
    let axis = axis.into().unwrap_or(0);
    let c_vec = VectorArray::from_op(|| {
        mlx_sys::mlx_split_equal_parts(a.c_array, num_parts, axis, stream.as_ref().as_ptr())
    });
    c_vec.into_values()
}

/// Split an array into equal parts along a given axis. Returns an error if the array cannot be
/// split into equal parts.
///
/// # Params
///
/// - `a`: The input array.
/// - `num_parts`: The number of parts to split into.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = try_split_equal(&a, 2, 0);
/// ```
#[default_device]
pub fn try_split_equal_device(
    a: &Array,
    num_parts: i32,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>, InvalidAxisError> {
    let ndim = a.ndim();
    let axis = axis.into().unwrap_or(0);
    let resolved_axis = resolve_index(axis, ndim).ok_or(InvalidAxisError { axis, ndim })? as i32;

    // Check if the array can be split into equal parts
    let size = a.shape()[resolved_axis as usize] as usize;
    if num_parts.is_negative() || size % num_parts as usize != 0 {
        return Err(InvalidAxisError {
            axis: num_parts,
            ndim: size,
        });
    }

    unsafe { Ok(a.split_equal_device_unchecked(num_parts, resolved_axis, stream)) }
}

/// Split an array into equal parts along a given axis. Panics if the array cannot be split into
/// equal parts.
///
/// # Params
///
/// - `a`: The input array.
/// - `num_parts`: The number of parts to split into.
/// - `axis`: The axis to split along. Default is `0` if not provided.
///
/// # Panics
///
/// Panics if the array cannot be split into equal parts. See [`try_split_equal`] for more
/// information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..10, &[10]);
/// let result = split_equal(&a, 2, 0);
/// ```
#[default_device]
pub fn split_equal_device(
    a: &Array,
    num_parts: i32,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Vec<Array> {
    a.split_equal_device(num_parts, axis, stream)
}

#[derive(Debug)]
pub enum PadWidth<'a> {
    Same((i32, i32)),
    Widths(&'a [(i32, i32)]),
}

impl From<i32> for PadWidth<'_> {
    fn from(width: i32) -> Self {
        PadWidth::Same((width, width))
    }
}

impl From<(i32, i32)> for PadWidth<'_> {
    fn from(width: (i32, i32)) -> Self {
        PadWidth::Same(width)
    }
}

impl<'a> From<&'a [(i32, i32)]> for PadWidth<'a> {
    fn from(widths: &'a [(i32, i32)]) -> Self {
        PadWidth::Widths(widths)
    }
}

impl<'a, const N: usize> From<&'a [(i32, i32); N]> for PadWidth<'a> {
    fn from(widths: &'a [(i32, i32); N]) -> Self {
        PadWidth::Widths(widths)
    }
}

impl<'a> PadWidth<'a> {
    fn low_pads(&self, ndim: usize) -> SmallVec<[i32; 4]> {
        match self {
            PadWidth::Same((low, _high)) => (0..ndim).map(|_| *low).collect(),
            PadWidth::Widths(widths) => widths.iter().map(|(low, _high)| *low).collect(),
        }
    }

    fn high_pads(&self, ndim: usize) -> SmallVec<[i32; 4]> {
        match self {
            PadWidth::Same((_low, high)) => (0..ndim).map(|_| *high).collect(),
            PadWidth::Widths(widths) => widths.iter().map(|(_low, high)| *high).collect(),
        }
    }
}

/// Pad an array with a constant value
///
/// # Params
///
/// - `array`: The input array.
/// - `width`: Number of padded values to add to the edges of each axis:`((before_1, after_1),
///   (before_2, after_2), ..., (before_N, after_N))`. If a single pair of integers is passed then
///   `(before_i, after_i)` are all the same. If a single integer or tuple with a single integer is
///   passed then all axes are extended by the same number on each side.
/// - `value`: The value to pad the array with. Default is `0` if not provided.
///
/// # Safety
///
/// The function is unsafe because it does not check if the width is valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let result = unsafe { pad_unchecked(&a, 1, Array::from_int(0)) };
/// ```
#[default_device]
pub unsafe fn pad_device_unchecked<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let width = width.into();
    let ndim = array.ndim();
    let axes: SmallVec<[i32; 4]> = (0..ndim).map(|i| i as i32).collect();
    let low_pads = width.low_pads(ndim);
    let high_pads = width.high_pads(ndim);
    let value = value
        .into()
        .unwrap_or_else(|| Array::from_int(0).as_dtype(array.dtype()));

    unsafe {
        let c_array = mlx_sys::mlx_pad(
            array.c_array,
            axes.as_ptr(),
            axes.len(),
            low_pads.as_ptr(),
            low_pads.len(),
            high_pads.as_ptr(),
            high_pads.len(),
            value.c_array,
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Pad an array with a constant value. Returns an error if the width is invalid.
///
/// # Params
///
/// - `array`: The input array.
/// - `width`: Number of padded values to add to the edges of each axis:`((before_1, after_1),
///   (before_2, after_2), ..., (before_N, after_N))`. If a single pair of integers is passed then
///   `(before_i, after_i)` are all the same. If a single integer or tuple with a single integer is
///   passed then all axes are extended by the same number on each side.
/// - `value`: The value to pad the array with. Default is `0` if not provided.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let result = try_pad(&a, 1, Array::from_int(0));
/// ```
#[default_device]
pub fn try_pad_device<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, PadError> {
    let width = width.into();
    let ndim = array.ndim();
    let axes: SmallVec<[i32; 4]> = (0..ndim).map(|i| i as i32).collect();
    let low_pads = width.low_pads(ndim);
    let high_pads = width.high_pads(ndim);
    let value = value
        .into()
        .unwrap_or_else(|| Array::from_int(0).as_dtype(array.dtype()));

    if low_pads.len() != ndim || high_pads.len() != ndim {
        return Err(PadError::InvalidWidths { axes_size: ndim });
    }

    // Check for negative padding sizes
    for (axis, (&low, &high)) in low_pads.iter().zip(high_pads.iter()).enumerate() {
        if low < 0 || high < 0 {
            return Err(PadError::NegativeWidth {
                axis,
                size: (low, high),
            });
        }
    }

    unsafe {
        let c_array = mlx_sys::mlx_pad(
            array.c_array,
            axes.as_ptr(),
            axes.len(),
            low_pads.as_ptr(),
            low_pads.len(),
            high_pads.as_ptr(),
            high_pads.len(),
            value.c_array,
            stream.as_ref().as_ptr(),
        );
        Ok(Array::from_ptr(c_array))
    }
}

/// Pad an array with a constant value. Panics if the width is invalid.
///
/// # Params
///
/// - `array`: The input array.
/// - `width`: Number of padded values to add to the edges of each axis:`((before_1, after_1),
///   (before_2, after_2), ..., (before_N, after_N))`. If a single pair of integers is passed then
///   `(before_i, after_i)` are all the same. If a single integer or tuple with a single integer is
///   passed then all axes are extended by the same number on each side.
/// - `value`: The value to pad the array with. Default is `0` if not provided.
///
/// # Panics
///
/// Panics if the width is invalid. See [`try_pad`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let result = pad(&a, 1, Array::from_int(0));
/// ```
#[default_device]
pub fn pad_device<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_pad_device(array, width, value, stream).unwrap()
}

fn stack_inner(arrays: &[impl AsRef<Array>], axis: i32, stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_vec = VectorArray::from_iter(arrays.iter());
        let c_array = mlx_sys::mlx_stack(c_vec.as_ptr(), axis, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Stacks the arrays along a new axis.
///
/// # Params
///
/// - `arrays`: The input arrays.
/// - `axis`: The axis in the result array along which the input arrays are stacked.
///
/// # Safety
///
/// The function is unsafe because it does not check if the arrays have the same shape.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = unsafe { stack_unchecked(&[&a, &b], 0) };
/// ```
#[default_device]
pub unsafe fn stack_device_unchecked(
    arrays: &[impl AsRef<Array>],
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    stack_inner(arrays, axis, stream)
}

#[default_device]
pub fn try_stack_device(
    arrays: &[impl AsRef<Array>],
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, StackError> {
    if arrays.is_empty() {
        return Err(StackError::NoInputArray);
    }

    if !is_same_shape(arrays) {
        return Err(StackError::InvalidShapes);
    }

    Ok(stack_inner(arrays, axis, stream))
}

/// Stacks the arrays along a new axis. Panics if the arrays have different shapes.
///
/// # Params
///
/// - `arrays`: The input arrays.
/// - `axis`: The axis in the result array along which the input arrays are stacked.
///
/// # Panics
///
/// Panics if the arrays have different shapes. See [`try_stack`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = stack(&[&a, &b], 0);
/// ```
#[default_device]
pub fn stack_device(arrays: &[impl AsRef<Array>], axis: i32, stream: impl AsRef<Stream>) -> Array {
    try_stack_device(arrays, axis, stream).unwrap()
}

fn stack_all_inner(arrays: &[impl AsRef<Array>], stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_vec = VectorArray::from_iter(arrays.iter());
        let c_array = mlx_sys::mlx_stack_all(c_vec.as_ptr(), stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Stacks the arrays along a new axis.
///
/// # Params
///
/// - `arrays`: The input arrays.
///
/// # Safety
///
/// The function is unsafe because it does not check if the arrays have the same shape.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = unsafe { stack_all_unchecked(&[&a, &b]) };
/// ```
#[default_device]
pub unsafe fn stack_all_device_unchecked(
    arrays: &[impl AsRef<Array>],
    stream: impl AsRef<Stream>,
) -> Array {
    stack_all_inner(arrays, stream)
}

/// Stacks the arrays along a new axis. Returns an error if the arrays have different shapes.
///
/// # Params
///
/// - `arrays`: The input arrays.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = try_stack_all(&[&a, &b]);
/// ```
#[default_device]
pub fn try_stack_all_device(
    arrays: &[impl AsRef<Array>],
    stream: impl AsRef<Stream>,
) -> Result<Array, StackError> {
    if arrays.is_empty() {
        return Err(StackError::NoInputArray);
    }

    if !is_same_shape(arrays) {
        return Err(StackError::InvalidShapes);
    }

    Ok(stack_all_inner(arrays, stream))
}

/// Stacks the arrays along a new axis. Panics if the arrays have different shapes.
///
/// # Params
///
/// - `arrays`: The input arrays.
///
/// # Panics
///
/// Panics if the arrays have different shapes. See [`try_stack_all`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..4, &[2, 2]);
/// let b = Array::from_iter(4..8, &[2, 2]);
/// let result = stack_all(&[&a, &b]);
/// ```
#[default_device]
pub fn stack_all_device(arrays: &[impl AsRef<Array>], stream: impl AsRef<Stream>) -> Array {
    try_stack_all_device(arrays, stream).unwrap()
}

/// Swap two axes of an array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis1`: The first axis.
/// - `axis2`: The second axis.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..6, &[2, 3]);
/// let result = unsafe { swap_axes_unchecked(&a, 0, 1) };
/// ```
#[default_device]
pub unsafe fn swap_axes_device_unchecked(
    a: &Array,
    axis1: i32,
    axis2: i32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_swapaxes(a.c_array, axis1, axis2, stream.as_ref().as_ptr());
        Array::from_ptr(c_array)
    }
}

/// Swap two axes of an array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis1`: The first axis.
/// - `axis2`: The second axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..6, &[2, 3]);
/// let result = try_swap_axes(&a, 0, 1);
/// ```
#[default_device]
pub fn try_swap_axes_device(
    a: &Array,
    axis1: i32,
    axis2: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array, InvalidAxisError> {
    let ndim = a.ndim();
    let resolved_axis1 =
        resolve_index(axis1, ndim).ok_or(InvalidAxisError { axis: axis1, ndim })? as i32;
    let resolved_axis2 =
        resolve_index(axis2, ndim).ok_or(InvalidAxisError { axis: axis2, ndim })? as i32;

    unsafe { Ok(a.swap_axes_device_unchecked(resolved_axis1, resolved_axis2, stream)) }
}

/// Swap two axes of an array. Panics if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axis1`: The first axis.
/// - `axis2`: The second axis.
///
/// # Panics
///
/// Panics if the axes are invalid. See [`try_swap_axes`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let a = Array::from_iter(0..6, &[2, 3]);
/// let result = swap_axes(&a, 0, 1);
/// ```
#[default_device]
pub fn swap_axes_device(a: &Array, axis1: i32, axis2: i32, stream: impl AsRef<Stream>) -> Array {
    a.swap_axes_device(axis1, axis2, stream)
}

/// Construct an array by repeating `a` the number of times given by `reps`.
///
/// # Params
///
/// - `a`: The input array.
/// - `reps`: The number of repetitions along each axis.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3], &[3]);
/// let y = tile(&x, &[2]);
/// ```
#[default_device]
pub fn tile_device(a: &Array, reps: &[i32], stream: impl AsRef<Stream>) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_tile(
            a.c_array,
            reps.as_ptr(),
            reps.len(),
            stream.as_ref().as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

/// Transpose the dimensions of the array.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Specifies the source axis for each axis in the new array. The default is to reverse
///   the axes.
///
/// # Safety
///
/// The function is unsafe because it does not check if the axes are valid.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
/// let y = transpose(&x, None);
/// ```
#[default_device]
pub unsafe fn transpose_device_unchecked<'a>(
    a: &'a Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let c_array = match axes.into() {
            Some(axes) => mlx_sys::mlx_transpose(
                a.c_array,
                axes.as_ptr(),
                axes.len(),
                stream.as_ref().as_ptr(),
            ),
            None => mlx_sys::mlx_transpose_all(a.c_array, stream.as_ref().as_ptr()),
        };
        Array::from_ptr(c_array)
    }
}

/// Transpose the dimensions of the array. Returns an error if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Specifies the source axis for each axis in the new array. The default is to reverse
///  the axes.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
/// let y = try_transpose(&x, None);
/// ```
#[default_device]
pub fn try_transpose_device<'a>(
    a: &Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, TransposeError> {
    unsafe {
        let c_array = match axes.into() {
            Some(axes) => {
                if axes.len() != a.ndim() {
                    return Err(TransposeError::InvalidArgument {
                        num_axes: axes.len(),
                        ndim: a.ndim(),
                    });
                }

                let mut unique_axes = HashSet::new();
                for axis in axes.iter() {
                    let resolved_axis = if axis.is_negative() {
                        axis + a.ndim() as i32
                    } else {
                        *axis
                    };
                    if resolved_axis < 0 || resolved_axis >= a.ndim() as i32 {
                        return Err(InvalidAxisError {
                            axis: *axis,
                            ndim: a.ndim(),
                        }
                        .into());
                    }

                    if !unique_axes.insert(resolved_axis) {
                        return Err(TransposeError::DuplicateAxis);
                    }
                }

                mlx_sys::mlx_transpose(
                    a.c_array,
                    axes.as_ptr(),
                    axes.len(),
                    stream.as_ref().as_ptr(),
                )
            }
            None => mlx_sys::mlx_transpose_all(a.c_array, stream.as_ref().as_ptr()),
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Transpose the dimensions of the array. Panics if the axes are invalid.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Specifies the source axis for each axis in the new array. The default is to reverse
/// the axes.
///
/// # Panics
///
/// Panics if the axes are invalid. See [`try_transpose`] for more information.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, ops::*};
///
/// let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
/// let y = transpose(&x, None);
/// ```
#[default_device]
pub fn transpose_device<'a>(
    a: &Array,
    axes: impl Into<Option<&'a [i32]>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_transpose_device(a, axes, stream).unwrap()
}

// The unit tests below are adapted from
// https://github.com/ml-explore/mlx/blob/main/tests/ops_tests.cpp
#[cfg(test)]
mod tests {
    use crate::{Array, Dtype};

    use super::*;

    #[test]
    fn test_squeeze() {
        let a = Array::zeros::<i32>(&[2, 1, 2, 1, 2, 1]);
        assert_eq!(squeeze(&a, &[1, 3, 5][..]).shape(), &[2, 2, 2]);
        assert_eq!(squeeze(&a, &[-1, -3, -5][..]).shape(), &[2, 2, 2]);
        assert_eq!(squeeze(&a, &[1][..]).shape(), &[2, 2, 1, 2, 1]);
        assert_eq!(squeeze(&a, &[-1][..]).shape(), &[2, 1, 2, 1, 2]);

        assert!(try_squeeze(&a, &[0][..]).is_err());
        assert!(try_squeeze(&a, &[2][..]).is_err());
        assert!(try_squeeze(&a, &[1, 3, 1][..]).is_err());
        assert!(try_squeeze(&a, &[1, 3, -3][..]).is_err());
    }

    #[test]
    fn test_expand() {
        let a = Array::zeros::<i32>(&[2, 2]);
        assert_eq!(expand_dims(&a, &[0][..]).shape(), &[1, 2, 2]);
        assert_eq!(expand_dims(&a, &[-1][..]).shape(), &[2, 2, 1]);
        assert_eq!(expand_dims(&a, &[1][..]).shape(), &[2, 1, 2]);
        assert_eq!(expand_dims(&a, &[0, 1, 2][..]).shape(), &[1, 1, 1, 2, 2]);
        assert_eq!(
            expand_dims(&a, &[0, 1, 2, 5, 6, 7][..]).shape(),
            &[1, 1, 1, 2, 2, 1, 1, 1]
        );

        assert!(try_expand_dims(&a, &[3][..]).is_err());
        assert!(try_expand_dims(&a, &[-4][..]).is_err());
        assert!(try_expand_dims(&a, &[0, 1, 0][..]).is_err());
        assert!(try_expand_dims(&a, &[0, 1, -4][..]).is_err());
    }

    #[test]
    fn test_flatten() {
        let x = Array::zeros::<i32>(&[2, 3, 4]);
        assert_eq!(flatten(&x, None, None).shape(), &[2 * 3 * 4]);

        assert_eq!(flatten(&x, 1, 1).shape(), &[2, 3, 4]);
        assert_eq!(flatten(&x, 1, 2).shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, 1, 3).shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, 1, -1).shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, -2, -1).shape(), &[2, 3 * 4]);
        assert_eq!(flatten(&x, -3, -1).shape(), &[2 * 3 * 4]);
        assert_eq!(flatten(&x, -4, -1).shape(), &[2 * 3 * 4]);

        assert!(try_flatten(&x, 2, 1).is_err());

        assert!(try_flatten(&x, 5, 6).is_err());

        assert!(try_flatten(&x, -5, -4).is_err());

        let x = Array::from_int(1);
        assert_eq!(flatten(&x, -3, -1).shape(), &[1]);
        assert_eq!(flatten(&x, 0, 0).shape(), &[1]);
    }

    #[test]
    fn test_reshape() {
        let x = Array::from_int(1);
        assert_eq!(reshape(&x, &[]).shape(), &[]);
        assert!(try_reshape(&x, &[2]).is_err());
        let y = reshape(&x, &[1, 1, 1]);
        assert_eq!(y.shape(), &[1, 1, 1]);
        let y = reshape(&x, &[-1, 1, 1]);
        assert_eq!(y.shape(), &[1, 1, 1]);
        let y = reshape(&x, &[1, 1, -1]);
        assert_eq!(y.shape(), &[1, 1, 1]);
        assert!(try_reshape(&x, &[1, -1, -1]).is_err());
        assert!(try_reshape(&x, &[2, -1]).is_err());

        let x = Array::zeros::<i32>(&[2, 2, 2]);
        let y = reshape(&x, &[8]);
        assert_eq!(y.shape(), &[8]);
        assert!(try_reshape(&x, &[7]).is_err());
        let y = reshape(&x, &[-1]);
        assert_eq!(y.shape(), &[8]);
        let y = reshape(&x, &[-1, 2]);
        assert_eq!(y.shape(), &[4, 2]);
        assert!(try_reshape(&x, &[-1, 7]).is_err());

        let x = Array::from_slice::<i32>(&[], &[0]);
        let mut y = reshape(&x, &[0, 0, 0]);
        assert_eq!(y.shape(), &[0, 0, 0]);
        y.eval();
        assert_eq!(y.size(), 0);
        assert!(try_reshape(&x, &[]).is_err());
        assert!(try_reshape(&x, &[1]).is_err());
        let y = reshape(&x, &[1, 5, 0]);
        assert_eq!(y.shape(), &[1, 5, 0]);
    }

    #[test]
    fn test_as_strided() {
        let x = Array::from_iter(0..10, &[10]);
        let y = as_strided(&x, &[3, 3][..], &[1, 1][..], 0);
        let expected = Array::from_slice(&[0, 1, 2, 1, 2, 3, 2, 3, 4], &[3, 3]);
        assert_eq!(y, expected);

        let y = as_strided(&x, &[3, 3][..], &[0, 3][..], 0);
        let expected = Array::from_slice(&[0, 3, 6, 0, 3, 6, 0, 3, 6], &[3, 3]);
        assert_eq!(y, expected);

        let x = x.reshape(&[2, 5]);
        let x = x.transpose(&[1, 0][..]);
        let y = as_strided(&x, &[3, 3][..], &[2, 1][..], 1);
        let expected = Array::from_slice(&[5, 1, 6, 6, 2, 7, 7, 3, 8], &[3, 3]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_at_least_1d() {
        let x = Array::from_int(1);
        let out = at_least_1d(&x);
        assert_eq!(out.ndim(), 1);
        assert_eq!(out.shape(), &[1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_1d(&x);
        assert_eq!(out.ndim(), 1);
        assert_eq!(out.shape(), &[3]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_1d(&x);
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn test_at_least_2d() {
        let x = Array::from_int(1);
        let out = at_least_2d(&x);
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[1, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_2d(&x);
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[1, 3]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_2d(&x);
        assert_eq!(out.ndim(), 2);
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn test_at_least_3d() {
        let x = Array::from_int(1);
        let out = at_least_3d(&x);
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[1, 1, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let out = at_least_3d(&x);
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[1, 3, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3, 1]);
        let out = at_least_3d(&x);
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.shape(), &[3, 1, 1]);
    }

    #[test]
    fn test_move_axis() {
        let a = Array::from_int(0);
        assert!(try_move_axis(&a, 0, 0).is_err());

        let a = Array::zeros::<i32>(&[2]);
        assert!(try_move_axis(&a, 0, 1).is_err());
        assert_eq!(move_axis(&a, 0, 0).shape(), &[2]);
        assert_eq!(move_axis(&a, -1, -1).shape(), &[2]);

        let a = Array::zeros::<i32>(&[2, 3, 4]);
        assert!(try_move_axis(&a, 0, -4).is_err());
        assert!(try_move_axis(&a, 0, 3).is_err());
        assert!(try_move_axis(&a, 3, 0).is_err());
        assert!(try_move_axis(&a, -4, 0).is_err());
        assert_eq!(move_axis(&a, 0, 2).shape(), &[3, 4, 2]);
        assert_eq!(move_axis(&a, 0, 1).shape(), &[3, 2, 4]);
        assert_eq!(move_axis(&a, 0, -1).shape(), &[3, 4, 2]);
        assert_eq!(move_axis(&a, -2, 2).shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_split_equal() {
        let x = Array::from_int(3);
        assert!(try_split_equal(&x, 0, 0).is_err());

        let x = Array::from_slice(&[0, 1, 2], &[3]);
        assert!(try_split_equal(&x, 3, 1).is_err());
        assert!(try_split_equal(&x, -2, 1).is_err());

        let out = split_equal(&x, 3, 0);
        assert_eq!(out.len(), 3);

        let mut out = split_equal(&x, 3, -1);
        assert_eq!(out.len(), 3);
        for (i, a) in out.iter_mut().enumerate() {
            assert_eq!(a.shape(), &[1]);
            assert_eq!(a.dtype(), Dtype::Int32);
            assert_eq!(a.item::<i32>(), i as i32);
        }

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5], &[2, 3]);
        let out = split_equal(&x, 2, None);
        assert_eq!(out[0], Array::from_slice(&[0, 1, 2], &[1, 3]));
        assert_eq!(out[1], Array::from_slice(&[3, 4, 5], &[1, 3]));

        let out = split_equal(&x, 3, 1);
        assert_eq!(out[0], Array::from_slice(&[0, 3], &[2, 1]));
        assert_eq!(out[1], Array::from_slice(&[1, 4], &[2, 1]));
        assert_eq!(out[2], Array::from_slice(&[2, 5], &[2, 1]));

        let x = Array::zeros::<i32>(&[8, 12]);
        let out = split_equal(&x, 2, None);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[4, 12]);
        assert_eq!(out[1].shape(), &[4, 12]);

        let out = split_equal(&x, 3, 1);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[8, 4]);
        assert_eq!(out[1].shape(), &[8, 4]);
        assert_eq!(out[2].shape(), &[8, 4]);
    }

    #[test]
    fn test_split() {
        let x = Array::zeros::<i32>(&[8, 12]);

        let out = split(&x, &[], None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), x.shape());

        let out = split(&x, &[3, 7], None);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[3, 12]);
        assert_eq!(out[1].shape(), &[4, 12]);
        assert_eq!(out[2].shape(), &[1, 12]);

        let out = split(&x, &[20], None);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[8, 12]);
        assert_eq!(out[1].shape(), &[0, 12]);

        let out = split(&x, &[-5], None);
        assert_eq!(out[0].shape(), &[3, 12]);
        assert_eq!(out[1].shape(), &[5, 12]);

        let out = split(&x, &[2, 8], Some(1));
        assert_eq!(out[0].shape(), &[8, 2]);
        assert_eq!(out[1].shape(), &[8, 6]);
        assert_eq!(out[2].shape(), &[8, 4]);

        let x = Array::from_iter(0i32..5, &[5]);
        let out = split(&x, &[2, 1, 2], None);
        assert_eq!(out[0], Array::from_slice(&[0, 1], &[2]));
        assert_eq!(out[1], Array::from_slice::<i32>(&[], &[0]));
        assert_eq!(out[2], Array::from_slice(&[1], &[1]));
        assert_eq!(out[3], Array::from_slice(&[2, 3, 4], &[3]));
    }

    #[test]
    fn test_pad() {
        let x = Array::zeros::<f32>(&[1, 2, 3]);
        assert_eq!(pad(&x, 1, None).shape(), &[3, 4, 5]);
        assert_eq!(pad(&x, (0, 1), None).shape(), &[2, 3, 4]);
        assert_eq!(pad(&x, &[(1, 1), (1, 2), (3, 1)], None).shape(), &[3, 5, 7]);
    }

    #[test]
    fn test_stack() {
        let x = Array::from_slice::<f32>(&[], &[0]);
        let x = vec![x];
        assert_eq!(stack(&x, 0).shape(), &[1, 0]);
        assert_eq!(stack(&x, 1).shape(), &[0, 1]);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let x = vec![x];
        assert_eq!(stack(&x, 0).shape(), &[1, 3]);
        assert_eq!(stack(&x, 1).shape(), &[3, 1]);

        let y = Array::from_slice(&[4, 5, 6], &[3]);
        let mut z = x;
        z.push(y);
        assert_eq!(stack_all(&z).shape(), &[2, 3]);
        assert_eq!(stack(&z, 1).shape(), &[3, 2]);
        assert_eq!(stack(&z, -1).shape(), &[3, 2]);
        assert_eq!(stack(&z, -2).shape(), &[2, 3]);

        let empty: Vec<Array> = Vec::new();
        assert!(try_stack(&empty, 0).is_err());

        let x = Array::from_slice(&[1, 2, 3], &[3]).as_dtype(Dtype::Float16);
        let y = Array::from_slice(&[4, 5, 6], &[3]).as_dtype(Dtype::Int32);
        assert_eq!(stack(&[x, y], 0).dtype(), Dtype::Float16);

        let x = Array::from_slice(&[1, 2, 3], &[3]).as_dtype(Dtype::Int32);
        let y = Array::from_slice(&[4, 5, 6, 7], &[4]).as_dtype(Dtype::Int32);
        assert!(try_stack(&[x, y], 0).is_err());
    }

    #[test]
    fn test_swap_axes() {
        let a = Array::from_int(0);
        assert!(try_swap_axes(&a, 0, 0).is_err());

        let a = Array::zeros::<i32>(&[2]);
        assert!(try_swap_axes(&a, 0, 1).is_err());
        assert_eq!(swap_axes(&a, 0, 0).shape(), &[2]);
        assert_eq!(swap_axes(&a, -1, -1).shape(), &[2]);

        let a = Array::zeros::<i32>(&[2, 3, 4]);
        assert!(try_swap_axes(&a, 0, -4).is_err());
        assert!(try_swap_axes(&a, 0, 3).is_err());
        assert!(try_swap_axes(&a, 3, 0).is_err());
        assert!(try_swap_axes(&a, -4, 0).is_err());
        assert_eq!(swap_axes(&a, 0, 2).shape(), &[4, 3, 2]);
        assert_eq!(swap_axes(&a, 0, 1).shape(), &[3, 2, 4]);
        assert_eq!(swap_axes(&a, 0, -1).shape(), &[4, 3, 2]);
        assert_eq!(swap_axes(&a, -2, 2).shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_tile() {
        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let y = tile(&x, &[2]);
        let expected = Array::from_slice(&[1, 2, 3, 1, 2, 3], &[6]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[2]);
        let expected = Array::from_slice(&[1, 2, 1, 2, 3, 4, 3, 4], &[2, 4]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[4, 1]);
        let expected =
            Array::from_slice(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], &[8, 2]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        let y = tile(&x, &[2, 2]);
        let expected =
            Array::from_slice(&[1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4], &[4, 4]);
        assert_eq!(y, expected);

        let x = Array::from_slice(&[1, 2, 3], &[3]);
        let y = tile(&x, &[2, 2, 2]);
        let expected = Array::from_slice(
            &[
                1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
            ],
            &[2, 2, 6],
        );
        assert_eq!(y, expected);
    }

    #[test]
    fn test_transpose() {
        let x = Array::from_int(1);
        let mut y = transpose(&x, None);
        assert_eq!(y.shape(), &[]);
        assert_eq!(y.item::<i32>(), 1);
        assert!(try_transpose(&x, &[0][..]).is_err());
        assert!(try_transpose(&x, &[1][..]).is_err());

        let x = Array::from_slice(&[1], &[1]);
        let mut y = transpose(&x, None);
        assert_eq!(y.shape(), &[1]);
        assert_eq!(y.item::<i32>(), 1);

        let mut y = transpose(&x, &[-1][..]);
        assert_eq!(y.shape(), &[1]);
        assert_eq!(y.item::<i32>(), 1);

        assert!(try_transpose(&x, &[1][..]).is_err());
        assert!(try_transpose(&x, &[0, 0][..]).is_err());

        let x = Array::from_slice::<i32>(&[], &[0]);
        let mut y = transpose(&x, None);
        assert_eq!(y.shape(), &[0]);
        y.eval();
        assert_eq!(y.size(), 0);

        let x = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let mut y = transpose(&x, None);
        assert_eq!(y.shape(), &[3, 2]);
        y = transpose(&x, &[-1, 0][..]);
        assert_eq!(y.shape(), &[3, 2]);
        y = transpose(&x, &[-1, -2][..]);
        assert_eq!(y.shape(), &[3, 2]);
        y.eval();
        assert_eq!(y, Array::from_slice(&[1, 4, 2, 5, 3, 6], &[3, 2]));

        let y = transpose(&x, &[0, 1][..]);
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y, x);

        let y = transpose(&x, &[0, -1][..]);
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y, x);

        assert!(try_transpose(&x, &[][..]).is_err());
        assert!(try_transpose(&x, &[0][..]).is_err());
        assert!(try_transpose(&x, &[0, 0][..]).is_err());
        assert!(try_transpose(&x, &[0, 0, 0][..]).is_err());
        assert!(try_transpose(&x, &[0, 1, 1][..]).is_err());

        let x = Array::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[2, 3, 2]);
        let y = transpose(&x, None);
        assert_eq!(y.shape(), &[2, 3, 2]);
        let expected = Array::from_slice(&[1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12], &[2, 3, 2]);
        assert_eq!(y, expected);

        let y = transpose(&x, &[0, 1, 2][..]);
        assert_eq!(y.shape(), &[2, 3, 2]);
        assert_eq!(y, x);

        let y = transpose(&x, &[1, 0, 2][..]);
        assert_eq!(y.shape(), &[3, 2, 2]);
        let expected = Array::from_slice(&[1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12], &[3, 2, 2]);
        assert_eq!(y, expected);

        let y = transpose(&x, &[0, 2, 1][..]);
        assert_eq!(y.shape(), &[2, 2, 3]);
        let expected = Array::from_slice(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12], &[2, 2, 3]);
        assert_eq!(y, expected);

        let mut x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[4, 2]);
        x = reshape(&transpose(&x, None), &[2, 2, 2]);
        let expected = Array::from_slice(&[0, 2, 4, 6, 1, 3, 5, 7], &[2, 2, 2]);
        assert_eq!(x, expected);

        let mut x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[1, 4, 1, 2]);
        // assert!(x.flags().row_contiguous);
        x = transpose(&x, &[2, 1, 0, 3][..]);
        x.eval();
        // assert!(x.flags().row_contiguous);
    }
}
