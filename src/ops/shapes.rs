use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    error::{DuplicateAxisError, ExpandDimsError, InvalidAxisError, ReshapeError},
    utils::{all_unique, resolve_index},
    Array, StreamOrDevice,
};

#[default_device]
pub unsafe fn expand_dims_device_unchecked(
    a: &Array,
    axes: &[i32],
    stream: StreamOrDevice,
) -> Array {
    unsafe {
        let c_array =
            mlx_sys::mlx_expand_dims(a.c_array, axes.as_ptr(), axes.len(), stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn try_expand_dims_device(
    a: &Array,
    axes: &[i32],
    stream: StreamOrDevice,
) -> Result<Array, ExpandDimsError> {
    // Check for valid axes
    // TODO: what is a good default capacity for SmallVec?
    let out_ndim = a.size() + axes.len();
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

    unsafe { Ok(expand_dims_device_unchecked(a, &out_axes, stream)) }
}

#[default_device]
pub fn expand_dims_device(a: &Array, axes: &[i32], stream: StreamOrDevice) -> Array {
    try_expand_dims_device(a, axes, stream).unwrap()
}

#[default_device]
pub unsafe fn reshape_device_unchecked(a: &Array, shape: &[i32], stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_reshape(a.c_array, shape.as_ptr(), shape.len(), stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn try_reshape_device<'a>(
    a: &Array,
    shape: &'a [i32],
    stream: StreamOrDevice,
) -> Result<Array, ReshapeError<'a>> {
    a.can_reshape_to(shape)?;
    unsafe { Ok(reshape_device_unchecked(a, shape, stream)) }
}

#[default_device]
pub fn reshape_device(a: &Array, shape: &[i32], stream: StreamOrDevice) -> Array {
    try_reshape_device(a, shape, stream).unwrap()
}

// Also provide reshape as a method on Array
impl Array {
    #[default_device]
    pub fn reshape_device_unchecked(&self, shape: &[i32], stream: StreamOrDevice) -> Array {
        unsafe { reshape_device_unchecked(self, shape, stream) }
    }

    #[default_device]
    pub fn try_reshape_device<'a>(
        &self,
        shape: &'a [i32],
        stream: StreamOrDevice,
    ) -> Result<Array, ReshapeError<'a>> {
        try_reshape_device(self, shape, stream)
    }

    #[default_device]
    pub fn reshape_device(&self, shape: &[i32], stream: StreamOrDevice) -> Array {
        reshape_device(self, shape, stream)
    }
}