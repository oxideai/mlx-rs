//! Indexing Arrays
//!
//! Arrays can be indexed in the following ways:
//!
//! 1. indexing with a single integer`i32`
//! 2. indexing with a slice `&[i32]`
//! 3. indexing with an iterator `impl Iterator<Item=i32>`

use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    error::{DuplicateAxisError, ExpandDimsError, InvalidAxisError, TakeAlongAxisError, TakeError},
    utils::{all_unique, resolve_index},
    Array, StreamOrDevice,
};

#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

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

        unsafe {
            Ok(self.take_device_unchecked(indices, axis, stream))
        }
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

        unsafe {
            Ok(self.take_along_axis_device_unchecked(indices, axis, stream))
        }
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
    pub unsafe fn expand_dims_device_unchecked(&self, axes: &[i32], stream: StreamOrDevice) -> Array {
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

        unsafe {
            Ok(self.expand_dims_device_unchecked(&out_axes, stream))
        }
    }

    #[default_device]
    pub fn expand_dims_device(&self, axes: &[i32], stream: StreamOrDevice) -> Array {
        self.try_expand_dims_device(axes, stream).unwrap()
    }
}

pub trait IndexOp<Idx> {
    type Output;

    fn index(&self, index: Idx) -> Self::Output;
}

impl IndexOp<i32> for Array {
    type Output = Array;

    fn index(&self, index: i32) -> Self::Output {
        let indices = index.into();
        self.take(&indices, 0)
    }
}

impl IndexOp<NewAxis> for Array {
    type Output = Array;

    fn index(&self, _: NewAxis) -> Self::Output {
        unsafe {
            let axes =  &[0];
            self.expand_dims_unchecked(axes)
        }
    }
}

#[cfg(test)]
mod tests {}
