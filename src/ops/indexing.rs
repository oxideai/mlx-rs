//! Indexing Arrays
//!
//! Arrays can be indexed in the following ways:
//!
//! 1. indexing with a single integer`i32`
//! 2. indexing with a slice `&[i32]`
//! 3. indexing with an iterator `impl Iterator<Item=i32>`

use mlx_macros::default_device;
use smallvec::{smallvec, SmallVec};

use crate::{
    error::{
        DuplicateAxisError, ExpandDimsError, InvalidAxisError, SliceError, TakeAlongAxisError,
        TakeError,
    },
    utils::{all_unique, resolve_index},
    Array, StreamOrDevice,
};

#[derive(Debug, Clone, Copy)]
pub struct NewAxis;

#[derive(Debug, Clone, Copy)]
pub struct StartStop {
    pub start: i32,
    pub stop: i32,
}

impl StartStop {
    pub fn new(start: i32, stop: i32) -> Self {
        Self { start, stop }
    }
}

impl From<[i32; 2]> for StartStop {
    fn from([start, stop]: [i32; 2]) -> Self {
        Self { start, stop }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StartStopStride {
    pub start: i32,
    pub stop: i32,
    pub stride: i32,
}

impl StartStopStride {
    pub fn new(start: i32, stop: i32, stride: i32) -> Self {
        Self {
            start,
            stop,
            stride,
        }
    }
}

impl From<[i32; 3]> for StartStopStride {
    fn from([start, stop, stride]: [i32; 3]) -> Self {
        Self {
            start,
            stop,
            stride,
        }
    }
}

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
        let axes = &[0];

        // SAFETY: 0 is always a valid axis
        unsafe { self.expand_dims_unchecked(axes) }
    }
}

impl IndexOp<Array> for Array {
    type Output = Array;

    fn index(&self, index: Array) -> Self::Output {
        self.take(&index, 0)
    }
}

impl<'a> IndexOp<&Array> for Array {
    type Output = Array;

    fn index(&self, index: &Array) -> Self::Output {
        self.take(index, 0)
    }
}

impl IndexOp<StartStopStride> for Array {
    type Output = Array;

    fn index(&self, index: StartStopStride) -> Self::Output {
        let ndim = self.ndim();

        // TODO: what is a good value for SmallVec capacity?
        let start: SmallVec<[i32; 4]> = smallvec![index.start; ndim];
        let stop: SmallVec<[i32; 4]> = smallvec![index.stop; ndim];
        let strides: SmallVec<[i32; 4]> = smallvec![index.stride; ndim];

        self.slice(&start, &stop, &strides)
    }
}

impl<'a> IndexOp<&StartStopStride> for Array {
    type Output = Array;

    fn index(&self, index: &StartStopStride) -> Self::Output {
        self.index(*index)
    }
}

impl IndexOp<StartStop> for Array {
    type Output = Array;

    fn index(&self, index: StartStop) -> Self::Output {
        let index = StartStopStride {
            start: index.start,
            stop: index.stop,
            stride: 1,
        };

        self.index(index)
    }
}

impl<'a> IndexOp<&StartStop> for Array {
    type Output = Array;

    fn index(&self, index: &StartStop) -> Self::Output {
        self.index(*index)
    }
}

#[cfg(test)]
mod tests {}
