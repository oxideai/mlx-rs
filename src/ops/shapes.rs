use std::{collections::HashSet, hash::Hash};

use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    error::{
        DuplicateAxisError, ExpandDimsError, FlattenError, InvalidAxisError, ReshapeError,
        SqueezeError,
    },
    utils::{all_unique, axes_or_default_to_all_size_one_axes, resolve_index},
    Array, StreamOrDevice,
};

impl Array {
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

    #[default_device]
    pub fn flatten_device_unchecked(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Array {
        let start_axis = start_axis.into().unwrap_or(0);
        let end_axis = end_axis.into().unwrap_or(-1);

        unsafe {
            let c_array = mlx_sys::mlx_flatten(self.c_array, start_axis, end_axis, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_flatten_device(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Result<Array, FlattenError> {
        let ndim = self.ndim();
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

        unsafe {
            let c_array = mlx_sys::mlx_flatten(self.c_array, start_axis, end_axis, stream.as_ptr());
            Ok(Array::from_ptr(c_array))
        }
    }

    #[default_device]
    pub fn flatten_device(
        &self,
        start_axis: impl Into<Option<i32>>,
        end_axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_flatten_device(start_axis, end_axis, stream)
            .unwrap()
    }

    #[default_device]
    pub unsafe fn reshape_device_unchecked(&self, shape: &[i32], stream: StreamOrDevice) -> Array {
        unsafe {
            let c_array =
                mlx_sys::mlx_reshape(self.c_array, shape.as_ptr(), shape.len(), stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_reshape_device<'a>(
        &self,
        shape: &'a [i32],
        stream: StreamOrDevice,
    ) -> Result<Array, ReshapeError<'a>> {
        self.can_reshape_to(shape)?;
        unsafe { Ok(self.reshape_device_unchecked(shape, stream)) }
    }

    #[default_device]
    pub fn reshape_device(&self, shape: &[i32], stream: StreamOrDevice) -> Array {
        self.try_reshape_device(shape, stream).unwrap()
    }

    #[default_device]
    pub fn squeeze_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Array {
        // All size 1 axes are removed if axes is None
        let axes: SmallVec<[i32; 4]> = axes_or_default_to_all_size_one_axes(axes, self.shape());
        unsafe {
            let c_array =
                mlx_sys::mlx_squeeze(self.c_array, axes.as_ptr(), axes.len(), stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_squeeze_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Result<Array, SqueezeError> {
        let axes = axes_or_default_to_all_size_one_axes(axes, self.shape());
        let mut unique_axes = HashSet::new();

        for axis in &axes {
            if *axis < 0 || *axis >= self.ndim() as i32 {
                return Err(InvalidAxisError {
                    axis: *axis,
                    ndim: self.ndim(),
                }
                .into());
            }

            let axis_size = self.shape()[*axis as usize];
            if axis_size != 1 {
                return Err(SqueezeError::AxisSizeGreaterThanOne {
                    axis: *axis,
                    size: axis_size,
                });
            }

            if !unique_axes.insert(*axis) {
                return Err(DuplicateAxisError { axis: *axis }.into());
            }
        }

        unsafe {
            let c_array =
                mlx_sys::mlx_squeeze(self.c_array, axes.as_ptr(), axes.len(), stream.as_ptr());
            Ok(Array::from_ptr(c_array))
        }
    }

    #[default_device]
    pub fn squeeze_device<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_squeeze_device(axes, stream).unwrap()
    }
}
