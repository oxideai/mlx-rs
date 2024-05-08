use std::{borrow::Cow, collections::HashSet, os::raw::c_void};

use mlx_macros::default_device;
use smallvec::SmallVec;

use crate::{
    error::{
        BroadcastError, ConcatenateError, ExpandDimsError, FlattenError, InvalidAxisError,
        PadError, ReshapeError, SqueezeError, StackAllError, StackError, TransposeError,
    },
    utils::{
        all_unique, is_broadcastable, is_same_shape, mlx_vector_array_values, new_mlx_vector_array,
        resolve_index,
    },
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
            let resolved_axis = resolve_index(*axis, out_ndim).ok_or_else(|| InvalidAxisError {
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

        unsafe { Ok(self.expand_dims_device_unchecked(&out_axes, stream)) }
    }

    #[default_device]
    pub fn expand_dims_device(&self, axes: &[i32], stream: StreamOrDevice) -> Array {
        self.try_expand_dims_device(axes, stream).unwrap()
    }

    #[default_device]
    pub unsafe fn flatten_device_unchecked(
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
    pub unsafe fn squeeze_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Array {
        // All size 1 axes are removed if axes is None
        let axes = axes_or_default_to_all_size_one_axes(axes, self.shape());
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

        for axis in axes.iter() {
            let resolved_axis = if axis.is_negative() {
                axis + self.ndim() as i32
            } else {
                *axis
            };
            if resolved_axis < 0 || resolved_axis >= self.ndim() as i32 {
                return Err(InvalidAxisError {
                    axis: resolved_axis,
                    ndim: self.ndim(),
                }
                .into());
            }

            let axis_size = self.shape()[*axis as usize];
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

    #[default_device]
    pub fn as_strided_device<'a>(
        &self,
        shape: impl Into<Option<&'a [i32]>>,
        strides: impl Into<Option<&'a [usize]>>,
        offset: impl Into<Option<usize>>,
        stream: StreamOrDevice,
    ) -> Array {
        let shape = shape.into().unwrap_or(self.shape());
        let resolved_strides = resolve_strides(shape, strides.into());
        let offset = offset.into().unwrap_or(0);

        unsafe {
            let c_array = mlx_sys::mlx_as_strided(
                self.c_array,
                shape.as_ptr(),
                shape.len(),
                resolved_strides.as_ptr(),
                resolved_strides.len(),
                offset,
                stream.as_ptr(),
            );
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn at_least_1d_device(&self, stream: StreamOrDevice) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_atleast_1d(self.c_array, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn at_least_2d_device(&self, stream: StreamOrDevice) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_atleast_2d(self.c_array, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn at_least_3d_device(&self, stream: StreamOrDevice) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_atleast_3d(self.c_array, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub unsafe fn move_axis_device_unchecked(
        &self,
        src: i32,
        dst: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_moveaxis(self.c_array, src, dst, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_move_axis_device(
        &self,
        src: i32,
        dst: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, InvalidAxisError> {
        let ndim = self.ndim();
        let src =
            resolve_index(src, ndim).ok_or_else(|| InvalidAxisError { axis: src, ndim })? as i32;
        let dst =
            resolve_index(dst, ndim).ok_or_else(|| InvalidAxisError { axis: dst, ndim })? as i32;

        unsafe { Ok(self.move_axis_device_unchecked(src, dst, stream)) }
    }

    #[default_device]
    pub fn move_axis_device(&self, src: i32, dst: i32, stream: StreamOrDevice) -> Array {
        self.try_move_axis_device(src, dst, stream).unwrap()
    }

    #[default_device]
    pub unsafe fn split_device_unchecked(
        &self,
        indices: &[i32],
        axis: i32,
        stream: StreamOrDevice,
    ) -> Vec<Array> {
        unsafe {
            let c_vec_arrays = mlx_sys::mlx_split(
                self.c_array,
                indices.as_ptr(),
                indices.len(),
                axis,
                stream.as_ptr(),
            );
            let len = mlx_sys::mlx_vector_array_size(c_vec_arrays);

            let mut arrays = Vec::with_capacity(len);
            for i in 0..len {
                let c_array = mlx_sys::mlx_vector_array_get(c_vec_arrays, i);
                arrays.push(Array::from_ptr(c_array));
            }
            arrays
        }
    }

    #[default_device]
    pub fn try_split_device(
        &self,
        indices: &[i32],
        axis: i32,
        stream: StreamOrDevice,
    ) -> Result<Vec<Array>, InvalidAxisError> {
        let ndim = self.ndim();
        let resolved_axis =
            resolve_index(axis, ndim).ok_or_else(|| InvalidAxisError { axis, ndim })? as i32;

        unsafe { Ok(self.split_device_unchecked(indices, resolved_axis, stream)) }
    }

    #[default_device]
    pub fn split_device(&self, indices: &[i32], axis: i32, stream: StreamOrDevice) -> Vec<Array> {
        self.try_split_device(indices, axis, stream).unwrap()
    }

    #[default_device]
    pub unsafe fn split_equal_device_unchecked(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Vec<Array> {
        let axis = axis.into().unwrap_or(0);

        unsafe {
            let c_vec =
                mlx_sys::mlx_split_equal_parts(self.c_array, num_parts, axis, stream.as_ptr());
            let result = mlx_vector_array_values(c_vec);

            mlx_sys::mlx_free(c_vec as *mut c_void);

            result
        }
    }

    #[default_device]
    pub fn try_split_equal_device(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Result<Vec<Array>, InvalidAxisError> {
        let ndim = self.ndim();
        let axis = axis.into().unwrap_or(0);
        let resolved_axis =
            resolve_index(axis, ndim).ok_or_else(|| InvalidAxisError { axis, ndim })? as i32;

        // Check if the array can be split into equal parts
        let size = self.shape()[resolved_axis as usize] as usize;
        if num_parts.is_negative() || size % num_parts as usize != 0 {
            return Err(InvalidAxisError {
                axis: num_parts,
                ndim: size,
            });
        }

        unsafe { Ok(self.split_equal_device_unchecked(num_parts, resolved_axis, stream)) }
    }

    #[default_device]
    pub fn split_equal_device(
        &self,
        num_parts: i32,
        axis: impl Into<Option<i32>>,
        stream: StreamOrDevice,
    ) -> Vec<Array> {
        self.try_split_equal_device(num_parts, axis, stream)
            .unwrap()
    }

    #[default_device]
    pub unsafe fn swap_axes_device_unchecked(
        &self,
        axis1: i32,
        axis2: i32,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = mlx_sys::mlx_swapaxes(self.c_array, axis1, axis2, stream.as_ptr());
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_swap_axes_device(
        &self,
        axis1: i32,
        axis2: i32,
        stream: StreamOrDevice,
    ) -> Result<Array, InvalidAxisError> {
        let ndim = self.ndim();
        let resolved_axis1 = resolve_index(axis1, ndim)
            .ok_or_else(|| InvalidAxisError { axis: axis1, ndim })?
            as i32;
        let resolved_axis2 = resolve_index(axis2, ndim)
            .ok_or_else(|| InvalidAxisError { axis: axis2, ndim })?
            as i32;

        unsafe { Ok(self.swap_axes_device_unchecked(resolved_axis1, resolved_axis2, stream)) }
    }

    #[default_device]
    pub fn swap_axes_device(&self, axis1: i32, axis2: i32, stream: StreamOrDevice) -> Array {
        self.try_swap_axes_device(axis1, axis2, stream).unwrap()
    }

    #[default_device]
    pub unsafe fn transpose_device_unchecked<'a>(
        &'a self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Array {
        unsafe {
            let c_array = match axes.into() {
                Some(axes) => {
                    mlx_sys::mlx_transpose(self.c_array, axes.as_ptr(), axes.len(), stream.as_ptr())
                }
                None => mlx_sys::mlx_transpose_all(self.c_array, stream.as_ptr()),
            };
            Array::from_ptr(c_array)
        }
    }

    #[default_device]
    pub fn try_transpose_device<'a>(
        &self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Result<Array, TransposeError> {
        unsafe {
            let c_array = match axes.into() {
                Some(axes) => {
                    if axes.len() != self.ndim() {
                        return Err(TransposeError::InvalidArgument {
                            num_axes: axes.len(),
                            ndim: self.ndim(),
                        });
                    }

                    let mut unique_axes = HashSet::new();
                    for axis in axes.iter() {
                        let resolved_axis = if axis.is_negative() {
                            axis + self.ndim() as i32
                        } else {
                            *axis
                        };
                        if resolved_axis < 0 || resolved_axis >= self.ndim() as i32 {
                            return Err(InvalidAxisError {
                                axis: *axis,
                                ndim: self.ndim(),
                            }
                            .into());
                        }

                        if !unique_axes.insert(resolved_axis) {
                            return Err(TransposeError::DuplicateAxis);
                        }
                    }

                    mlx_sys::mlx_transpose(self.c_array, axes.as_ptr(), axes.len(), stream.as_ptr())
                }
                None => mlx_sys::mlx_transpose_all(self.c_array, stream.as_ptr()),
            };

            Ok(Array::from_ptr(c_array))
        }
    }

    #[default_device]
    pub fn transpose_device<'a>(
        &self,
        axes: impl Into<Option<&'a [i32]>>,
        stream: StreamOrDevice,
    ) -> Array {
        self.try_transpose_device(axes, stream).unwrap()
    }

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
    match strides.into() {
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

#[default_device]
pub fn as_strided_device<'a>(
    a: &'a Array,
    shape: impl Into<Option<&'a [i32]>>,
    strides: impl Into<Option<&'a [usize]>>,
    offset: impl Into<Option<usize>>,
    stream: StreamOrDevice,
) -> Array {
    a.as_strided_device(shape, strides, offset, stream)
}

#[default_device]
pub unsafe fn broadcast_device_unchecked(
    a: &Array,
    shape: &[i32],
    stream: StreamOrDevice,
) -> Array {
    unsafe {
        let c_array =
            mlx_sys::mlx_broadcast_to(a.c_array, shape.as_ptr(), shape.len(), stream.as_ptr());
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn try_broadcast_device<'a>(
    a: &'a Array,
    shape: &'a [i32],
    stream: StreamOrDevice,
) -> Result<Array, BroadcastError<'a>> {
    if !is_broadcastable(a.shape(), shape) {
        return Err(BroadcastError {
            src_shape: a.shape(),
            dst_shape: shape,
        });
    }
    unsafe { Ok(broadcast_device_unchecked(a, shape, stream)) }
}

#[default_device]
pub fn broadcast_device<'a>(a: &'a Array, shape: &'a [i32], stream: StreamOrDevice) -> Array {
    try_broadcast_device(a, shape, stream).unwrap()
}

#[default_device]
pub unsafe fn concatenate_device_unchecked(
    arrays: &[Array],
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    let axis = axis.into().unwrap_or(0);

    unsafe {
        let c_arrays = new_mlx_vector_array(arrays);
        let c_array = mlx_sys::mlx_concatenate(c_arrays, axis, stream.as_ptr());

        let result = Array::from_ptr(c_array);
        mlx_sys::mlx_free(c_arrays as *mut c_void);

        result
    }
}

#[default_device]
pub fn try_concatenate_device(
    arrays: &[Array],
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Result<Array, ConcatenateError> {
    let axis = axis.into().unwrap_or(0);

    if arrays.is_empty() {
        return Err(ConcatenateError::NoInputArray);
    }

    // let resolved_axis = if axis.is_negative() {
    //     axis + arrays[0].ndim() as i32
    // } else {
    //     axis
    // };
    // if resolved_axis < 0 || resolved_axis >= arrays[0].ndim() as i32 {
    //     return Err(InvalidAxisError {
    //         axis,
    //         ndim: arrays[0].ndim(),
    //     }
    //     .into());
    // }
    let resolved_axis = resolve_index(axis, arrays[0].ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: arrays[0].ndim(),
    })? as i32;

    // validate shapes
    let shape = arrays[0].shape();
    for array in arrays[1..].iter() {
        if array.ndim() != shape.len() {
            return Err(ConcatenateError::InvalidAxis(InvalidAxisError {
                axis,
                ndim: array.ndim(),
            }));
        }

        for (i, axis_shape) in array.shape().iter().enumerate() {
            if i as i32 == resolved_axis {
                continue;
            }

            if axis_shape != &shape[i] {
                return Err(ConcatenateError::InvalidAxis(InvalidAxisError {
                    axis,
                    ndim: array.ndim(),
                }));
            }
        }
    }

    Ok(unsafe { concatenate_device_unchecked(arrays, axis, stream) })
}

#[default_device]
pub fn concatenate_device(
    arrays: &[Array],
    axis: impl Into<Option<i32>>,
    stream: StreamOrDevice,
) -> Array {
    try_concatenate_device(arrays, axis, stream).unwrap()
}

#[default_device]
pub unsafe fn expand_dims_device_unchecked(
    a: &Array,
    axes: &[i32],
    stream: StreamOrDevice,
) -> Array {
    a.expand_dims_device_unchecked(axes, stream)
}

#[default_device]
pub fn try_expand_dims_device(
    a: &Array,
    axes: &[i32],
    stream: StreamOrDevice,
) -> Result<Array, ExpandDimsError> {
    a.try_expand_dims_device(axes, stream)
}

#[default_device]
pub fn expand_dims_device(a: &Array, axes: &[i32], stream: StreamOrDevice) -> Array {
    a.expand_dims_device(axes, stream)
}

#[default_device]
pub unsafe fn move_axis_device_unchecked(
    a: &Array,
    src: i32,
    dst: i32,
    stream: StreamOrDevice,
) -> Array {
    a.move_axis_device_unchecked(src, dst, stream)
}

#[default_device]
pub fn try_move_axis_device(
    a: &Array,
    src: i32,
    dst: i32,
    stream: StreamOrDevice,
) -> Result<Array, InvalidAxisError> {
    a.try_move_axis_device(src, dst, stream)
}

#[default_device]
pub fn move_axis_device(a: &Array, src: i32, dst: i32, stream: StreamOrDevice) -> Array {
    a.move_axis_device(src, dst, stream)
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

#[default_device]
pub unsafe fn pad_device_unchecked<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: StreamOrDevice,
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
            stream.as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn try_pad_device<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: StreamOrDevice,
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
            stream.as_ptr(),
        );
        Ok(Array::from_ptr(c_array))
    }
}

#[default_device]
pub fn pad_device<'a>(
    array: &'a Array,
    width: impl Into<PadWidth<'a>>,
    value: impl Into<Option<Array>>,
    stream: StreamOrDevice,
) -> Array {
    try_pad_device(array, width, value, stream).unwrap()
}

#[default_device]
pub unsafe fn split_device_unchecked(
    a: &Array,
    indices: &[i32],
    axis: i32,
    stream: StreamOrDevice,
) -> Vec<Array> {
    a.split_device_unchecked(indices, axis, stream)
}

#[default_device]
pub fn try_split_device(
    a: &Array,
    indices: &[i32],
    axis: i32,
    stream: StreamOrDevice,
) -> Result<Vec<Array>, InvalidAxisError> {
    a.try_split_device(indices, axis, stream)
}

#[default_device]
pub fn split_device(a: &Array, indices: &[i32], axis: i32, stream: StreamOrDevice) -> Vec<Array> {
    a.split_device(indices, axis, stream)
}

#[default_device]
pub unsafe fn stack_device_unchecked(arrays: &[Array], axis: i32, stream: StreamOrDevice) -> Array {
    unsafe {
        let c_arrays = new_mlx_vector_array(arrays);
        let c_array = mlx_sys::mlx_stack(c_arrays, axis, stream.as_ptr());

        let result = Array::from_ptr(c_array);
        mlx_sys::mlx_free(c_arrays as *mut c_void);

        result
    }
}

#[default_device]
pub fn try_stack_device(
    arrays: &[Array],
    axis: i32,
    stream: StreamOrDevice,
) -> Result<Array, StackError> {
    if arrays.is_empty() {
        return Err(StackError::NoInputArray);
    }

    resolve_index(axis, arrays[0].ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: arrays[0].ndim(),
    })?;

    if !is_same_shape(arrays) {
        return Err(StackError::InvalidShapes);
    }

    unsafe { Ok(stack_device_unchecked(arrays, axis, stream)) }
}

#[default_device]
pub fn stack_device(arrays: &[Array], axis: i32, stream: StreamOrDevice) -> Array {
    try_stack_device(arrays, axis, stream).unwrap()
}

#[default_device]
pub unsafe fn stack_all_device_unchecked(arrays: &[Array], stream: StreamOrDevice) -> Array {
    unsafe {
        let c_arrays = new_mlx_vector_array(arrays);
        let c_array = mlx_sys::mlx_stack_all(c_arrays, stream.as_ptr());

        let result = Array::from_ptr(c_array);
        mlx_sys::mlx_free(c_arrays as *mut c_void);

        result
    }
}

#[default_device]
pub fn try_stack_all_device(
    arrays: &[Array],
    stream: StreamOrDevice,
) -> Result<Array, StackAllError> {
    if arrays.is_empty() {
        return Err(StackAllError::NoInputArray);
    }

    if !is_same_shape(arrays) {
        return Err(StackAllError::InvalidShapes);
    }

    unsafe { Ok(stack_all_device_unchecked(arrays, stream)) }
}

#[default_device]
pub fn stack_all_device(arrays: &[Array], stream: StreamOrDevice) -> Array {
    try_stack_all_device(arrays, stream).unwrap()
}

#[default_device]
pub unsafe fn swap_axes_device_unchecked(
    a: &Array,
    axis1: i32,
    axis2: i32,
    stream: StreamOrDevice,
) -> Array {
    a.swap_axes_device_unchecked(axis1, axis2, stream)
}

#[default_device]
pub fn try_swap_axes_device(
    a: &Array,
    axis1: i32,
    axis2: i32,
    stream: StreamOrDevice,
) -> Result<Array, InvalidAxisError> {
    a.try_swap_axes_device(axis1, axis2, stream)
}

#[default_device]
pub fn swap_axes_device(a: &Array, axis1: i32, axis2: i32, stream: StreamOrDevice) -> Array {
    a.swap_axes_device(axis1, axis2, stream)
}

#[default_device]
pub fn tile_device(array: &Array, repetitions: &[i32], stream: StreamOrDevice) -> Array {
    unsafe {
        let c_array = mlx_sys::mlx_tile(
            array.c_array,
            repetitions.as_ptr(),
            repetitions.len(),
            stream.as_ptr(),
        );
        Array::from_ptr(c_array)
    }
}

#[default_device]
pub unsafe fn transpose_device_unchecked(
    device: &Array,
    axes: Option<&[i32]>,
    stream: StreamOrDevice,
) -> Array {
    device.transpose_device(axes, stream)
}

#[default_device]
pub fn try_transpose_device(
    device: &Array,
    axes: Option<&[i32]>,
    stream: StreamOrDevice,
) -> Result<Array, TransposeError> {
    device.try_transpose_device(axes, stream)
}

#[default_device]
pub fn transpose_device(device: &Array, axes: Option<&[i32]>, stream: StreamOrDevice) -> Array {
    device.transpose_device(axes, stream)
}
