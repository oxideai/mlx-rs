mod fftn;
mod ifftn;
mod irfftn;
mod rfftn;

use smallvec::SmallVec;

use crate::{
    error::FftnError,
    utils::{all_unique, resolve_index},
    Array,
};

pub use self::{fftn::*, ifftn::*, irfftn::*, rfftn::*};

#[inline]
fn try_resolve_size_and_axis(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
) -> Result<(i32, i32), FftnError> {
    if a.ndim() < 1 {
        return Err(FftnError::ScalarArray);
    }

    let axis = axis.into().unwrap_or(-1);
    let axis_index =
        resolve_index(axis, a.ndim()).ok_or_else(|| FftnError::InvalidAxis { ndim: a.ndim() })?;
    let n = n.into().unwrap_or(a.shape()[axis_index]);

    Ok((n, axis))
}

// It's probably rare to perform fft on more than 4 axes
// TODO: check if this is a good default value
#[inline]
fn try_resolve_sizes_and_axes<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
) -> Result<(SmallVec<[i32; 4]>, SmallVec<[i32; 4]>), FftnError> {
    if a.ndim() < 1 {
        return Err(FftnError::ScalarArray);
    }

    let (valid_s, valid_axes) = match (s.into(), axes.into()) {
        (Some(s), Some(axes)) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (Some(s), None) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
        (None, Some(axes)) => {
            // SmallVec somehow doesn't implement FromIterator with result
            let mut valid_s = SmallVec::<[i32; 4]>::new();
            for &axis in axes {
                let axis_index = resolve_index(axis, a.ndim())
                    .ok_or_else(|| FftnError::InvalidAxis { ndim: a.ndim() })?;
                valid_s.push(a.shape()[axis_index]);
            }
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (None, None) => {
            let valid_s: SmallVec<[i32; 4]> = (0..a.ndim()).map(|axis| a.shape()[axis]).collect();
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
    };

    // Check duplicate axes
    all_unique(&valid_axes).map_err(|axis| FftnError::DuplicateAxis { axis })?;

    // Check if shape and axes have the same size
    if valid_s.len() != valid_axes.len() {
        return Err(FftnError::IncompatibleShapeAndAxes {
            shape_size: valid_s.len(),
            axes_size: valid_axes.len(),
        });
    }

    // Check if more axes are provided than the array has
    if valid_s.len() > a.ndim() {
        return Err(FftnError::InvalidAxis { ndim: a.ndim() });
    }

    // Check if output sizes are valid
    if valid_s.iter().any(|val| *val <= 0) {
        return Err(FftnError::InvalidOutputSize);
    }

    Ok((valid_s, valid_axes))
}
