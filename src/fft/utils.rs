use smallvec::SmallVec;

use crate::{
    error::{DuplicateAxisError, FftError, InvalidAxisError},
    utils::{all_unique, resolve_index, resolve_index_unchecked},
    Array,
};

#[inline]
pub(super) fn resolve_size_and_axis_unchecked(
    a: &Array,
    n: Option<i32>,
    axis: Option<i32>,
) -> (i32, i32) {
    let axis = axis.unwrap_or(-1);
    let n = n.unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    (n, axis)
}

#[inline]
pub(super) fn try_resolve_size_and_axis(
    a: &Array,
    n: Option<i32>,
    axis: Option<i32>,
) -> Result<(i32, i32), FftError> {
    if a.ndim() < 1 {
        return Err(FftError::ScalarArray);
    }

    let axis = axis.unwrap_or(-1);
    let axis_index = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
        axis,
        ndim: a.ndim(),
    })?;
    let n = n.unwrap_or(a.shape()[axis_index]);

    if n <= 0 {
        return Err(FftError::InvalidOutputSize);
    }

    Ok((n, axis))
}

// TODO: Use Cow or SmallVec?
#[inline]
pub(super) fn resolve_sizes_and_axes_unchecked<'a>(
    a: &'a Array,
    s: Option<&'a [i32]>,
    axes: Option<&'a [i32]>,
) -> (SmallVec<[i32; 4]>, SmallVec<[i32; 4]>) {
    match (s, axes) {
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
            let valid_s = axes
                .iter()
                .map(|&axis| {
                    let axis_index = resolve_index_unchecked(axis, a.ndim());
                    a.shape()[axis_index]
                })
                .collect();
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (None, None) => {
            let valid_s: SmallVec<[i32; 4]> = (0..a.ndim()).map(|axis| a.shape()[axis]).collect();
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
    }
}

// It's probably rare to perform fft on more than 4 axes
// TODO: check if this is a good default value
#[inline]
#[allow(clippy::type_complexity)]
pub(super) fn try_resolve_sizes_and_axes<'a>(
    a: &'a Array,
    s: Option<&'a [i32]>,
    axes: Option<&'a [i32]>,
) -> Result<(SmallVec<[i32; 4]>, SmallVec<[i32; 4]>), FftError> {
    if a.ndim() < 1 {
        return Err(FftError::ScalarArray);
    }

    let (valid_s, valid_axes) = match (s, axes) {
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
                let axis_index = resolve_index(axis, a.ndim()).ok_or_else(|| InvalidAxisError {
                    axis,
                    ndim: a.ndim(),
                })?;
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
    all_unique(&valid_axes).map_err(|axis| DuplicateAxisError { axis })?;

    // Check if shape and axes have the same size
    if valid_s.len() != valid_axes.len() {
        return Err(FftError::IncompatibleShapeAndAxes {
            shape_size: valid_s.len(),
            axes_size: valid_axes.len(),
        });
    }

    // Check if more axes are provided than the array has
    if valid_s.len() > a.ndim() {
        return Err(InvalidAxisError {
            axis: valid_s.len() as i32,
            ndim: a.ndim(),
        }
        .into());
    }

    // Check if output sizes are valid
    if valid_s.iter().any(|val| *val <= 0) {
        return Err(FftError::InvalidOutputSize);
    }

    Ok((valid_s, valid_axes))
}

#[cfg(test)]
mod try_resolve_size_and_axis_tests {
    use crate::Array;

    use super::{try_resolve_size_and_axis, FftError};

    #[test]
    fn scalar_array_returns_error() {
        // Returns an error if the array is a scalar
        let a = Array::from_float(1.0);
        let result = try_resolve_size_and_axis(&a, Some(0), Some(0));
        assert_eq!(result, Err(FftError::ScalarArray));
    }

    #[test]
    fn out_of_bound_axis_returns_error() {
        // Returns an error if the axis is invalid (out of bounds)
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, Some(0), Some(1));
        assert!(matches!(result, Err(FftError::InvalidAxis(_))));
    }

    #[test]
    fn negative_output_size_returns_error() {
        // Returns an error if the output size is negative
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, Some(-1), Some(0));
        assert_eq!(result, Err(FftError::InvalidOutputSize));
    }

    #[test]
    fn valid_input_returns_sizes_and_axis() {
        // Returns the output size and axis if the input is valid
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, Some(4), Some(0));
        assert_eq!(result, Ok((4, 0)));
    }
}

#[cfg(test)]
mod try_resolve_sizes_and_axes_tests {
    use crate::{error::DuplicateAxisError, Array};

    use super::{try_resolve_sizes_and_axes, FftError};

    #[test]
    fn scalar_array_returns_error() {
        // Returns an error if the array is a scalar
        let a = Array::from_float(1.0);
        let result = try_resolve_sizes_and_axes(&a, None, None);
        assert_eq!(result, Err(FftError::ScalarArray));
    }

    #[test]
    fn out_of_bound_axis_returns_error() {
        // Returns an error if the axis is invalid (out of bounds)
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, Some(&[2, 2, 2][..]), Some(&[0, 1, 2][..]));
        assert!(matches!(result, Err(FftError::InvalidAxis(_))));
    }

    #[test]
    fn different_num_sizes_and_num_axes_returns_error() {
        // Returns an error if the number of sizes and axes are different
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, Some(&[2, 2, 2][..]), Some(&[0, 1][..]));
        assert_eq!(
            result,
            Err(FftError::IncompatibleShapeAndAxes {
                shape_size: 3,
                axes_size: 2
            })
        );
    }

    #[test]
    fn duplicate_axes_returns_error() {
        // Returns an error if there are duplicate axes
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, Some(&[2, 2][..]), Some(&[0, 0][..]));
        assert_eq!(result, Err(DuplicateAxisError { axis: 0 }.into()));
    }

    #[test]
    fn negative_output_size_returns_error() {
        // Returns an error if the output size is negative
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, Some(&[-2, 2][..]), None);
        assert_eq!(result, Err(FftError::InvalidOutputSize));
    }
}
