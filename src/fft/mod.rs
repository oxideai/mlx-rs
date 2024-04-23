//! Fast Fourier Transform (FFT) and its inverse (IFFT) for one, two, and `N` dimensions.
//!
//! Like all other functions in `mlx-rs`, three variants are provided for each FFT function, plus
//! each variant has a version that uses the default `StreamOrDevice` or takes a user-specified
//! `StreamOrDevice`.
//!
//! The difference are explained below using `fftn` as an example:
//!
//! 1. `fftn_unchecked`/`fftn_device_unchecked`: This function is simply a wrapper around the C API
//!   and does not perform any checks on the input. It may panic or get an fatal error that cannot
//!   be caught by the rust runtime if the input is invalid.
//! 2. `try_fftn`/`try_fftn_device`: This function performs checks on the input and returns a
//!   `Result` instead of panicking.
//! 3. `fftn`/`fftn_device`: This function is a wrapper around `try_fftn` and unwraps the result. It
//!   panics if the input is invalid.
//!
//! The functions that contains `device` in their name are meant to be used with a user-specified
//! `StreamOrDevice`. If you don't care about the stream, you can use the functions without `device`
//! in their names. Please note that GPU device support is not yet implemented.
//!
//! # Examples
//!
//! ## One dimension
//!
//! ```rust
//! use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
//!
//! let src = [1.0f32, 2.0, 3.0, 4.0];
//! let array = Array::from_slice(&src[..], &[4]);
//!
//! let mut fft_result = fft(&array, 4, 0);
//! fft_result.eval();
//! assert_eq!(fft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(10.0, 0.0),
//!     complex64::new(-2.0, 2.0),
//!     complex64::new(-2.0, 0.0),
//!     complex64::new(-2.0, -2.0),
//! ];
//! assert_eq!(fft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft_result = ifft(&fft_result, 4, 0);
//! ifft_result.eval();
//! assert_eq!(ifft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(2.0, 0.0),
//!    complex64::new(3.0, 0.0),
//!    complex64::new(4.0, 0.0),
//! ];
//! assert_eq!(ifft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft_result = rfft(&array, 4, 0);
//! rfft_result.eval();
//! assert_eq!(rfft_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(10.0, 0.0),
//!    complex64::new(-2.0, 2.0),
//!    complex64::new(-2.0, 0.0),
//! ];
//! assert_eq!(rfft_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft_result = irfft(&rfft_result, 4, 0);
//! irfft_result.eval();
//! assert_eq!(irfft_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft_result.as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &src[..]);
//! ```
//!
//! ## Two dimensions
//!
//! ```rust
//! use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
//!
//! let src = [1.0f32, 1.0, 1.0, 1.0];
//! let array = Array::from_slice(&src[..], &[2, 2]);
//!
//! let mut fft2_result = fft2(&array, None, None);
//! fft2_result.eval();
//! assert_eq!(fft2_result.dtype(), Dtype::Complex64);
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! assert_eq!(fft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifft2_result = ifft2(&fft2_result, None, None);
//! ifft2_result.eval();
//! assert_eq!(ifft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//! ];
//! assert_eq!(ifft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfft2_result = rfft2(&array, None, None);
//! rfft2_result.eval();
//! assert_eq!(rfft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = &[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ];
//! assert_eq!(rfft2_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfft2_result = irfft2(&rfft2_result, None, None);
//! irfft2_result.eval();
//! assert_eq!(irfft2_result.dtype(), Dtype::Float32);
//! assert_eq!(irfft2_result.as_slice::<f32>(), &src[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
//! ```
//!
//! ## `N` dimensions
//!
//! ```rust
//! use mlx::{Dtype, Array, StreamOrDevice, complex64, fft::*};
//!
//! let array = Array::ones::<f32>(&[2, 2, 2]);
//! let mut fftn_result = fftn(&array, None, None);
//! fftn_result.eval();
//! assert_eq!(fftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_eq!(fftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut ifftn_result = ifftn(&fftn_result, None, None);
//! ifftn_result.eval();
//! assert_eq!(ifftn_result.dtype(), Dtype::Complex64);
//!
//! let expected = [complex64::new(1.0, 0.0); 8];
//! assert_eq!(ifftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut rfftn_result = rfftn(&array, None, None);
//! rfftn_result.eval();
//! assert_eq!(rfftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_eq!(rfftn_result.as_slice::<complex64>(), &expected[..]);
//!
//! let mut irfftn_result = irfftn(&rfftn_result, None, None);
//! irfftn_result.eval();
//! assert_eq!(irfftn_result.dtype(), Dtype::Float32);
//!
//! let expected = [1.0; 8];
//! assert_eq!(irfftn_result.as_slice::<f32>(), &expected[..]);
//!
//! // The original array is not modified
//! let data: &[f32] = array.as_slice();
//! assert_eq!(data, &[1.0; 8]);
//! ```

mod fftn;
mod ifftn;
mod irfftn;
mod rfftn;

use smallvec::SmallVec;

use crate::{
    error::FftError,
    utils::{all_unique, resolve_index, resolve_index_unchecked},
    Array,
};

pub use self::{fftn::*, ifftn::*, irfftn::*, rfftn::*};

#[inline]
fn resolve_size_and_axis_unchecked(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
) -> (i32, i32) {
    let axis = axis.into().unwrap_or(-1);
    let n = n.into().unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    (n, axis)
}

#[inline]
fn try_resolve_size_and_axis(
    a: &Array,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
) -> Result<(i32, i32), FftError> {
    if a.ndim() < 1 {
        return Err(FftError::ScalarArray);
    }

    let axis = axis.into().unwrap_or(-1);
    let axis_index =
        resolve_index(axis, a.ndim()).ok_or_else(|| FftError::InvalidAxis { ndim: a.ndim() })?;
    let n = n.into().unwrap_or(a.shape()[axis_index]);

    if n <= 0 {
        return Err(FftError::InvalidOutputSize);
    }

    Ok((n, axis))
}

#[inline]
fn resolve_sizes_and_axes_unchecked<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
) -> (SmallVec<[i32; 4]>, SmallVec<[i32; 4]>) {
    match (s.into(), axes.into()) {
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
fn try_resolve_sizes_and_axes<'a>(
    a: &'a Array,
    s: impl Into<Option<&'a [i32]>>,
    axes: impl Into<Option<&'a [i32]>>,
) -> Result<(SmallVec<[i32; 4]>, SmallVec<[i32; 4]>), FftError> {
    if a.ndim() < 1 {
        return Err(FftError::ScalarArray);
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
                    .ok_or_else(|| FftError::InvalidAxis { ndim: a.ndim() })?;
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
    all_unique(&valid_axes).map_err(|axis| FftError::DuplicateAxis { axis })?;

    // Check if shape and axes have the same size
    if valid_s.len() != valid_axes.len() {
        return Err(FftError::IncompatibleShapeAndAxes {
            shape_size: valid_s.len(),
            axes_size: valid_axes.len(),
        });
    }

    // Check if more axes are provided than the array has
    if valid_s.len() > a.ndim() {
        return Err(FftError::InvalidAxis { ndim: a.ndim() });
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
        let result = try_resolve_size_and_axis(&a, 0, 0);
        assert_eq!(result, Err(FftError::ScalarArray));
    }

    #[test]
    fn out_of_bound_axis_returns_error() {
        // Returns an error if the axis is invalid (out of bounds)
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, 0, 1);
        assert_eq!(result, Err(FftError::InvalidAxis { ndim: 1 }));
    }

    #[test]
    fn negative_output_size_returns_error() {
        // Returns an error if the output size is negative
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, -1, 0);
        assert_eq!(result, Err(FftError::InvalidOutputSize));
    }

    #[test]
    fn valid_input_returns_sizes_and_axis() {
        // Returns the output size and axis if the input is valid
        let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = try_resolve_size_and_axis(&a, 4, 0);
        assert_eq!(result, Ok((4, 0)));
    }
}

#[cfg(test)]
mod try_resolve_sizes_and_axes_tests {
    use crate::Array;

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
        let result = try_resolve_sizes_and_axes(&a, &[2, 2, 2][..], &[0, 1, 2][..]);
        assert_eq!(result, Err(FftError::InvalidAxis { ndim: 2 }));
    }

    #[test]
    fn different_num_sizes_and_num_axes_returns_error() {
        // Returns an error if the number of sizes and axes are different
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, &[2, 2, 2][..], &[0, 1][..]);
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
        let result = try_resolve_sizes_and_axes(&a, &[2, 2][..], &[0, 0][..]);
        assert_eq!(result, Err(FftError::DuplicateAxis { axis: 0 }));
    }

    #[test]
    fn negative_output_size_returns_error() {
        // Returns an error if the output size is negative
        let a = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);
        let result = try_resolve_sizes_and_axes(&a, &[-2, 2][..], None);
        assert_eq!(result, Err(FftError::InvalidOutputSize));
    }
}
