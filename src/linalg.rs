use crate::error::{InvalidAxisError, NormError, OrdNotImplementedError};
use crate::utils::{axes_or_default_to_all, resolve_index};
use crate::{Array, Stream, StreamOrDevice};
use mlx_macros::default_device;
use std::f64;
use std::ffi::{CStr, CString};

#[derive(Debug, Clone, Copy, Default)]
pub enum Ord {
    #[default]
    Fro,
    P(f64),
}

impl std::fmt::Display for Ord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ord::Fro => write!(f, "fro"),
            Ord::P(p) => write!(f, "{}", p),
        }
    }
}

impl Ord {
    pub fn inf() -> Self {
        Ord::P(f64::INFINITY)
    }

    pub fn neg_inf() -> Self {
        Ord::P(f64::NEG_INFINITY)
    }
}

impl From<f64> for Ord {
    fn from(value: f64) -> Self {
        Ord::P(value)
    }
}

impl<'a> TryFrom<&'a str> for Ord {
    type Error = OrdNotImplementedError<'a>;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match value {
            "fro" => Ok(Ord::Fro),
            "inf" => Ok(Ord::P(f64::INFINITY)),
            "-inf" => Ok(Ord::P(f64::NEG_INFINITY)),
            _ => Err(OrdNotImplementedError { ord: value }),
        }
    }
}

pub trait TryIntoOptionOrd<'a> {
    fn try_into_option_ord(self) -> Result<Option<Ord>, OrdNotImplementedError<'a>>;
}

impl<'a> TryIntoOptionOrd<'a> for f64 {
    fn try_into_option_ord(self) -> Result<Option<Ord>, OrdNotImplementedError<'a>> {
        Ok(Some(Ord::P(self)))
    }
}

impl<'a> TryIntoOptionOrd<'a> for &'a str {
    fn try_into_option_ord(self) -> Result<Option<Ord>, OrdNotImplementedError<'a>> {
        let ord = Ord::try_from(self)?;
        Ok(Some(ord))
    }
}

impl<'a> TryIntoOptionOrd<'a> for Option<Ord> {
    fn try_into_option_ord(self) -> Result<Option<Ord>, OrdNotImplementedError<'a>> {
        Ok(self)
    }
}

/// Matrix or vector norm.
///
/// # Safety
///
/// This is unsafe because it does not check if the arguments are valid.
#[default_device]
pub unsafe fn norm_p_device_unchecked<'a>(
    array: &Array,
    ord: f64,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = match axes.into() {
            Some(axes) => mlx_sys::mlx_linalg_norm_p(
                array.as_ptr(),
                ord,
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_ref().as_ptr(),
            ),
            None => {
                // mlx-c already handles the case where axes is null
                let axes_ptr = std::ptr::null();
                mlx_sys::mlx_linalg_norm_p(
                    array.as_ptr(),
                    ord,
                    axes_ptr,
                    0,
                    keep_dims,
                    stream.as_ref().as_ptr(),
                )
            }
        };

        Array::from_ptr(c_array)
    }
}

#[default_device]
pub fn try_norm_p_device<'a>(
    array: &Array,
    ord: f64,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, NormError> {
    let axes = axes_or_default_to_all(axes, array.ndim() as i32);

    match axes.len() {
        1 => {}
        2 => {
            if ord == 2.0 || ord == -2.0 {
                return Err(NormError::SingularValueNormNotImplemented);
            } else if ord != -1.0 && ord != 1.0 && ord != f64::NEG_INFINITY && ord != f64::INFINITY
            {
                return Err(NormError::InvalidMatrixOrd { ord: Ord::P(ord) });
            }
        }
        _ => return Err(NormError::TooManyAxes),
    }

    // Check if the axes are valid
    for axis in axes.iter() {
        resolve_index(*axis, array.ndim()).ok_or_else(|| InvalidAxisError {
            axis: *axis,
            ndim: array.ndim(),
        })?;
    }

    unsafe {
        Ok(norm_p_device_unchecked(
            array,
            ord,
            axes.as_slice(),
            keep_dims,
            stream,
        ))
    }
}

/// Matrix or vector norm.
///
/// # Safety
///
/// This is unsafe because it does not check if the arguments are valid.
#[default_device]
pub unsafe fn norm_ord_device_unchecked<'a>(
    array: &Array,
    ord: &'a CStr,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let mlx_ord = mlx_sys::mlx_string_new(ord.as_ptr());

        let c_array = match axes.into() {
            Some(axes) => mlx_sys::mlx_linalg_norm_ord(
                array.as_ptr(),
                mlx_ord,
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ),
            None => {
                // mlx-c already handles the case where axes is null
                let axes_ptr = std::ptr::null();
                mlx_sys::mlx_linalg_norm_ord(
                    array.as_ptr(),
                    mlx_ord,
                    axes_ptr,
                    0,
                    keep_dims.into().unwrap_or(false),
                    stream.as_ref().as_ptr(),
                )
            }
        };

        mlx_sys::mlx_free(mlx_ord as *mut ::std::os::raw::c_void);

        Array::from_ptr(c_array)
    }
}

fn try_norm_fro_device<'a>(
    array: &Array,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, NormError<'a>> {
    let axes = axes_or_default_to_all(axes, array.ndim() as i32);

    if axes.len() != 2 {
        return Err(NormError::InvalidMatrixOrd { ord: Ord::Fro });
    }

    unsafe {
        Ok(norm_ord_device_unchecked(
            array,
            CString::new("fro").unwrap().as_c_str(),
            axes.as_slice(),
            keep_dims,
            stream,
        ))
    }
}

#[default_device]
pub fn try_norm_ord_device<'a>(
    array: &'a Array,
    ord: &'a str,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, NormError<'a>> {
    let ord = Ord::try_from(ord)?;

    match ord {
        Ord::Fro => try_norm_fro_device(array, axes, keep_dims, stream),
        Ord::P(ord) => try_norm_p_device(array, ord, axes, keep_dims, stream),
    }
}

/// 2-norm of a matrix or vector.
///
/// # Safety
///
/// This is unsafe because it does not check if the arguments are valid.
pub unsafe fn l2_norm_device_unchecked<'a>(
    array: &Array,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let axes = axes.into();

    unsafe {
        let c_array = match axes {
            Some(axes) => mlx_sys::mlx_linalg_norm(
                array.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            ),
            None => {
                // mlx-c already handles the case where axes is null
                let axes_ptr = std::ptr::null();
                mlx_sys::mlx_linalg_norm(
                    array.as_ptr(),
                    axes_ptr,
                    0,
                    keep_dims.into().unwrap_or(false),
                    stream.as_ref().as_ptr(),
                )
            }
        };

        Array::from_ptr(c_array)
    }
}

/// Matrix or vector norm.
///
/// For values of `ord < 1`, the result is, strictly speaking, not a
/// mathematical norm, but it may still be useful for various numerical
/// purposes.
///
/// The following norms can be calculated:
///
/// ord   | norm for matrices            | norm for vectors
/// ----- | ---------------------------- | --------------------------
/// None  | Frobenius norm               | 2-norm
/// 'fro' | Frobenius norm               | --
/// inf   | max(sum(abs(x), axis-1))     | max(abs(x))
/// -inf  | min(sum(abs(x), axis-1))     | min(abs(x))
/// 0     | --                           | sum(x !- 0)
/// 1     | max(sum(abs(x), axis-0))     | as below
/// -1    | min(sum(abs(x), axis-0))     | as below
/// 2     | 2-norm (largest sing. value) | as below
/// -2    | smallest singular value      | as below
/// other | --                           | sum(abs(x)**ord)**(1./ord)
///
/// > Nuclear norm and norms based on singular values are not yet implemented.
///
/// The Frobenius norm is given by G. H. Golub and C. F. Van Loan, *Matrix Computations*,
///        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
///
/// The nuclear norm is the sum of the singular values.
///
/// Both the Frobenius and nuclear norm orders are only defined for
/// matrices and produce a fatal error when `array.ndim != 2`
///
/// # Params
/// - array: input array
/// - ord: order of the norm, see table
/// - axes: axes that hold 2d matrices
/// - keep_dims: if `true` the axes which are normed over are left in the result as dimensions with size one
/// - stream: stream to evaluate on
#[default_device]
pub fn try_norm_device<'a>(
    array: &'a Array,
    ord: impl TryIntoOptionOrd<'a>,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, NormError<'a>> {
    let ord = ord.try_into_option_ord()?;
    let axes = axes.into();
    let keep_dims = keep_dims.into().unwrap_or(false);

    match (ord, axes) {
        // If axis and ord are both unspecified, computes the 2-norm of flatten(x).
        (None, None) => unsafe {
            let axes_ptr = std::ptr::null(); // mlx-c already handles the case where axes is null
            let c_array = mlx_sys::mlx_linalg_norm(
                array.as_ptr(),
                axes_ptr,
                0,
                keep_dims,
                stream.as_ref().as_ptr(),
            );
            Ok(Array::from_ptr(c_array))
        },
        // If axis is not provided but ord is, then x must be either 1D or 2D.
        //
        // Frobenius norm is only supported for matrices
        (Some(Ord::Fro), None) => try_norm_fro_device(array, axes, keep_dims, stream),
        (Some(Ord::P(p)), None) => try_norm_p_device(array, p, axes, keep_dims, stream),
        // If axis is provided, but ord is not, then the 2-norm (or Frobenius norm for matrices) is
        // computed along the given axes. At most 2 axes can be specified.
        (None, Some(axes)) => {
            if axes.len() > 2 {
                return Err(NormError::TooManyAxes);
            }

            unsafe {
                let c_array = mlx_sys::mlx_linalg_norm(
                    array.as_ptr(),
                    axes.as_ptr(),
                    axes.len(),
                    keep_dims,
                    stream.as_ref().as_ptr(),
                );
                Ok(Array::from_ptr(c_array))
            }
        }
        // If both axis and ord are provided, then the corresponding matrix or vector
        // norm is computed. At most 2 axes can be specified.
        (Some(Ord::Fro), Some(axes)) => try_norm_fro_device(array, axes, keep_dims, stream),
        (Some(Ord::P(p)), Some(axes)) => try_norm_p_device(array, p, axes, keep_dims, stream),
    }
}

#[default_device]
pub fn norm_device<'a>(
    array: &'a Array,
    ord: impl TryIntoOptionOrd<'a>,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_norm_device(array, ord, axes, keep_dims, stream).unwrap()
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    // The tests below are adapted from the swift bindings tests
    // and they are not exhaustive. Additional tests should be added
    // to cover the error cases

    #[test]
    fn test_norm_no_axes() {
        let a = Array::from_iter(0..9, &[9]).as_ref() - 4;
        let b = a.reshape(&[3, 3]);

        assert_float_eq!(
            norm(&a, None, None, None).item::<f32>(),
            7.74597,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, None, None, None).item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&b, "fro", None, None).item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(norm(&a, "inf", None, None).item::<f32>(), 4.0, abs <= 0.001);
        assert_float_eq!(norm(&b, "inf", None, None).item::<f32>(), 9.0, abs <= 0.001);

        assert_float_eq!(
            norm(&a, "-inf", None, None).item::<f32>(),
            0.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, "-inf", None, None).item::<f32>(),
            2.0,
            abs <= 0.001
        );

        assert_float_eq!(norm(&a, 1.0, None, None).item::<f32>(), 20.0, abs <= 0.001);
        assert_float_eq!(norm(&b, 1.0, None, None).item::<f32>(), 7.0, abs <= 0.001);

        assert_float_eq!(norm(&a, -1.0, None, None).item::<f32>(), 0.0, abs <= 0.001);
        assert_float_eq!(norm(&b, -1.0, None, None).item::<f32>(), 6.0, abs <= 0.001);
    }

    #[test]
    fn test_norm_axis() {
        let c = Array::from_slice(&[1, 2, 3, -1, 1, 4], &[2, 3]);

        let result = norm(&c, None, &[0][..], None);
        let expected = Array::from_slice(&[1.41421, 2.23607, 5.0], &[3]);
        assert!(result.all_close(&expected, None, None, None).item::<bool>());
    }

    #[test]
    fn test_norm_axes() {
        let m = Array::from_iter(0..8, &[2, 2, 2]);

        let result = norm(&m, None, &[1, 2][..], None);
        let expected = Array::from_slice(&[3.74166, 11.225], &[2]);
        assert!(result.all_close(&expected, None, None, None).item::<bool>());
    }
}
