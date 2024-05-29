use crate::error::{NormError, OrdNotImplementedError};
use crate::{Array, Stream, StreamOrDevice};
use mlx_macros::default_device;
use std::f64;
use std::ffi::{CStr, CString};

#[derive(Debug, Clone, Copy)]
pub enum Ord {
    Fro,
    // Nuc, // Not yet implemented
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

impl Default for Ord {
    fn default() -> Self {
        Ord::Fro
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
    array: &Array,
    ord: impl TryInto<Option<Ord>, Error = impl Into<OrdNotImplementedError<'a>>>,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, NormError<'a>> {
    let ord: Ord = ord
        .try_into()
        .map_err(Into::into)?
        .unwrap_or(Ord::default());
    let keep_dims = keep_dims.into().unwrap_or(false);
    let axes = axes.into();

    if let Some(axes) = &axes {
        if matches!(ord, Ord::Fro) && axes.len() != 2 {
            return Err(NormError::OrdRequiresMatrix { ord });
        }

        if axes.len() > 2 {
            return Err(NormError::TooManyAxes);
        }

        if axes.len() == 2 {
            match ord {
                Ord::P(p) if p == 2.0 || p == -2.0  => {
                    return Err(NormError::SingularValueNormNotImplemented)
                }
                Ord::P(p)
                    if p != -1.0
                        && p != 1.0
                        && p != f64::NEG_INFINITY
                        && p != f64::INFINITY =>
                {
                    return Err(NormError::InvalidMatrixOrd { ord })
                }
                _ => {},
            }
        }
    };

    match ord {
        Ord::Fro => {
            let ord = CString::new("fro").unwrap(); // "fro" does not contain any null bytes
            unsafe { Ok(norm_ord_device_unchecked(array, ord.as_c_str(), axes, keep_dims, stream)) }
        }
        Ord::P(ord) => unsafe { Ok(norm_p_device_unchecked(array, ord, axes, keep_dims, stream)) },
    }
}

#[default_device]
pub fn norm_device<'a>(
    array: &Array,
    ord: impl TryInto<Option<Ord>, Error = impl Into<OrdNotImplementedError<'a>>>,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Array {
    try_norm_device(array, ord, axes, keep_dims, stream).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tests below are adapted from the swift bindings tests
    // and they are not exhaustive. Additional tests should be added
    // to cover the error cases

    #[test]
    fn test_norm_no_axes() {
        let a = Array::from_iter(0..9, &[9]).as_ref() - 4;
        let b = a.reshape(&[3, 3]);

        norm(&a, None, None, None);
    }
}