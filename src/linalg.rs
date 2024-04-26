use crate::error::LinAlgError;
use crate::{Array, StreamOrDevice};
use mlx_macros::default_device;
use std::f64;

#[derive(Debug)]
pub enum Ord {
    Fro,
    // Nuc, // Not yet implemented
}

impl Ord {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Fro => "fro",
        }
    }
}

/// Calculates the Frobenius norm for matrices and the L2 norm for vectors. If axes are specified, the norm is computed along that axis.
/// If no axes are provided, the norm is calculated over the entire flattened array.
///
/// # Params
/// - array: input array
/// - axes: axes that hold 2d matrices
/// - keep_dims: if `true` the axes which are normed over are left in the result as dimensions with size one
/// - stream: stream to evaluate on
#[default_device]
pub fn try_norm_device<'a>(
    array: &Array,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Result<Array, LinAlgError> {
    let axes = axes.into();
    let keep_dims = keep_dims.into().unwrap_or(false);

    if let Some(axes) = axes {
        if axes.len() > 2 {
            return Err(LinAlgError::TooManyAxes);
        }
    }

    return unsafe {
        Ok(Array::from_ptr(mlx_sys::mlx_linalg_norm(
            array.as_ptr(),
            if axes.is_some() {
                axes.unwrap().as_ptr()
            } else {
                std::ptr::null()
            },
            if axes.is_some() {
                axes.unwrap().len()
            } else {
                0
            },
            keep_dims,
            stream.as_ptr(),
        )))
    };
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
pub fn try_norm_p_device<'a>(
    array: &Array,
    ord: f64,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Result<Array, LinAlgError> {
    let axes = axes
        .into()
        .map_or_else(|| (0..array.ndim() as i32).collect(), |axes| axes.to_vec());
    let keep_dims = keep_dims.into().unwrap_or(false);

    if axes.len() > 2 {
        return Err(LinAlgError::TooManyAxes);
    }

    if axes.len() == 2 {
        if ord == 2.0 || ord == -2.0 {
            return Err(LinAlgError::SingularValueNormNotImplemented);
        } else if ord != -1.0 && ord != 1.0 && ord != f64::NEG_INFINITY && ord != f64::INFINITY {
            return Err(LinAlgError::InvalidMatrixF64Ord { ord });
        }
    }

    return unsafe {
        Ok(Array::from_ptr(mlx_sys::mlx_linalg_norm_p(
            array.as_ptr(),
            ord,
            axes.as_ptr(),
            axes.len(),
            keep_dims,
            stream.as_ptr(),
        )))
    };
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
pub fn try_norm_ord_device<'a>(
    array: &Array,
    ord: impl Into<Option<Ord>>,
    axes: impl Into<Option<&'a [i32]>>,
    keep_dims: impl Into<Option<bool>>,
    stream: StreamOrDevice,
) -> Result<Array, LinAlgError> {
    let ord = match ord.into() {
        Some(ord) => ord,
        None => return try_norm_device(array, axes, keep_dims, stream),
    };

    let axes = axes
        .into()
        .map_or_else(|| (0..array.ndim() as i32).collect(), |axes| axes.to_vec());

    let keep_dims = keep_dims.into().unwrap_or(false);

    if axes.len() != 2 {
        return Err(LinAlgError::RequiresMatrix { ord: ord.as_str() });
    }

    // create a string from the ord (takes *const ::std::os::raw::c_char)
    let ord_str = unsafe { mlx_sys::mlx_string_new(ord.as_str().as_ptr() as *const i8) };
    let norm = unsafe {
        Ok(Array::from_ptr(mlx_sys::mlx_linalg_norm_ord(
            array.as_ptr(),
            ord_str,
            axes.as_ptr(),
            axes.len(),
            keep_dims,
            stream.as_ptr(),
        )))
    };

    unsafe { mlx_sys::mlx_free(ord_str as *mut ::std::os::raw::c_void) }
    norm
}
