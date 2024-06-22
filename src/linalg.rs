use crate::error::Exception;
use crate::utils::{IntoOption, MlxString, VectorArray};
use crate::{Array, Stream, StreamOrDevice};
use mlx_macros::default_device;
use smallvec::SmallVec;
use std::f64;
use std::ffi::CString;

#[derive(Debug, Clone, Copy)]
pub enum Ord<'a> {
    Str(&'a str),
    P(f64),
}

impl<'a> Default for Ord<'a> {
    fn default() -> Self {
        Ord::Str("fro")
    }
}

impl<'a> std::fmt::Display for Ord<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ord::Str(s) => write!(f, "{}", s),
            Ord::P(p) => write!(f, "{}", p),
        }
    }
}

impl<'a> From<&'a str> for Ord<'a> {
    fn from(value: &'a str) -> Self {
        Ord::Str(value)
    }
}

impl<'a> From<f64> for Ord<'a> {
    fn from(value: f64) -> Self {
        Ord::P(value)
    }
}

impl<'a> IntoOption<Ord<'a>> for &'a str {
    fn into_option(self) -> Option<Ord<'a>> {
        Some(Ord::Str(self))
    }
}

impl<'a> IntoOption<Ord<'a>> for f64 {
    fn into_option(self) -> Option<Ord<'a>> {
        Some(Ord::P(self))
    }
}

#[default_device]
pub fn norm_p_device<'a>(
    array: &Array,
    ord: f64,
    axes: impl IntoOption<&'a [i32]>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            match axes.into_option() {
                Some(axes) => {
                    mlx_sys::mlx_linalg_norm_p(
                        array.as_ptr(),
                        ord,
                        axes.as_ptr(),
                        axes.len(),
                        keep_dims,
                        stream.as_ref().as_ptr(),
                    )
                }
                None => {
                    mlx_sys::mlx_linalg_norm_p(
                        array.as_ptr(),
                        ord,
                        std::ptr::null(),
                        0,
                        keep_dims,
                        stream.as_ref().as_ptr(),
                    )
                }
            }
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Matrix or vector norm.
#[default_device]
pub fn norm_ord_device<'a>(
    array: &Array,
    ord: &'a str,
    axes: impl IntoOption<&'a [i32]>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    unsafe {
        let ord = MlxString::try_from(ord).map_err(|_e| Exception {
            what: CString::new("NulError").unwrap(),
        })?;

        let c_array = try_catch_c_ptr_expr! {
            match axes.into_option() {
                Some(axes) => {
                    mlx_sys::mlx_linalg_norm_ord(
                        array.as_ptr(),
                        ord.as_ptr(),
                        axes.as_ptr(),
                        axes.len(),
                        keep_dims.into().unwrap_or(false),
                        stream.as_ref().as_ptr(),
                    )
                }
                None => {
                    mlx_sys::mlx_linalg_norm_ord(
                        array.as_ptr(),
                        ord.as_ptr(),
                        std::ptr::null(),
                        0,
                        keep_dims.into().unwrap_or(false),
                        stream.as_ref().as_ptr(),
                    )
                }
            }
        };

        Ok(Array::from_ptr(c_array))
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
///
/// - `array`: input array
/// - `ord`: order of the norm, see table
/// - `axes`: axes that hold 2d matrices
/// - `keep_dims`: if `true` the axes which are normed over are left in the result as dimensions
///   with size one
#[default_device]
pub fn norm_device<'a>(
    array: &Array,
    ord: impl IntoOption<Ord<'a>>,
    axes: impl IntoOption<&'a [i32]>,
    keep_dims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let ord = ord.into_option();
    let axes = axes.into_option();
    let keep_dims = keep_dims.into().unwrap_or(false);

    match (ord, axes) {
        // If axis and ord are both unspecified, computes the 2-norm of flatten(x).
        (None, None) => unsafe {
            let axes_ptr = std::ptr::null(); // mlx-c already handles the case where axes is null
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_linalg_norm(
                    array.as_ptr(),
                    axes_ptr,
                    0,
                    keep_dims,
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        },
        // If axis is not provided but ord is, then x must be either 1D or 2D.
        //
        // Frobenius norm is only supported for matrices
        (Some(Ord::Str(ord)), None) => norm_ord_device(array, ord, axes, keep_dims, stream),
        (Some(Ord::P(p)), None) => norm_p_device(array, p, axes, keep_dims, stream),
        // If axis is provided, but ord is not, then the 2-norm (or Frobenius norm for matrices) is
        // computed along the given axes. At most 2 axes can be specified.
        (None, Some(axes)) => unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_linalg_norm(
                    array.as_ptr(),
                    axes.as_ptr(),
                    axes.len(),
                    keep_dims,
                    stream.as_ref().as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        },
        // If both axis and ord are provided, then the corresponding matrix or vector
        // norm is computed. At most 2 axes can be specified.
        (Some(Ord::Str(ord)), Some(axes)) => norm_ord_device(array, ord, axes, keep_dims, stream),
        (Some(Ord::P(p)), Some(axes)) => norm_p_device(array, p, axes, keep_dims, stream),
    }
}

/// The QR factorization of the input matrix. Returns an error if the input is not valid.
///
/// This function supports arrays with at least 2 dimensions. The matrices which are factorized are
/// assumed to be in the last two dimensions of the input.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, linalg::*};
///
/// let a = Array::from_slice(&[2.0f32, 3.0, 1.0, 2.0], &[2, 2]);
///
/// let (q, r) = qr_device(&a, StreamOrDevice::cpu()).unwrap();
///
/// let q_expected = Array::from_slice(&[-0.894427, -0.447214, -0.447214, 0.894427], &[2, 2]);
/// let r_expected = Array::from_slice(&[-2.23607, -3.57771, 0.0, 0.447214], &[2, 2]);
///
/// assert!(q.all_close(&q_expected, None, None, None).unwrap().item::<bool>());
/// assert!(r.all_close(&r_expected, None, None, None).unwrap().item::<bool>());
/// ```
#[default_device]
pub fn qr_device(a: &Array, stream: impl AsRef<Stream>) -> Result<(Array, Array), Exception> {
    unsafe {
        let c_vec = try_catch_c_ptr_expr! {
            mlx_sys::mlx_linalg_qr(a.as_ptr(), stream.as_ref().as_ptr())
        };

        let v = VectorArray::from_ptr(c_vec);

        let vals: SmallVec<[Array; 2]> = v.into_values();
        let mut iter = vals.into_iter();
        let q = iter.next().unwrap();
        let r = iter.next().unwrap();

        Ok((q, r))
    }
}

/// The Singular Value Decomposition (SVD) of the input matrix. Returns an error if the input is not
/// valid.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the function iterates over all indices of the first a.ndim - 2 dimensions and for
/// each combination SVD is applied to the last two indices.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, linalg::*};
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
/// let (u, s, vt) = svd_device(&a, StreamOrDevice::cpu()).unwrap();
/// let u_expected = Array::from_slice(&[-0.404554, 0.914514, -0.914514, -0.404554], &[2, 2]);
/// let s_expected = Array::from_slice(&[5.46499, 0.365966], &[2]);
/// let vt_expected = Array::from_slice(&[-0.576048, -0.817416, -0.817415, 0.576048], &[2, 2]);
/// assert!(u.all_close(&u_expected, None, None, None).unwrap().item::<bool>());
/// assert!(s.all_close(&s_expected, None, None, None).unwrap().item::<bool>());
/// assert!(vt.all_close(&vt_expected, None, None, None).unwrap().item::<bool>());
/// ```
#[default_device]
pub fn svd_device(
    array: &Array,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array), Exception> {
    unsafe {
        let c_vec = try_catch_c_ptr_expr! {
            mlx_sys::mlx_linalg_svd(array.as_ptr(), stream.as_ref().as_ptr())
        };

        let v = VectorArray::from_ptr(c_vec);

        let vals: SmallVec<[Array; 3]> = v.into_values();
        let mut iter = vals.into_iter();
        let u = iter.next().unwrap();
        let s = iter.next().unwrap();
        let vt = iter.next().unwrap();

        Ok((u, s, vt))
    }
}

/// Compute the inverse of a square matrix. Returns an error if the input is not valid.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the inverse is computed for each matrix in the last two dimensions of `a`.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `a`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{prelude::*, linalg::*};
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
/// let a_inv = inv_device(&a, StreamOrDevice::cpu()).unwrap();
/// let expected = Array::from_slice(&[-2.0, 1.0, 1.5, -0.5], &[2, 2]);
/// assert!(a_inv.all_close(&expected, None, None, None).unwrap().item::<bool>());
/// ```
#[default_device]
pub fn inv_device(a: &Array, stream: impl AsRef<Stream>) -> Result<Array, Exception> {
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_linalg_inv(a.as_ptr(), stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the Cholesky decomposition is computed for each matrix in the last two dimensions of
/// `a`.
///
/// If the input matrix is not symmetric positive semi-definite, behaviour is undefined.
///
/// # Params
///
/// - `a`: input array
/// - `upper`: If `true`, return the upper triangular Cholesky factor. If `false`, return the lower
///   triangular Cholesky factor. Default: `false`.
pub fn cholesky(
    a: &Array,
    upper: Option<bool>,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let upper = upper.unwrap_or(false);
    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_linalg_cholesky(a.as_ptr(), upper, stream.as_ref().as_ptr())
        };

        Ok(Array::from_ptr(c_array))
    }
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
        let a = Array::from_iter(0..9, &[9]) - 4;
        let b = a.reshape(&[3, 3]).unwrap();

        assert_float_eq!(
            norm(&a, None, None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, None, None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&b, "fro", None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, f64::INFINITY, None, None).unwrap().item::<f32>(),
            4.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, f64::INFINITY, None, None).unwrap().item::<f32>(),
            9.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, f64::NEG_INFINITY, None, None)
                .unwrap()
                .item::<f32>(),
            0.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, f64::NEG_INFINITY, None, None)
                .unwrap()
                .item::<f32>(),
            2.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, 1.0, None, None).unwrap().item::<f32>(),
            20.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, 1.0, None, None).unwrap().item::<f32>(),
            7.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, -1.0, None, None).unwrap().item::<f32>(),
            0.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, -1.0, None, None).unwrap().item::<f32>(),
            6.0,
            abs <= 0.001
        );
    }

    #[test]
    fn test_norm_axis() {
        let c = Array::from_slice(&[1, 2, 3, -1, 1, 4], &[2, 3]);

        let result = norm(&c, None, &[0][..], None).unwrap();
        let expected = Array::from_slice(&[1.41421, 2.23607, 5.0], &[3]);
        assert!(result
            .all_close(&expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_norm_axes() {
        let m = Array::from_iter(0..8, &[2, 2, 2]);

        let result = norm(&m, None, &[1, 2][..], None).unwrap();
        let expected = Array::from_slice(&[3.74166, 11.225], &[2]);
        assert!(result
            .all_close(&expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_qr() {
        let a = Array::from_slice(&[2.0f32, 3.0, 1.0, 2.0], &[2, 2]);

        let (q, r) = qr_device(&a, StreamOrDevice::cpu()).unwrap();

        let q_expected = Array::from_slice(&[-0.894427, -0.447214, -0.447214, 0.894427], &[2, 2]);
        let r_expected = Array::from_slice(&[-2.23607, -3.57771, 0.0, 0.447214], &[2, 2]);

        assert!(q
            .all_close(&q_expected, None, None, None)
            .unwrap()
            .item::<bool>());
        assert!(r
            .all_close(&r_expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    // The tests below are adapted from the c++ tests

    #[test]
    fn test_svd() {
        // TEST_CASE("test SVD factorization") {
        //     // 0D and 1D throw
        //     CHECK_THROWS(linalg::svd(array(0.0)));
        //     CHECK_THROWS(linalg::svd(array({0.0, 1.0})));

        //     // Unsupported types throw
        //     CHECK_THROWS(linalg::svd(array({0, 1}, {1, 2})));

        //     const auto prng_key = random::key(42);
        //     const auto A = mlx::core::random::normal({5, 4}, prng_key);
        //     const auto outs = linalg::svd(A, Device::cpu);
        //     CHECK_EQ(outs.size(), 3);

        //     const auto& U = outs[0];
        //     const auto& S = outs[1];
        //     const auto& Vt = outs[2];

        //     CHECK_EQ(U.shape(), std::vector<int>{5, 5});
        //     CHECK_EQ(S.shape(), std::vector<int>{4});
        //     CHECK_EQ(Vt.shape(), std::vector<int>{4, 4});

        //     const auto U_slice = slice(U, {0, 0}, {U.shape(0), S.shape(0)});

        //     const auto A_again = matmul(matmul(U_slice, diag(S)), Vt);

        //     CHECK(
        //         allclose(A_again, A, /* rtol = */ 1e-4, /* atol = */ 1e-4).item<bool>());
        //     CHECK_EQ(U.dtype(), float32);
        //     CHECK_EQ(S.dtype(), float32);
        //     CHECK_EQ(Vt.dtype(), float32);
        //   }

        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_float(0.0);
        assert!(svd_device(&a, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(svd_device(&a, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[0, 1], &[1, 2]);
        assert!(svd_device(&a, &stream).is_err());

        // TODO: wait for random
    }

    #[test]
    fn test_inv() {
        // TEST_CASE("test matrix inversion") {
        //     // 0D and 1D throw
        //     CHECK_THROWS(linalg::inv(array(0.0), Device::cpu));
        //     CHECK_THROWS(linalg::inv(array({0.0, 1.0}), Device::cpu));

        //     // Unsupported types throw
        //     CHECK_THROWS(linalg::inv(array({0, 1}, {1, 2}), Device::cpu));

        //     // Non-square throws.
        //     CHECK_THROWS(linalg::inv(array({1, 2, 3, 4, 5, 6}, {2, 3}), Device::cpu));

        //     const auto prng_key = random::key(42);
        //     const auto A = random::normal({5, 5}, prng_key);
        //     const auto A_inv = linalg::inv(A, Device::cpu);
        //     const auto identity = eye(A.shape(0));

        //     CHECK(allclose(matmul(A, A_inv), identity, /* rtol = */ 0, /* atol = */ 1e-6)
        //               .item<bool>());
        //     CHECK(allclose(matmul(A_inv, A), identity, /* rtol = */ 0, /* atol = */ 1e-6)
        //               .item<bool>());
        //   }

        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_float(0.0);
        assert!(inv_device(&a, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(inv_device(&a, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert!(inv_device(&a, &stream).is_err());

        // TODO: wait for random
    }

    #[test]
    fn test_cholesky() {
        // TEST_CASE("test matrix cholesky") {
        //     // 0D and 1D throw
        //     CHECK_THROWS(linalg::cholesky(array(0.0), /* upper = */ false, Device::cpu));
        //     CHECK_THROWS(
        //         linalg::cholesky(array({0.0, 1.0}), /* upper = */ false, Device::cpu));

        //     // Unsupported types throw
        //     CHECK_THROWS(linalg::cholesky(
        //         array({0, 1}, {1, 2}), /* upper = */ false, Device::cpu));

        //     // Non-square throws.
        //     CHECK_THROWS(linalg::cholesky(
        //         array({1, 2, 3, 4, 5, 6}, {2, 3}), /* upper = */ false, Device::cpu));

        //     const auto prng_key = random::key(220398);
        //     const auto sqrtA = random::normal({5, 5}, prng_key);
        //     const auto A = matmul(sqrtA, transpose(sqrtA));
        //     const auto L = linalg::cholesky(A, /* upper = */ false, Device::cpu);
        //     const auto U = linalg::cholesky(A, /* upper = */ true, Device::cpu);

        //     CHECK(allclose(matmul(L, transpose(L)), A, /* rtol = */ 0, /* atol = */ 1e-6)
        //               .item<bool>());
        //     CHECK(allclose(matmul(transpose(U), U), A, /* rtol = */ 0, /* atol = */ 1e-6)
        //               .item<bool>());
        //   }

        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_float(0.0);
        assert!(cholesky(&a, None, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(cholesky(&a, None, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[0, 1, 1, 2], &[2, 2]);
        assert!(cholesky(&a, None, &stream).is_err());

        // Non-square returns error
        let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert!(cholesky(&a, None, &stream).is_err());

        // TODO: wait for random
    }
}
