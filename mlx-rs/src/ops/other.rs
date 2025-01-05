use std::ffi::CString;

use mlx_internal_macros::default_device;

use crate::utils::guard::Guarded;
use crate::utils::VectorArray;
use crate::{
    error::{Exception, Result},
    Array, Stream, StreamOrDevice,
};

impl Array {
    /// Extract a diagonal or construct a diagonal matrix.
    ///
    /// If self is 1-D then a diagonal matrix is constructed with self on the `k`-th diagonal. If
    /// self is 2-D then the `k`-th diagonal is returned.
    ///
    /// # Params:
    ///
    /// - `k`: the diagonal to extract or construct
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diag_device(
        &self,
        k: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_diag(
                res,
                self.as_ptr(),
                k.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Return specified diagonals.
    ///
    /// If self is 2-D, then a 1-D array containing the diagonal at the given `offset` is returned.
    ///
    /// If self has more than two dimensions, then `axis1` and `axis2` determine the 2D subarrays
    /// from which diagonals are extracted. The new shape is the original shape with `axis1` and
    /// `axis2` removed and a new dimension inserted at the end corresponding to the diagonal.
    ///
    /// # Params:
    ///
    /// - `offset`: offset of the diagonal.  Can be positive or negative
    /// - `axis1`: first axis of the 2-D sub-array from which the diagonals should be taken
    /// - `axis2`: second axis of the 2-D sub-array from which the diagonals should be taken
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diagonal_device(
        &self,
        offset: impl Into<Option<i32>>,
        axis1: impl Into<Option<i32>>,
        axis2: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_diagonal(
                res,
                self.as_ptr(),
                offset.into().unwrap_or(0),
                axis1.into().unwrap_or(0),
                axis2.into().unwrap_or(1),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Perform the Walsh-Hadamard transform along the final axis.
    ///
    /// Supports sizes `n = m*2^k` for `m` in `(1, 12, 20, 28)` and `2^k <= 8192`
    /// for ``DType/float32`` and `2^k <= 16384` for ``DType/float16`` and ``DType/bfloat16``.
    ///
    /// # Params
    /// - scale: scale the output by this factor -- default is `1.0/sqrt(array.dim(-1))`
    /// - stream: stream to evaluate on.
    #[default_device]
    pub fn hadamard_transform_device(
        &self,
        scale: impl Into<Option<f32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let scale = scale.into();
        let scale = mlx_sys::mlx_optional_float {
            value: scale.unwrap_or(0.0),
            has_value: scale.is_some(),
        };

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_hadamard_transform(res, self.as_ptr(), scale, stream.as_ref().as_ptr())
        })
    }
}

/// See [`Array::diag`]
#[default_device]
pub fn diag_device(
    a: impl AsRef<Array>,
    k: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().diag_device(k, stream)
}

/// See [`Array::diagonal`]
#[default_device]
pub fn diagonal_device(
    a: impl AsRef<Array>,
    offset: impl Into<Option<i32>>,
    axis1: impl Into<Option<i32>>,
    axis2: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().diagonal_device(offset, axis1, axis2, stream)
}

/// Perform the Einstein summation convention on the operands.
///
/// # Params
///
/// - subscripts: Einstein summation convention equation
/// - operands: input arrays
/// - stream: stream or device to evaluate on
#[default_device]
pub fn einsum_device<'a>(
    subscripts: &str,
    operands: impl IntoIterator<Item = &'a Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_subscripts =
        CString::new(subscripts).map_err(|_| Exception::from("Invalid subscripts"))?;
    let c_operands = VectorArray::try_from_iter(operands.into_iter())?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_einsum(
            res,
            c_subscripts.as_ptr(),
            c_operands.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        ops::{arange, diag, einsum, reshape},
        Array,
    };
    use pretty_assertions::assert_eq;

    use super::diagonal;

    #[test]
    fn test_diagonal() {
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[4, 2]);
        let out = diagonal(&x, None, None, None).unwrap();
        assert_eq!(out, array!([0, 3]));

        assert!(diagonal(&x, 1, 6, 0).is_err());
        assert!(diagonal(&x, 1, 0, -3).is_err());

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let out = diagonal(&x, 2, 1, 0).unwrap();
        assert_eq!(out, array!([8]));

        let out = diagonal(&x, -1, 0, 1).unwrap();
        assert_eq!(out, array!([4, 9]));

        let out = diagonal(&x, -5, 0, 1).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[0]);

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 2, 2]);
        let out = diagonal(&x, 1, 0, 1).unwrap();
        assert_eq!(out, array!([[2], [3]]));

        let out = diagonal(&x, 0, 2, 0).unwrap();
        assert_eq!(out, array!([[0, 5], [2, 7]]));

        let out = diagonal(&x, 1, -1, 0).unwrap();
        assert_eq!(out, array!([[4, 9], [6, 11]]));

        let x = reshape(&arange::<f32, _>(None, 16, None).unwrap(), &[2, 2, 2, 2]).unwrap();
        let out = diagonal(&x, 0, 0, 1).unwrap();
        assert_eq!(
            out,
            Array::from_slice(&[0, 12, 1, 13, 2, 14, 3, 15], &[2, 2, 2])
        );

        assert!(diagonal(&x, 0, 1, 1).is_err());

        let x = array!([0, 1]);
        assert!(diagonal(&x, 0, 0, 1).is_err());
    }

    #[test]
    fn test_diag() {
        // Too few or too many dimensions
        assert!(diag(&Array::from_float(0.0), None).is_err());
        assert!(diag(&Array::from_slice(&[0.0], &[1, 1, 1]), None).is_err());

        // Test with 1D array
        let x = array!([0, 1, 2, 3]);
        let out = diag(&x, 0).unwrap();
        assert_eq!(
            out,
            array!([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]])
        );

        let out = diag(&x, 1).unwrap();
        assert_eq!(
            out,
            array!([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 3],
                [0, 0, 0, 0, 0]
            ])
        );

        let out = diag(&x, -1).unwrap();
        assert_eq!(
            out,
            array!([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 3, 0]
            ])
        );

        // Test with 2D array
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8], &[3, 3]);
        let out = diag(&x, 0).unwrap();
        assert_eq!(out, array!([0, 4, 8]));

        let out = diag(&x, 1).unwrap();
        assert_eq!(out, array!([1, 5]));

        let out = diag(&x, -1).unwrap();
        assert_eq!(out, array!([3, 7]));
    }

    #[test]
    fn test_einsum() {
        // Test dot product (vector-vector)
        let a = array!([0.0, 1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0, 6.0, 7.0]);
        let out = einsum("i,i->", &[a, b]).unwrap();
        assert_eq!(out, array!(38.0));

        // Test trace (diagonal sum)
        let m = array!([[1, 2], [3, 4]]);
        let out = einsum("ii->", &[m]).unwrap();
        assert_eq!(out, array!(5.0));
    }

    #[test]
    fn test_hadamard_transform() {
        let input = Array::from_slice(&[1.0, -1.0, -1.0, 1.0], &[2, 2]);
        let expected = Array::from_slice(
            &[
                0.0,
                2.0_f32 / 2.0_f32.sqrt(),
                0.0,
                -2.0_f32 / 2.0_f32.sqrt(),
            ],
            &[2, 2],
        );
        let result = input.hadamard_transform(None).unwrap();

        let c = result.all_close(&expected, 1e-5, 1e-5, None).unwrap();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }
}
