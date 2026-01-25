use std::ffi::CStr;

use mlx_internal_macros::{default_device, generate_macro};

use crate::{
    error::Result,
    utils::{guard::Guarded, VectorArray},
    Array, Stream,
};

const DEFAULT_MODE: &CStr = c"affine";

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// # Params
///
/// - `w`: The input matrix
/// - `group_size`: The size of the group in `w` that shares a scale and bias. (default: `64`)
/// - `bits`: The number of bits occupied by each element of w in the returned quantized matrix.
///   (default: 4)
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    let result = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            DEFAULT_MODE.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })?;

    let arrays: Vec<Array> = result.try_into_values()?;
    if arrays.len() != 3 {
        return Err(crate::error::Exception::custom(format!(
            "Expected 3 arrays from quantize, got {}",
            arrays.len()
        )));
    }
    let mut iter = arrays.into_iter();
    Ok((
        iter.next().unwrap(),
        iter.next().unwrap(),
        iter.next().unwrap(),
    ))
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(false);
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            transpose,
            group_size,
            bits,
            DEFAULT_MODE.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[generate_macro]
#[default_device]
pub fn dequantize_device<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            group_size,
            bits,
            DEFAULT_MODE.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{dequantize, expand_dims, quantize, quantized_matmul},
        random, Array,
    };

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i).unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    // Test adapted from Python test `test_quantized.py/test_qmm`
    #[test]
    fn test_quantized_matmul() {
        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;
        let m = 32;
        let n = 128;
        let k = 128;

        let scale = 1.0 / (k as f32).sqrt();
        let x = random::normal::<f32>(&[m, k], None, None, None).unwrap() * scale;
        let w = random::normal::<f32>(&[k, n], None, None, None).unwrap() * scale;

        let (w_q, scales, biases) = quantize(&w, group_size, bits).unwrap();
        let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits).unwrap();

        // Test with biases
        let y_q = quantized_matmul(&x, &w_q, &scales, &biases, false, group_size, bits).unwrap();
        let y_hat = x.matmul(&w_hat).unwrap();

        assert_eq!(y_q.shape(), y_hat.shape());
        let max_diff = ((&y_q - &y_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
        assert!(max_diff < 1e-3, "max_diff: {}", max_diff);
    }
}
