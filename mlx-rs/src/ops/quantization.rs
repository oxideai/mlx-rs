// quantized(_:groupSize:bits:stream:)
// quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)
// dequantized(_:scales:biases:groupSize:bits:stream:)

use mlx_internal_macros::default_device;

use crate::{error::{Exception, Result}, Array, Stream, StreamOrDevice};

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
#[default_device]
pub fn quantize_device(
    w: &Array,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    unsafe {
        let mut res_0 = mlx_sys::mlx_array_new();
        let mut res_1 = mlx_sys::mlx_array_new();
        let mut res_2 = mlx_sys::mlx_array_new();
        check_status! {
            mlx_sys::mlx_quantize(
                &mut res_0 as *mut _,
                &mut res_1 as *mut _,
                &mut res_2 as *mut _,
                w.as_ptr(), group_size, bits, stream.as_ref().as_ptr()),
            {
                mlx_sys::mlx_array_free(res_0);
                mlx_sys::mlx_array_free(res_1);
                mlx_sys::mlx_array_free(res_2);
            }
        };

        Ok((Array::from_ptr(res_0), Array::from_ptr(res_1), Array::from_ptr(res_2)))
    }
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[default_device]
pub fn quantized_matmul_device(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: impl Into<Option<bool>>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(false);
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    unsafe {
        let mut c_array = mlx_sys::mlx_array_new();
        check_status! {
            mlx_sys::mlx_quantized_matmul(
                &mut c_array as *mut _,
                x.as_ptr(),
                w.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                transpose,
                group_size,
                bits,
                stream.as_ref().as_ptr()
            ),
            mlx_sys::mlx_array_free(c_array)
        };

        Ok(Array::from_ptr(c_array))
    }
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[default_device]
pub fn dequantize_device(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    unsafe {
        let mut c_array = mlx_sys::mlx_array_new();
        check_status! {
            mlx_sys::mlx_dequantize(
                &mut c_array as *mut _,
                w.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                group_size,
                bits,
                stream.as_ref().as_ptr()
            ),
            mlx_sys::mlx_array_free(c_array)
        };

        Ok(Array::from_ptr(c_array))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{dequantize, expand_dims, quantize},
        Array,
    };

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(&Array::arange::<f32, _>(0, 512, None).unwrap(), &[0]).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i).unwrap();
            let max_diff = ((&x - &x_hat).abs().max(None, None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }
}
