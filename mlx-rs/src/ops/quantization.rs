use mlx_internal_macros::{default_device, generate_macro};

use crate::{error::Result, utils::guard::Guarded, Array, Stream};

/// Helper to create mlx_optional_int_ from i32
fn optional_int(value: i32) -> mlx_sys::mlx_optional_int_ {
    mlx_sys::mlx_optional_int_ {
        value,
        has_value: true,
    }
}

/// Helper to create an empty mlx_optional_dtype_ (no value)
fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype_ {
    mlx_sys::mlx_optional_dtype_ {
        value: 0, // placeholder
        has_value: false,
    }
}

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
/// - `mode`: Quantization mode ("affine" or "none", default: "affine")
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'static str>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);
    let mode_str = mode.into().unwrap_or("affine");
    let mode_cstr = std::ffi::CString::new(mode_str).unwrap();

    unsafe {
        let mut res = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_quantize(
            &mut res,
            w.as_ref().as_ptr(),
            optional_int(group_size),
            optional_int(bits),
            mode_cstr.as_ptr(),
            stream.as_ref().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_vector_array_free(res);
            return Err(crate::error::Exception::custom("mlx_quantize failed").into());
        }

        let mut arr0 = mlx_sys::mlx_array_new();
        let mut arr1 = mlx_sys::mlx_array_new();
        let mut arr2 = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut arr0, res, 0);
        mlx_sys::mlx_vector_array_get(&mut arr1, res, 1);
        mlx_sys::mlx_vector_array_get(&mut arr2, res, 2);
        mlx_sys::mlx_vector_array_free(res);

        Ok((Array::from_ptr(arr0), Array::from_ptr(arr1), Array::from_ptr(arr2)))
    }
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'static str>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(false);
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);
    let mode_str = mode.into().unwrap_or("affine");
    let mode_cstr = std::ffi::CString::new(mode_str).unwrap();

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            transpose,
            optional_int(group_size),
            optional_int(bits),
            mode_cstr.as_ptr(),
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
pub fn dequantize_device(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'static str>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);
    let mode_str = mode.into().unwrap_or("affine");
    let mode_cstr = std::ffi::CString::new(mode_str).unwrap();

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            optional_int(group_size),
            optional_int(bits),
            mode_cstr.as_ptr(),
            optional_dtype_none(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Perform matrix multiplication with gathered indices.
///
/// This operation allows efficient batched matrix multiplication where different
/// rows/columns of the matrices are selected for each element. Useful for Mixture
/// of Experts models where different experts are selected per token.
///
/// # Arguments
/// * `a` - First input array
/// * `b` - Second input array
/// * `lhs_indices` - Optional indices for selecting rows from `a`
/// * `rhs_indices` - Optional indices for selecting columns from `b`
/// * `sorted_indices` - If true, indices are assumed to be sorted for optimization
#[generate_macro]
#[default_device]
pub fn gather_mm_device<'lhs, 'rhs>(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] lhs_indices: impl Into<Option<&'lhs Array>>,
    #[optional] rhs_indices: impl Into<Option<&'rhs Array>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a_ptr = a.as_ref().as_ptr();
    let b_ptr = b.as_ref().as_ptr();
    let sorted = sorted_indices.into().unwrap_or(false);

    unsafe {
        let lhs_ptr = lhs_indices
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());
        let rhs_ptr = rhs_indices
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());

        <Array as Guarded>::try_from_op(|res| {
            mlx_sys::mlx_gather_mm(
                res,
                a_ptr,
                b_ptr,
                lhs_ptr,
                rhs_ptr,
                sorted,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// Perform quantized matrix multiplication with gathered indices.
///
/// This operation allows efficient batched quantized matrix multiplication where
/// different experts are selected per token. Essential for Mixture of Experts
/// inference with quantized weights.
///
/// # Arguments
/// * `x` - Input activations array
/// * `w` - Quantized weights array (packed integers)
/// * `scales` - Quantization scales
/// * `biases` - Quantization biases
/// * `lhs_indices` - Optional indices for selecting rows from `x`
/// * `rhs_indices` - Optional indices for selecting experts from `w`
/// * `transpose` - Whether to transpose the weights
/// * `group_size` - Quantization group size (default: 64)
/// * `bits` - Bits per element (default: 4)
/// * `mode` - Quantization mode ("affine" or "none", default: "affine")
/// * `sorted_indices` - If true, indices are assumed to be sorted for optimization
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn gather_qmm_device<'lhs, 'rhs>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    #[optional] lhs_indices: impl Into<Option<&'lhs Array>>,
    #[optional] rhs_indices: impl Into<Option<&'rhs Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'static str>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x_ptr = x.as_ref().as_ptr();
    let w_ptr = w.as_ref().as_ptr();
    let scales_ptr = scales.as_ref().as_ptr();
    let biases_ptr = biases.as_ref().as_ptr();
    let transpose = transpose.into().unwrap_or(true);
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);
    let mode_str = mode.into().unwrap_or("affine");
    let mode_cstr = std::ffi::CString::new(mode_str).unwrap();
    let sorted = sorted_indices.into().unwrap_or(false);

    unsafe {
        let lhs_ptr = lhs_indices
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());
        let rhs_ptr = rhs_indices
            .into()
            .map(|m| m.as_ptr())
            .unwrap_or(mlx_sys::mlx_array_new());

        <Array as Guarded>::try_from_op(|res| {
            mlx_sys::mlx_gather_qmm(
                res,
                x_ptr,
                w_ptr,
                scales_ptr,
                biases_ptr,
                lhs_ptr,
                rhs_ptr,
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode_cstr.as_ptr(),
                sorted,
                stream.as_ref().as_ptr(),
            )
        })
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
}
