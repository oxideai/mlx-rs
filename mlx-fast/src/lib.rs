use mlx_internal_macros::default_device;
use mlx_rs::{Array, Stream, StreamOrDevice};

// TODO: Implement custom kernels:
// https://github.com/ml-explore/mlx-swift/blob/d649c62b77c487c25012910b0d02b30283d388ca/Source/MLXFast/MLXFastKernel.swift

/// Optimized implementation of `NN.RoPE`.
#[allow(non_snake_case)]
#[default_device]
pub fn RoPE_device<'a>(
    array: impl AsRef<Array>,
    dimensions: i32,
    traditional: bool,
    base: impl Into<Option<f32>>,
    scale: f32,
    offset: i32,
    freqs: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };

    unsafe {
        let new_array = mlx_sys::mlx_fast_rope(
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(std::ptr::null_mut()),
            stream.as_ref().as_ptr(),
        );

        Array::from_ptr(new_array)
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl Into<Option<&'a Array>>,
    memory_efficient_threshold: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) {
    let memory_efficient_threshold = memory_efficient_threshold.into();
    let memory_efficient_threshold = mlx_sys::mlx_optional_int {
        value: memory_efficient_threshold.unwrap_or(0),
        has_value: memory_efficient_threshold.is_some(),
    };

    unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask.into()
                .map(|a| a.as_ptr())
                .unwrap_or(std::ptr::null_mut()),
            memory_efficient_threshold,
            stream.as_ref().as_ptr(),
        );
    }
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[default_device]
pub fn rms_norm_device<'a>(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let new_array = mlx_sys::mlx_fast_rms_norm(
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        );

        Array::from_ptr(new_array)
    }
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[default_device]
pub fn layer_norm_device<'a>(
    x: impl AsRef<Array>,
    weight: impl Into<Option<&'a Array>>,
    bias: impl Into<Option<&'a Array>>,
    eps: f32,
    stream: impl AsRef<Stream>,
) -> Array {
    unsafe {
        let new_array = mlx_sys::mlx_fast_layer_norm(
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(std::ptr::null_mut()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(std::ptr::null_mut()),
            eps,
            stream.as_ref().as_ptr(),
        );

        Array::from_ptr(new_array)
    }
}

/// Quantize the matrix `w` using the provided `scales` and `biases` and the `groupSize` and `bits` configuration.

/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.affine_quantize.html)
///
/// # Params
///
/// - w: Matrix to be quantized
/// - scales: The scales to use per `groupSize` elements of `w`
/// - biases: The biases to use per `groupSize` elements of `w`
/// - groupSize: The size of the group in `w` that shares a scale and bias. Defaults to 64 if not provided.
/// - bits: The number of bits occupied by each element in `w`. Defaults to 4 if not provided.
/// - stream: stream or device to evaluate on
///
/// @return: The quantized version of `w`.
pub fn affine_quantized(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Array {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    unsafe {
        let new_array = mlx_sys::mlx_fast_affine_quantize(
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            group_size,
            bits,
            stream.as_ref().as_ptr(),
        );

        Array::from_ptr(new_array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use mlx_rs::ops::indexing::ArrayIndexOp;
    use mlx_rs::prelude::IndexOp;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        mlx_rs::random::seed(71);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);

        let result = RoPE(a, 8, false, 10000., 1.0, 0, None);
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), mlx_rs::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.4562537670135498,
            abs <= 0.009125075340270997
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            116.80096435546875,
            abs <= 2.3360192871093752
        );
    }

    #[test]
    fn test_rms_norm() {
        mlx_rs::random::seed(103);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5);
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), mlx_rs::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.8729387521743774,
            abs <= 0.01745877504348755
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            223.47232055664062,
            abs <= 4.469446411132813
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        mlx_rs::random::seed(635);
        let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), mlx_rs::Dtype::Float32);
        assert_float_eq!(
            result.mean(None, None).unwrap().item::<f32>(),
            0.2909903824329376,
            abs <= 0.005819807648658752
        );
        assert_float_eq!(
            result.sum(None, None).unwrap().item::<f32>(),
            4.655846118927002,
            abs <= 0.09311692237854004
        );
    }
}
