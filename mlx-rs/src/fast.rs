use mlx_sys::{utils::StreamOrDevice, Optional};

use crate::array::Array;

pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    s: StreamOrDevice,
) -> Array {
    let result = mlx_sys::fast::ffi::rope(x.as_ref(), dims, traditional, base, scale, offset, s);
    Array {
        inner: result.unwrap(),
    }
}

pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: &Optional<Array>,
    s: StreamOrDevice,
) -> Array {
    let mask = unsafe {
        // SAFETY: `Array` is a transparent wrapper around `UniquePtr<array>`.
        std::mem::transmute(mask)
    };

    let result = mlx_sys::fast::ffi::scaled_dot_product_attention(
        queries.as_ref(),
        keys.as_ref(),
        values.as_ref(),
        scale,
        mask,
        s,
    );
    Array {
        inner: result.unwrap(),
    }
}

pub fn rms_norm(
    x: &Array,
    weight: &Array,
    eps: f32,
    s: StreamOrDevice,
) -> Array {
    let result = mlx_sys::fast::ffi::rms_norm(x.as_ref(), weight.as_ref(), eps, s);
    Array {
        inner: result.unwrap(),
    }
}

pub fn layer_norm(
    x: &Array,
    weight: &Optional<Array>,
    bias: &Optional<Array>,
    eps: f32,
    s: StreamOrDevice,
) -> Array {
    let weight = unsafe {
        // SAFETY: `Array` is a transparent wrapper around `UniquePtr<array>`.
        std::mem::transmute(weight)
    };
    let bias = unsafe {
        // SAFETY: `Array` is a transparent wrapper around `UniquePtr<array>`.
        std::mem::transmute(bias)
    };

    let result = mlx_sys::fast::ffi::layer_norm(x.as_ref(), weight, bias, eps, s);
    Array {
        inner: result.unwrap(),
    }
}