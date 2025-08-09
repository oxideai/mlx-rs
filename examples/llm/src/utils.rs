use mlx_rs::{error::Exception, fast::ScaledDotProductAttentionMask, Array};

use crate::cache_utils::KeyValueCache;

pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    // Implement the quantized attention logic here
    todo!()
}

pub(crate) fn scaled_dot_product_attention<C>(
    queries: Array,
    keys: Array,
    values: Array,
    cache: Option<C>,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    if let Some(cache) = cache {
        if cache.is_quantized() {
            let group_size = cache
                .group_size()
                .ok_or_else(|| Exception::custom("Cache is quantized but group size is not set"))?;
            let bits = cache
                .bits()
                .ok_or_else(|| Exception::custom("Cache is quantized but bits are not set"))?;
            return quantized_scaled_dot_product_attention(queries, keys, values, scale, mask, group_size, bits)
        } 
    }

    mlx_rs::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
    )
}
