use mlx_rs::{
    arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

use crate::cache::KeyValueCache;

pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
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
            return quantized_scaled_dot_product_attention(
                queries, keys, values, scale, mask, group_size, bits,
            );
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

#[derive(Debug, Clone)]
pub(crate) enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N)?;
    let linds = arange!(start = offset, stop = offset + N)?;
    let linds = linds.index((.., NewAxis));
    let rinds = rinds.index(NewAxis);

    let mut mask = linds.ge(&rinds)?;
    if let Some(window_size) = window_size {
        mask = mask.logical_and(&linds.le(&(rinds + window_size))?)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.index((.., NewAxis, NewAxis, NewAxis));
        mask = mask.logical_and(&linds.lt(&lengths)?)?;
    }

    Ok(mask)
}

#[allow(non_snake_case)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    return_array: Option<bool>,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let mut return_array = return_array.unwrap_or(false);
    let T = h.shape()[1];
    if T > 1 {
        let mut offset = 0;
        let mut window_size = None;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            offset = c.offset();
            if let Some(window_size_) = c.max_size() {
                window_size = Some(window_size_);
                offset = offset.min(window_size_);

                return_array = return_array || (offset + T) > window_size_;
            }
        }

        if return_array {
            create_causal_mask(T, Some(offset), window_size, None)
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}
