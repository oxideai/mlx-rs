use mlx_rs::{error::Exception, ops::concatenate_axis, ops::zeros_dtype, Array};
use mlx_rs::ops::indexing::{IndexMutOp, IndexOp, Ellipsis};
use mlx_rs::utils::Updatable;

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate_axis(&[k, keys], -2)?);
                self.values = Some(concatenate_axis(&[v, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// Step-based KV Cache with pre-allocation (matches Python mlx-lm KVCache)
///
/// This cache pre-allocates buffers in steps of 256 tokens and uses in-place
/// slice updates, avoiding expensive concatenation on every token.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    step: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self::with_step(256)
    }

    pub fn with_step(step: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step,
        }
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let prev = self.offset;
        let keys_shape = keys.shape();
        let values_shape = values.shape();
        let num_new = keys_shape[2];

        // Check if we need to grow the buffer
        let needs_grow = match &self.keys {
            None => true,
            Some(k) => (prev + num_new) > k.shape()[2],
        };

        if needs_grow {
            let b = keys_shape[0];
            let n_kv_heads = keys_shape[1];
            let k_head_dim = keys_shape[3];
            let v_head_dim = values_shape[3];

            // Calculate new size in steps
            let n_steps = (self.step + num_new - 1) / self.step;
            let new_size = n_steps * self.step;

            let k_shape = &[b, n_kv_heads, new_size, k_head_dim];
            let v_shape = &[b, n_kv_heads, new_size, v_head_dim];

            // Use the input dtype to preserve bf16/fp16/fp32 types
            let k_dtype = keys.dtype();
            let v_dtype = values.dtype();
            let new_k = zeros_dtype(k_shape, k_dtype)?;
            let new_v = zeros_dtype(v_shape, v_dtype)?;

            match (self.keys.take(), self.values.take()) {
                (Some(old_k), Some(old_v)) => {
                    // Trim to actual used size if needed
                    let (old_k, old_v) = if prev % self.step != 0 {
                        (
                            old_k.index((Ellipsis, ..prev, ..)),
                            old_v.index((Ellipsis, ..prev, ..)),
                        )
                    } else {
                        (old_k, old_v)
                    };
                    self.keys = Some(concatenate_axis(&[old_k, new_k], 2)?);
                    self.values = Some(concatenate_axis(&[old_v, new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }

        self.offset += num_new;

        // Update slice: self.keys[..., prev:offset, :] = keys
        let k = self.keys.as_mut().unwrap();
        let v = self.values.as_mut().unwrap();
        k.index_mut((Ellipsis, prev..self.offset, ..), &keys);
        v.index_mut((Ellipsis, prev..self.offset, ..), &values);

        // Return slice up to current offset
        Ok((
            k.index((Ellipsis, ..self.offset, ..)),
            v.index((Ellipsis, ..self.offset, ..)),
        ))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}

// ============================================================================
// Updatable implementations for compile_with_state support
// ============================================================================

impl KVCache {
    /// Get reference to keys array if present
    pub fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    /// Get reference to values array if present
    pub fn values(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Get mutable reference to keys array if present
    pub fn keys_mut(&mut self) -> Option<&mut Array> {
        self.keys.as_mut()
    }

    /// Get mutable reference to values array if present
    pub fn values_mut(&mut self) -> Option<&mut Array> {
        self.values.as_mut()
    }
}

impl Updatable for KVCache {
    fn updatable_states_len(&self) -> usize {
        let mut count = 0;
        if self.keys.is_some() { count += 1; }
        if self.values.is_some() { count += 1; }
        count
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        let mut states = Vec::with_capacity(2);
        if let Some(ref k) = self.keys {
            states.push(k);
        }
        if let Some(ref v) = self.values {
            states.push(v);
        }
        states
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        let mut states = Vec::with_capacity(2);
        if let Some(ref mut k) = self.keys {
            states.push(k);
        }
        if let Some(ref mut v) = self.values {
            states.push(v);
        }
        states
    }
}

/// Wrapper for Vec<KVCache> that implements Updatable
///
/// This is needed because we can't implement Updatable for Vec<T> directly
/// due to Rust's orphan rules.
#[derive(Debug, Clone, Default)]
pub struct CacheState(pub Vec<KVCache>);

impl CacheState {
    pub fn new(caches: Vec<KVCache>) -> Self {
        Self(caches)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &KVCache> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut KVCache> {
        self.0.iter_mut()
    }
}

impl std::ops::Deref for CacheState {
    type Target = Vec<KVCache>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for CacheState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Updatable for CacheState {
    fn updatable_states_len(&self) -> usize {
        self.0.iter().map(|c| c.updatable_states_len()).sum()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        self.0.iter()
            .flat_map(|c| c.updatable_states().into_iter())
            .collect::<Vec<_>>()
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        self.0.iter_mut()
            .flat_map(|c| c.updatable_states_mut().into_iter())
            .collect::<Vec<_>>()
    }
}
