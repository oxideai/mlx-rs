//! Text2Semantic (T2S) model for GPT-SoVITS
//!
//! This model converts text (phonemes + BERT features) to semantic tokens.
//! Architecture based on dora-primespeech Text2SemanticDecoder.
//!
//! Key characteristics:
//! - 24 transformer layers, 512 hidden size, 16 heads
//! - Combined QKV projection (in_proj) instead of separate Q/K/V
//! - LayerNorm instead of RmsNorm
//! - Dual embeddings: phoneme (732 vocab) + semantic (1025 vocab)
//! - BERT feature projection (1024 -> 512)
//! - Sinusoidal position encoding with learned alpha scaling

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        indexing::{IndexOp, NewAxis},
        softmax_axis, concatenate_axis, tril, argpartition_axis,
    },
    Array,
};
use serde::Deserialize;

use crate::{cache::KeyValueCache, error::Error};

/// Configuration for T2S model
#[derive(Debug, Clone, Deserialize)]
pub struct T2SConfig {
    /// Hidden dimension (512)
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Number of transformer layers (24)
    #[serde(default = "default_num_layers")]
    pub num_layers: i32,
    /// Number of attention heads (16)
    #[serde(default = "default_num_heads")]
    pub num_heads: i32,
    /// FFN intermediate size (2048)
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Phoneme vocabulary size (732)
    #[serde(default = "default_phoneme_vocab_size")]
    pub phoneme_vocab_size: i32,
    /// Semantic token vocabulary size (1025, includes EOS at 1024)
    #[serde(default = "default_semantic_vocab_size")]
    pub semantic_vocab_size: i32,
    /// BERT feature dimension (1024)
    #[serde(default = "default_bert_dim")]
    pub bert_dim: i32,
    /// EOS token ID (1024)
    #[serde(default = "default_eos_token")]
    pub eos_token: i32,
    /// Layer norm epsilon
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_hidden_size() -> i32 { 512 }
fn default_num_layers() -> i32 { 24 }
fn default_num_heads() -> i32 { 16 }
fn default_intermediate_size() -> i32 { 2048 }
fn default_phoneme_vocab_size() -> i32 { 732 }
fn default_semantic_vocab_size() -> i32 { 1025 }
fn default_bert_dim() -> i32 { 1024 }
fn default_eos_token() -> i32 { 1024 }
fn default_layer_norm_eps() -> f32 { 1e-5 }

impl Default for T2SConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_layers: default_num_layers(),
            num_heads: default_num_heads(),
            intermediate_size: default_intermediate_size(),
            phoneme_vocab_size: default_phoneme_vocab_size(),
            semantic_vocab_size: default_semantic_vocab_size(),
            bert_dim: default_bert_dim(),
            eos_token: default_eos_token(),
            layer_norm_eps: default_layer_norm_eps(),
        }
    }
}

impl T2SConfig {
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }
}

/// Sinusoidal Position Encoding with learned alpha scaling
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// The embedding is: x + alpha * PE(pos)
#[derive(Debug, Clone)]
pub struct SinusoidalPositionEncoding {
    /// Learned scaling factor
    pub alpha: f32,
    /// Hidden dimension
    pub hidden_size: i32,
    /// Maximum sequence length for precomputed PE
    pub max_seq_len: i32,
}

impl SinusoidalPositionEncoding {
    pub fn new(hidden_size: i32, alpha: f32, max_seq_len: i32) -> Self {
        Self {
            alpha,
            hidden_size,
            max_seq_len,
        }
    }

    /// Generate sinusoidal position encoding for given positions
    ///
    /// Following PyTorch implementation:
    ///   pe[:, 0::2] = sin(position * div_term)
    ///   pe[:, 1::2] = cos(position * div_term)
    /// Where div_term = exp(arange(0, d, 2) * -log(10000) / d)
    ///
    /// Returns:
    ///   Position encodings [seq_len, hidden] scaled by alpha
    pub fn forward(&self, seq_len: i32, offset: i32) -> Result<Array, Exception> {
        let hidden = self.hidden_size;
        let half_dim = hidden / 2;

        // Create position indices: [seq_len, 1]
        let positions: Vec<f32> = (offset..(offset + seq_len)).map(|p| p as f32).collect();
        let pos = Array::from_slice(&positions, &[seq_len, 1]);

        // Create dimension indices for sin/cos: [1, half_dim]
        // div_term = exp(arange(0, hidden, 2) * -(log(10000) / hidden))
        let log_10000 = 10000.0_f32.ln();
        let div_terms: Vec<f32> = (0..half_dim)
            .map(|i| (-log_10000 * (2 * i) as f32 / hidden as f32).exp())
            .collect();
        let div_term = Array::from_slice(&div_terms, &[1, half_dim]);

        // Compute angles: [seq_len, half_dim]
        let angles = pos.matmul(&div_term)?;

        // Compute sin and cos: both [seq_len, half_dim]
        let sin_enc = angles.sin()?;
        let cos_enc = angles.cos()?;

        // Interleave sin and cos: [seq_len, hidden]
        // PE[:, 0::2] = sin, PE[:, 1::2] = cos
        // Stack and reshape to interleave: [seq_len, half_dim, 2] -> [seq_len, hidden]
        let sin_expanded = sin_enc.reshape(&[seq_len, half_dim, 1])?;
        let cos_expanded = cos_enc.reshape(&[seq_len, half_dim, 1])?;
        let stacked = concatenate_axis(&[&sin_expanded, &cos_expanded], -1)?; // [seq_len, half_dim, 2]
        let pe = stacked.reshape(&[seq_len, hidden])?; // [seq_len, hidden]

        // Apply alpha scaling
        pe.multiply(array!(self.alpha))
    }

    /// Apply position encoding to embeddings
    ///
    /// Args:
    ///   x: [batch, seq_len, hidden] embeddings
    ///   offset: Position offset (for decode phase)
    ///
    /// Returns:
    ///   x + alpha * PE
    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let seq_len = x.shape()[1] as i32;
        let pe = self.forward(seq_len, offset)?;
        // Broadcast PE [seq_len, hidden] to [batch, seq_len, hidden]
        x.add(&pe)
    }
}

/// Self-attention with combined QKV projection
///
/// T2S uses a single in_proj weight matrix that combines Q, K, V projections.
/// Shape: (3 * hidden_size, hidden_size) = (1536, 512)
#[derive(Debug, Clone, ModuleParameters)]
pub struct T2SAttention {
    pub n_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    /// Combined QKV projection (3*hidden, hidden)
    #[param]
    pub in_proj: nn::Linear,

    /// Output projection (hidden, hidden)
    #[param]
    pub out_proj: nn::Linear,
}

impl T2SAttention {
    pub fn new(config: &T2SConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let n_heads = config.num_heads;
        let head_dim = config.head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        // Combined QKV projection: (3*hidden, hidden) with bias
        let in_proj = nn::LinearBuilder::new(hidden_size, 3 * hidden_size)
            .bias(true)
            .build()?;

        // Output projection: (hidden, hidden) with bias
        let out_proj = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(true)
            .build()?;

        Ok(Self {
            n_heads,
            head_dim,
            scale,
            in_proj,
            out_proj,
        })
    }
}

/// Input for T2S attention
pub struct T2SAttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<T2SAttentionInput<'_, C>> for T2SAttention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: T2SAttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let T2SAttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Combined QKV projection
        let qkv = self.in_proj.forward(x)?;

        // Split into Q, K, V
        // qkv shape: (B, L, 3*hidden)
        let hidden = self.n_heads * self.head_dim;
        let q = qkv.index((.., .., 0..hidden));
        let k = qkv.index((.., .., hidden..(2 * hidden)));
        let v = qkv.index((.., .., (2 * hidden)..));

        // Reshape for multi-head attention: (B, n_heads, L, head_dim)
        let queries = q
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = k
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = v
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Update KV cache if provided
        if let Some(cache) = cache.as_mut() {
            (keys, values) = cache.update_and_fetch(keys, values)?;
        }

        // Scaled dot-product attention
        // scores = Q @ K^T / sqrt(d_k)
        let scores = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(array!(self.scale))?;

        // Apply attention mask if provided
        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        // Softmax and apply to values
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let output = attn_weights.matmul(&values)?;

        // Reshape back: (B, n_heads, L, head_dim) -> (B, L, hidden)
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.out_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

/// Feed-forward network with GELU activation
///
/// Standard FFN: Linear -> GELU -> Linear
#[derive(Debug, Clone, ModuleParameters)]
pub struct T2SFFN {
    #[param]
    pub linear1: nn::Linear,
    #[param]
    pub linear2: nn::Linear,
}

impl T2SFFN {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let linear1 = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(true)
            .build()?;
        let linear2 = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(true)
            .build()?;

        Ok(Self { linear1, linear2 })
    }
}

impl Module<&Array> for T2SFFN {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.linear1.forward(x)?;
        // GPT-SoVITS uses ReLU, not GELU
        let h = nn::relu(&h)?;
        self.linear2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear1.training_mode(mode);
        self.linear2.training_mode(mode);
    }
}

/// Transformer block for T2S
///
/// Post-norm architecture with LayerNorm (GPT-SoVITS style)
/// x = x + attn; x = LN(x); x = x + ffn; x = LN(x)
#[derive(Debug, Clone, ModuleParameters)]
pub struct T2STransformerBlock {
    #[param]
    pub self_attn: T2SAttention,
    #[param]
    pub ffn: T2SFFN,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
}

impl T2STransformerBlock {
    pub fn new(config: &T2SConfig) -> Result<Self, Exception> {
        let self_attn = T2SAttention::new(config)?;
        let ffn = T2SFFN::new(config.hidden_size, config.intermediate_size)?;

        let norm1 = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;
        let norm2 = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
        })
    }
}

impl<C> Module<T2SAttentionInput<'_, C>> for T2STransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: T2SAttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let T2SAttentionInput { x, mask, cache } = input;

        // GPT-SoVITS uses POST-LN: x = x + attn; x = LN(x); x = x + ffn; x = LN(x)
        // Self-attention with residual, then layer norm
        let attn_input = T2SAttentionInput {
            x,
            mask,
            cache,
        };
        let attn_out = self.self_attn.forward(attn_input)?;
        let h = x.add(&attn_out)?;
        let h = self.norm1.forward(&h)?;

        // FFN with residual, then layer norm
        let ffn_out = self.ffn.forward(&h)?;
        let h = h.add(&ffn_out)?;
        self.norm2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        <T2SAttention as Module<T2SAttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.ffn.training_mode(mode);
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
    }
}

/// Text2Semantic Model
///
/// Converts phoneme IDs + BERT features to semantic tokens autoregressively.
#[derive(Debug, Clone, ModuleParameters)]
pub struct T2SModel {
    pub config: T2SConfig,

    /// Phoneme token embedding
    #[param]
    pub phoneme_embedding: nn::Embedding,

    /// Semantic token embedding
    #[param]
    pub semantic_embedding: nn::Embedding,

    /// BERT feature projection (1024 -> 512)
    #[param]
    pub bert_proj: nn::Linear,

    /// Transformer layers
    #[param]
    pub layers: Vec<T2STransformerBlock>,

    /// Output prediction layer
    #[param]
    pub predict_layer: nn::Linear,

    /// Position encoding for text (phoneme + BERT)
    pub text_position: SinusoidalPositionEncoding,

    /// Position encoding for audio (semantic tokens)
    pub audio_position: SinusoidalPositionEncoding,
}

impl T2SModel {
    pub fn new(config: T2SConfig) -> Result<Self, Exception> {
        let phoneme_embedding =
            nn::Embedding::new(config.phoneme_vocab_size, config.hidden_size)?;
        let semantic_embedding =
            nn::Embedding::new(config.semantic_vocab_size, config.hidden_size)?;

        let bert_proj = nn::LinearBuilder::new(config.bert_dim, config.hidden_size)
            .bias(true)  // The actual model has bias
            .build()?;

        let layers = (0..config.num_layers)
            .map(|_| T2STransformerBlock::new(&config))
            .collect::<Result<Vec<_>, _>>()?;

        let predict_layer = nn::LinearBuilder::new(config.hidden_size, config.semantic_vocab_size)
            .bias(false)
            .build()?;

        // Initialize position encodings with default alpha values
        // These will be overwritten when loading weights
        let text_position = SinusoidalPositionEncoding::new(
            config.hidden_size,
            3.8242,  // Default from GPT-SoVITS
            4096,
        );
        let audio_position = SinusoidalPositionEncoding::new(
            config.hidden_size,
            3.4824,  // Default from GPT-SoVITS
            4096,
        );

        Ok(Self {
            config,
            phoneme_embedding,
            semantic_embedding,
            bert_proj,
            layers,
            predict_layer,
            text_position,
            audio_position,
        })
    }

    /// Create a causal attention mask
    pub fn create_causal_mask(&self, seq_len: i32) -> Result<Array, Exception> {
        // Upper triangular mask with -inf for future positions
        // Use where() to avoid NaN from 0 * -inf
        let ones = Array::ones::<f32>(&[seq_len, seq_len])?;
        let zeros = Array::zeros::<f32>(&[seq_len, seq_len])?;
        // Create lower triangular matrix (1s on and below diagonal = can attend)
        let lower = tril(&ones, Some(0))?;
        // Use where: if lower==1, use 0 (can attend), else use -inf (mask)
        let neg_inf = Array::full::<f32>(&[seq_len, seq_len], array!(f32::NEG_INFINITY))?;
        // lower > 0.5 gives boolean mask for attended positions
        let can_attend = lower.gt(array!(0.5f32))?;
        mlx_rs::ops::r#where(&can_attend, &zeros, &neg_inf)
    }

    /// Create T2S-style attention mask (GPT-SoVITS)
    ///
    /// Text tokens: bidirectional attention to text, masked from audio
    /// Audio tokens: can attend to all text, causal for audio
    ///
    /// This creates a mask like:
    /// ```
    /// Text rows:  [0, 0, ..., -inf, -inf]  (attend to text, mask audio)
    /// Audio rows: [0, 0, ..., causal    ]  (attend to text + causal audio)
    /// ```
    pub fn create_t2s_mask(&self, text_len: i32, audio_len: i32) -> Result<Array, Exception> {
        let total_len = text_len + audio_len;

        // Text rows: attend to text (0), masked from audio (-inf)
        let text_to_text = Array::zeros::<f32>(&[text_len, text_len])?;
        let text_to_audio = Array::full::<f32>(&[text_len, audio_len], array!(f32::NEG_INFINITY))?;
        let text_mask = concatenate_axis(&[&text_to_text, &text_to_audio], 1)?;

        // Audio rows: attend to text (0), causal for audio
        let audio_to_text = Array::zeros::<f32>(&[audio_len, text_len])?;

        // Causal mask for audio-to-audio
        let ones = Array::ones::<f32>(&[audio_len, audio_len])?;
        let zeros = Array::zeros::<f32>(&[audio_len, audio_len])?;
        let lower = tril(&ones, Some(0))?;
        let neg_inf = Array::full::<f32>(&[audio_len, audio_len], array!(f32::NEG_INFINITY))?;
        let can_attend = lower.gt(array!(0.5f32))?;
        let audio_causal = mlx_rs::ops::r#where(&can_attend, &zeros, &neg_inf)?;

        let audio_mask = concatenate_axis(&[&audio_to_text, &audio_causal], 1)?;

        // Combine text and audio masks
        concatenate_axis(&[&text_mask, &audio_mask], 0)
    }
}

/// Input for T2S model forward pass
pub struct T2SInput<'a, C> {
    /// Phoneme token IDs: (batch, phoneme_seq_len)
    pub phoneme_ids: &'a Array,
    /// Semantic token IDs: (batch, semantic_seq_len)
    pub semantic_ids: &'a Array,
    /// BERT features: (batch, text_seq_len, bert_dim)
    pub bert_features: &'a Array,
    /// KV cache for each layer
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<T2SInput<'_, C>> for T2SModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: T2SInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let T2SInput {
            phoneme_ids,
            semantic_ids,
            bert_features,
            cache,
        } = input;

        // Check if this is prefill (cache not yet initialized) or decode (cache populated)
        // Cache is Vec<Option<C>>: empty vec or first element is None = prefill
        let is_prefill = cache.is_empty() || cache.first().map_or(true, |c| c.is_none());

        let mut h;
        let mask;

        if is_prefill {
            // Prefill: Process full context (text + semantic)
            //
            // Python does:
            //   x = self.ar_text_embedding(x)
            //   x = x + self.bert_proj(bert_feature.transpose(1, 2))
            //   x = self.ar_text_position(x)
            // BERT features are ADDED to phoneme embeddings, not concatenated!

            // Embed phonemes: (B, text_len, hidden)
            let phoneme_emb = self.phoneme_embedding.forward(phoneme_ids)?;
            let text_len = phoneme_emb.shape()[1] as i32;

            // Project and ADD BERT features: (B, text_len, hidden)
            let bert_proj = self.bert_proj.forward(bert_features)?;
            let text_emb = phoneme_emb.add(&bert_proj)?;

            // Apply text position encoding to combined text embedding
            let text_emb = self.text_position.apply(&text_emb, 0)?;

            // Embed semantic tokens: (B, semantic_len, hidden)
            let semantic_emb = self.semantic_embedding.forward(semantic_ids)?;
            let semantic_len = semantic_emb.shape()[1] as i32;

            // Apply audio position encoding starting at position 0
            let semantic_emb = self.audio_position.apply(&semantic_emb, 0)?;

            // Concatenate: text + semantic
            h = concatenate_axis(&[&text_emb, &semantic_emb], 1)?;

            // Create T2S-style mask for prefill:
            // - Text tokens: bidirectional to text, masked from audio
            // - Audio tokens: attend to all text, causal for audio
            mask = Some(self.create_t2s_mask(text_len, semantic_len)?);

            // Initialize cache
            *cache = (0..self.layers.len())
                .map(|_| Some(C::default()))
                .collect();
        } else {
            // Decode: Only process new semantic token(s)
            // The text context is already in the KV cache
            let semantic_emb = self.semantic_embedding.forward(semantic_ids)?;

            // Get current audio position from cache length
            // Cache contains [text + previous_semantic] tokens
            // Text length = phoneme length (BERT is added, not concatenated)
            let cache_len = cache.first()
                .and_then(|c| c.as_ref())
                .map(|c| c.offset())
                .unwrap_or(0);
            let text_len = phoneme_ids.shape()[1] as i32;
            let audio_offset = cache_len - text_len;

            // Apply audio position encoding at current position
            h = self.audio_position.apply(&semantic_emb, audio_offset)?;

            // No mask needed for single token (L=1), causal is implicit
            mask = None;
        }

        // Process through transformer layers
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = T2SAttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
        }

        // Project to vocabulary (all positions in decode, last semantic in prefill)
        self.predict_layer.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.phoneme_embedding.training_mode(mode);
        self.semantic_embedding.training_mode(mode);
        self.bert_proj.training_mode(mode);
        for layer in &mut self.layers {
            <T2STransformerBlock as Module<T2SAttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.predict_layer.training_mode(mode);
    }
}

/// Sample from logits with temperature
pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

/// Sample with top-k filtering
pub fn sample_top_k(logits: &Array, temp: f32, top_k: i32) -> Result<Array, Exception> {
    if top_k <= 0 || top_k >= logits.shape().last().copied().unwrap_or(0) as i32 {
        return sample(logits, temp);
    }

    // For top-k, we want the k largest values
    // argpartition with negative k gives us indices such that smallest k are partitioned
    // So we need vocab_size - top_k as kth to get top_k largest at the end
    let vocab_size = logits.shape().last().copied().unwrap_or(0) as i32;
    let kth = vocab_size - top_k;

    let all_indices = argpartition_axis(logits, kth, -1)?;
    let top_k_indices = all_indices.index((.., kth..));
    let top_k_logits = logits.take_along_axis(&top_k_indices, -1)?;

    // Apply temperature and sample
    let scaled = top_k_logits.multiply(array!(1.0 / temp))?;
    let idx = categorical!(scaled)?;

    // Map back to original vocabulary indices
    // take_along_axis returns (batch, 1), we want (batch,)
    top_k_indices.take_along_axis(&idx.index((.., NewAxis)), -1)?
        .squeeze()
}

/// Load T2S model weights from PyTorch checkpoint
pub fn load_t2s_weights(model: &mut T2SModel, weights: &HashMap<String, Array>) -> Result<(), Error> {
    // Helper to get weight with fallback names
    let get_weight = |keys: &[&str]| -> Result<Array, Error> {
        for key in keys {
            if let Some(w) = weights.get(*key) {
                return Ok(w.clone());
            }
        }
        Err(Error::Message(format!("Weight not found: {:?}", keys)))
    };

    // Load embeddings - handle both naming conventions
    model.phoneme_embedding.weight = Param::new(get_weight(&[
        "phoneme_embed.weight",
        "model.ar_text_embedding.word_embeddings.weight",
    ])?);
    model.semantic_embedding.weight = Param::new(get_weight(&[
        "semantic_embed.weight",
        "model.ar_audio_embedding.word_embeddings.weight",
    ])?);

    // Load BERT projection - not present in converted weights, skip if missing
    if let Ok(w) = get_weight(&["audio_proj.weight", "model.bert_proj.weight"]) {
        model.bert_proj.weight = Param::new(w);
    }
    if let Ok(b) = get_weight(&["audio_proj.bias", "model.bert_proj.bias"]) {
        model.bert_proj.bias = Param::new(Some(b));
    }

    // Load layers
    for (i, layer) in model.layers.iter_mut().enumerate() {
        // New naming: layers.{i}.self_attn.{q,k,v}_proj
        // Old naming: model.h.layers.{i}.self_attn.in_proj_weight

        // Try new naming with separate Q/K/V first
        let q_key = format!("layers.{}.self_attn.q_proj.weight", i);
        let k_key = format!("layers.{}.self_attn.k_proj.weight", i);
        let v_key = format!("layers.{}.self_attn.v_proj.weight", i);

        if weights.contains_key(&q_key) {
            // Concatenate Q, K, V into combined QKV
            let q = weights.get(&q_key).unwrap().clone();
            let k = weights.get(&k_key).unwrap().clone();
            let v = weights.get(&v_key).unwrap().clone();
            let qkv = concatenate_axis(&[&q, &k, &v], 0)?;
            layer.self_attn.in_proj.weight = Param::new(qkv);

            // Biases
            let q_bias_key = format!("layers.{}.self_attn.q_proj.bias", i);
            let k_bias_key = format!("layers.{}.self_attn.k_proj.bias", i);
            let v_bias_key = format!("layers.{}.self_attn.v_proj.bias", i);
            if let (Some(qb), Some(kb), Some(vb)) = (
                weights.get(&q_bias_key),
                weights.get(&k_bias_key),
                weights.get(&v_bias_key),
            ) {
                let qkv_bias = concatenate_axis(&[qb, kb, vb], 0)?;
                layer.self_attn.in_proj.bias = Param::new(Some(qkv_bias));
            }
        } else {
            // Try old naming
            let prefix = format!("model.h.layers.{}", i);
            layer.self_attn.in_proj.weight =
                Param::new(get_weight(&[&format!("{}.self_attn.in_proj_weight", prefix)])?);
            if let Ok(bias) = get_weight(&[&format!("{}.self_attn.in_proj_bias", prefix)]) {
                layer.self_attn.in_proj.bias = Param::new(Some(bias));
            }
        }

        // Output projection
        let o_key = format!("layers.{}.self_attn.o_proj.weight", i);
        let o_old = format!("model.h.layers.{}.self_attn.out_proj.weight", i);
        layer.self_attn.out_proj.weight = Param::new(get_weight(&[&o_key, &o_old])?);
        if let Ok(bias) = get_weight(&[
            &format!("layers.{}.self_attn.o_proj.bias", i),
            &format!("model.h.layers.{}.self_attn.out_proj.bias", i),
        ]) {
            layer.self_attn.out_proj.bias = Param::new(Some(bias));
        }

        // FFN - new naming uses gate_proj/down_proj, old uses linear1/linear2
        layer.ffn.linear1.weight = Param::new(get_weight(&[
            &format!("layers.{}.mlp.gate_proj.weight", i),
            &format!("model.h.layers.{}.linear1.weight", i),
        ])?);
        if let Ok(bias) = get_weight(&[
            &format!("layers.{}.mlp.gate_proj.bias", i),
            &format!("model.h.layers.{}.linear1.bias", i),
        ]) {
            layer.ffn.linear1.bias = Param::new(Some(bias));
        }

        layer.ffn.linear2.weight = Param::new(get_weight(&[
            &format!("layers.{}.mlp.down_proj.weight", i),
            &format!("model.h.layers.{}.linear2.weight", i),
        ])?);
        if let Ok(bias) = get_weight(&[
            &format!("layers.{}.mlp.down_proj.bias", i),
            &format!("model.h.layers.{}.linear2.bias", i),
        ]) {
            layer.ffn.linear2.bias = Param::new(Some(bias));
        }

        // LayerNorms
        layer.norm1.weight = Param::new(Some(get_weight(&[
            &format!("layers.{}.input_layernorm.weight", i),
            &format!("model.h.layers.{}.norm1.weight", i),
        ])?));
        if let Ok(bias) = get_weight(&[
            &format!("layers.{}.input_layernorm.bias", i),
            &format!("model.h.layers.{}.norm1.bias", i),
        ]) {
            layer.norm1.bias = Param::new(Some(bias));
        }

        layer.norm2.weight = Param::new(Some(get_weight(&[
            &format!("layers.{}.post_attention_layernorm.weight", i),
            &format!("model.h.layers.{}.norm2.weight", i),
        ])?));
        if let Ok(bias) = get_weight(&[
            &format!("layers.{}.post_attention_layernorm.bias", i),
            &format!("model.h.layers.{}.norm2.bias", i),
        ]) {
            layer.norm2.bias = Param::new(Some(bias));
        }
    }

    // Load prediction layer
    model.predict_layer.weight = Param::new(get_weight(&[
        "lm_head.weight",
        "model.ar_predict_layer.weight",
    ])?);

    // Load position encoding alpha values
    if let Ok(alpha) = get_weight(&["model.ar_text_position.alpha"]) {
        let alpha_val: f32 = alpha.item();
        model.text_position.alpha = alpha_val;
    }
    if let Ok(alpha) = get_weight(&["model.ar_audio_position.alpha"]) {
        let alpha_val: f32 = alpha.item();
        model.audio_position.alpha = alpha_val;
    }

    Ok(())
}

/// Load T2S model from checkpoint directory
pub fn load_t2s_model(checkpoint_path: impl AsRef<Path>) -> Result<T2SModel, Error> {
    let path = checkpoint_path.as_ref();

    // Try to load config if exists
    let config = T2SConfig::default();

    // Create model
    let mut model = T2SModel::new(config)?;

    // Load weights from .ckpt or .safetensors
    if path.extension().map_or(false, |e| e == "ckpt") {
        // PyTorch checkpoint - need to convert first
        return Err(Error::Message(
            "Direct .ckpt loading not supported. Convert to safetensors first.".to_string(),
        ));
    } else if path.extension().map_or(false, |e| e == "safetensors") {
        let weights = Array::load_safetensors(path)?;
        load_t2s_weights(&mut model, &weights)?;
    } else {
        return Err(Error::Message(format!(
            "Unsupported weight format: {:?}",
            path
        )));
    }

    Ok(model)
}

/// Generator for T2S model
pub struct T2SGenerate<'a, C> {
    model: &'a mut T2SModel,
    phoneme_ids: &'a Array,
    bert_features: &'a Array,
    cache: &'a mut Vec<Option<C>>,
    current_token: Array,
    temp: f32,
    top_k: i32,
    max_tokens: usize,
    generated: usize,
    finished: bool,
}

impl<'a, C> T2SGenerate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut T2SModel,
        phoneme_ids: &'a Array,
        bert_features: &'a Array,
        cache: &'a mut Vec<Option<C>>,
        start_token: i32,
        temp: f32,
        top_k: i32,
        max_tokens: usize,
    ) -> Result<Self, Exception> {
        let batch_size = phoneme_ids.shape()[0] as i32;
        let current_token = Array::full::<i32>(&[batch_size, 1], array!(start_token))?;

        Ok(Self {
            model,
            phoneme_ids,
            bert_features,
            cache,
            current_token,
            temp,
            top_k,
            max_tokens,
            generated: 0,
            finished: false,
        })
    }
}

impl<'a, C> Iterator for T2SGenerate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::async_eval;

        if self.finished || self.generated >= self.max_tokens {
            return None;
        }

        // Forward pass
        let input = T2SInput {
            phoneme_ids: self.phoneme_ids,
            semantic_ids: &self.current_token,
            bert_features: self.bert_features,
            cache: self.cache,
        };

        let logits = match self.model.forward(input) {
            Ok(l) => l,
            Err(e) => return Some(Err(e)),
        };

        // Get logits for last position
        let last_logits = logits.index((.., -1, ..));

        // Sample next token
        let next_token = match sample_top_k(&last_logits, self.temp, self.top_k) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Queue async eval for pipelining
        let _ = async_eval([&next_token]);

        // Check for EOS
        let eos = self.model.config.eos_token;
        // Simple check - in production would check all batch elements
        let first_token = next_token.index(0).item::<i32>();
        if first_token == eos {
            self.finished = true;
        }

        // Update for next iteration - add new axis to make it (batch, 1) shape
        self.current_token = next_token.index((.., NewAxis));
        self.generated += 1;

        // Periodic cache clearing
        if self.generated % 256 == 0 {
            unsafe {
                mlx_sys::mlx_clear_cache();
            }
        }

        Some(Ok(next_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_t2s_config_default() {
        let config = T2SConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn test_t2s_attention_shape() {
        let config = T2SConfig::default();
        let mut attn = T2SAttention::new(&config).unwrap();

        let x = Array::zeros::<f32>(&[1, 10, 512]).unwrap();
        let input = T2SAttentionInput::<crate::cache::ConcatKeyValueCache> {
            x: &x,
            mask: None,
            cache: None,
        };

        let output = attn.forward(input).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 512]);
    }

    #[test]
    fn test_t2s_model_forward() {
        let config = T2SConfig {
            num_layers: 2, // Use fewer layers for testing
            ..Default::default()
        };
        let mut model = T2SModel::new(config).unwrap();

        let phoneme_ids = Array::zeros::<i32>(&[1, 5]).unwrap();
        let semantic_ids = Array::zeros::<i32>(&[1, 1]).unwrap();
        let bert_features = Array::zeros::<f32>(&[1, 5, 1024]).unwrap();

        let mut cache: Vec<Option<crate::cache::ConcatKeyValueCache>> = Vec::new();

        let input = T2SInput {
            phoneme_ids: &phoneme_ids,
            semantic_ids: &semantic_ids,
            bert_features: &bert_features,
            cache: &mut cache,
        };

        let logits = model.forward(input).unwrap();
        eval([&logits]).unwrap();

        // Output should be (batch, full_seq_len, vocab_size) during prefill
        // full_seq_len = phoneme (5) + bert (5) + semantic (1) = 11
        assert_eq!(logits.shape(), &[1, 11, 1025]);
    }
}
