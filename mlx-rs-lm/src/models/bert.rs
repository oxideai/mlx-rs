//! BERT / RoBERTa Text Encoder for GPT-SoVITS
//!
//! This module provides text feature extraction using a BERT-like architecture.
//! GPT-SoVITS uses Chinese RoBERTa (hfl/chinese-roberta-wwm-ext-large) to extract
//! 1024-dimensional features from text for conditioning the TTS model.
//!
//! Architecture:
//! - Token embeddings + Position embeddings
//! - Transformer encoder (24 layers for large model)
//! - Final layer norm
//!
//! Input: Token IDs [batch, seq_len]
//! Output: Features [batch, seq_len, 1024]

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{softmax_axis, transpose_axes},
    Array,
};
use serde::Deserialize;

use crate::error::Error;

/// Configuration for BERT encoder
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension
    #[serde(default = "default_hidden_dim")]
    pub hidden_dim: i32,
    /// Number of attention heads
    #[serde(default = "default_num_heads")]
    pub num_heads: i32,
    /// Number of transformer layers
    #[serde(default = "default_num_layers")]
    pub num_layers: i32,
    /// FFN intermediate dimension
    #[serde(default = "default_intermediate_dim")]
    pub intermediate_dim: i32,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: i32,
    /// Dropout rate
    #[serde(default = "default_dropout")]
    pub dropout: f32,
    /// Layer norm epsilon
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Number of token types (for sentence pair tasks)
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: i32,
}

fn default_vocab_size() -> i32 { 21128 }  // Chinese BERT vocab
fn default_hidden_dim() -> i32 { 1024 }   // Large model
fn default_num_heads() -> i32 { 16 }      // Large model
fn default_num_layers() -> i32 { 24 }     // Large model
fn default_intermediate_dim() -> i32 { 4096 }  // 4x hidden
fn default_max_seq_len() -> i32 { 512 }
fn default_dropout() -> f32 { 0.1 }
fn default_layer_norm_eps() -> f32 { 1e-12 }
fn default_type_vocab_size() -> i32 { 2 }

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_dim: default_hidden_dim(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            intermediate_dim: default_intermediate_dim(),
            max_seq_len: default_max_seq_len(),
            dropout: default_dropout(),
            layer_norm_eps: default_layer_norm_eps(),
            type_vocab_size: default_type_vocab_size(),
        }
    }
}

impl BertConfig {
    /// Create config for BERT base model (12 layers, 768 hidden)
    pub fn base() -> Self {
        Self {
            vocab_size: 21128,
            hidden_dim: 768,
            num_heads: 12,
            num_layers: 12,
            intermediate_dim: 3072,
            max_seq_len: 512,
            dropout: 0.1,
            layer_norm_eps: 1e-12,
            type_vocab_size: 2,
        }
    }

    /// Create config for BERT large model (24 layers, 1024 hidden)
    pub fn large() -> Self {
        Self::default()
    }
}

/// BERT embeddings: token + position + token_type
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertEmbeddings {
    #[param]
    pub word_embeddings: nn::Embedding,
    #[param]
    pub position_embeddings: nn::Embedding,
    #[param]
    pub token_type_embeddings: nn::Embedding,
    #[param]
    pub layer_norm: nn::LayerNorm,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let word_embeddings = nn::Embedding::new(config.vocab_size, config.hidden_dim)?;
        let position_embeddings = nn::Embedding::new(config.max_seq_len, config.hidden_dim)?;
        let token_type_embeddings = nn::Embedding::new(config.type_vocab_size, config.hidden_dim)?;
        let layer_norm = nn::LayerNormBuilder::new(config.hidden_dim)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }
}

/// Input for BERT embeddings
pub struct BertEmbeddingInput<'a> {
    /// Token IDs [batch, seq_len]
    pub input_ids: &'a Array,
    /// Token type IDs [batch, seq_len] (optional, defaults to zeros)
    pub token_type_ids: Option<&'a Array>,
    /// Position IDs [batch, seq_len] (optional, defaults to 0..seq_len)
    pub position_ids: Option<&'a Array>,
}

impl Module<BertEmbeddingInput<'_>> for BertEmbeddings {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertEmbeddingInput<'_>) -> Result<Self::Output, Self::Error> {
        let input_ids = input.input_ids;
        let seq_len = input_ids.shape()[1] as i32;

        // Word embeddings
        let word_embeds = self.word_embeddings.forward(input_ids)?;

        // Position embeddings
        let position_ids = match input.position_ids {
            Some(ids) => ids.clone(),
            None => {
                // Create position IDs: [0, 1, 2, ..., seq_len-1]
                let positions: Vec<i32> = (0..seq_len).collect();
                Array::from_slice(&positions, &[1, seq_len])
            }
        };
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Token type embeddings
        let token_type_ids = match input.token_type_ids {
            Some(ids) => ids.clone(),
            None => Array::zeros::<i32>(&[input_ids.shape()[0] as i32, seq_len])?,
        };
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;

        // Sum all embeddings
        let embeddings = word_embeds.add(&position_embeds)?.add(&token_type_embeds)?;

        // Layer norm
        self.layer_norm.forward(&embeddings)
    }

    fn training_mode(&mut self, mode: bool) {
        self.word_embeddings.training_mode(mode);
        self.position_embeddings.training_mode(mode);
        self.token_type_embeddings.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

/// BERT self-attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertSelfAttention {
    #[param]
    pub query: nn::Linear,
    #[param]
    pub key: nn::Linear,
    #[param]
    pub value: nn::Linear,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl BertSelfAttention {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let head_dim = config.hidden_dim / config.num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let query = nn::LinearBuilder::new(config.hidden_dim, config.hidden_dim)
            .bias(true)
            .build()?;
        let key = nn::LinearBuilder::new(config.hidden_dim, config.hidden_dim)
            .bias(true)
            .build()?;
        let value = nn::LinearBuilder::new(config.hidden_dim, config.hidden_dim)
            .bias(true)
            .build()?;

        Ok(Self {
            query,
            key,
            value,
            num_heads: config.num_heads,
            head_dim,
            scale,
        })
    }
}

/// Input for BERT self-attention
pub struct BertAttentionInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl Module<BertAttentionInput<'_>> for BertSelfAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let x = input.hidden_states;
        let shape = x.shape();
        let batch = shape[0] as i32;
        let seq_len = shape[1] as i32;

        // Project Q, K, V
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention
        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let q = transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = transpose_axes(&v, &[0, 2, 1, 3])?;

        // Attention scores
        let k_t = transpose_axes(&k, &[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;

        // Apply attention mask if provided
        if let Some(mask) = input.attention_mask {
            // Mask should be [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            // with 0 for positions to attend and -inf for positions to mask
            scores = scores.add(mask)?;
        }

        // Softmax
        let attn_weights = softmax_axis(&scores, -1, None)?;

        // Apply attention
        let context = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let context = transpose_axes(&context, &[0, 2, 1, 3])?;
        context.reshape(&[batch, seq_len, self.num_heads * self.head_dim])
    }

    fn training_mode(&mut self, mode: bool) {
        self.query.training_mode(mode);
        self.key.training_mode(mode);
        self.value.training_mode(mode);
    }
}

/// BERT attention output projection
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertSelfOutput {
    #[param]
    pub dense: nn::Linear,
    #[param]
    pub layer_norm: nn::LayerNorm,
}

impl BertSelfOutput {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let dense = nn::LinearBuilder::new(config.hidden_dim, config.hidden_dim)
            .bias(true)
            .build()?;
        let layer_norm = nn::LayerNormBuilder::new(config.hidden_dim)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self { dense, layer_norm })
    }
}

/// Input for BERT self output
pub struct BertSelfOutputInput<'a> {
    pub hidden_states: &'a Array,
    pub input_tensor: &'a Array,
}

impl Module<BertSelfOutputInput<'_>> for BertSelfOutput {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertSelfOutputInput<'_>) -> Result<Self::Output, Self::Error> {
        let h = self.dense.forward(input.hidden_states)?;
        // Residual connection + layer norm
        let h = h.add(input.input_tensor)?;
        self.layer_norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

/// Full BERT attention block
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertAttention {
    #[param]
    pub self_attn: BertSelfAttention,
    #[param]
    pub output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let self_attn = BertSelfAttention::new(config)?;
        let output = BertSelfOutput::new(config)?;
        Ok(Self { self_attn, output })
    }
}

impl Module<BertAttentionInput<'_>> for BertAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let self_output = self.self_attn.forward(BertAttentionInput {
            hidden_states: input.hidden_states,
            attention_mask: input.attention_mask,
        })?;
        self.output.forward(BertSelfOutputInput {
            hidden_states: &self_output,
            input_tensor: input.hidden_states,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.output.training_mode(mode);
    }
}

/// BERT intermediate (FFN first layer)
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertIntermediate {
    #[param]
    pub dense: nn::Linear,
}

impl BertIntermediate {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let dense = nn::LinearBuilder::new(config.hidden_dim, config.intermediate_dim)
            .bias(true)
            .build()?;
        Ok(Self { dense })
    }
}

impl Module<&Array> for BertIntermediate {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.dense.forward(x)?;
        nn::gelu(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
    }
}

/// BERT output (FFN second layer + residual + layer norm)
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertOutput {
    #[param]
    pub dense: nn::Linear,
    #[param]
    pub layer_norm: nn::LayerNorm,
}

impl BertOutput {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let dense = nn::LinearBuilder::new(config.intermediate_dim, config.hidden_dim)
            .bias(true)
            .build()?;
        let layer_norm = nn::LayerNormBuilder::new(config.hidden_dim)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self { dense, layer_norm })
    }
}

/// Input for BERT output
pub struct BertOutputInput<'a> {
    pub hidden_states: &'a Array,
    pub input_tensor: &'a Array,
}

impl Module<BertOutputInput<'_>> for BertOutput {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertOutputInput<'_>) -> Result<Self::Output, Self::Error> {
        let h = self.dense.forward(input.hidden_states)?;
        // Residual connection + layer norm
        let h = h.add(input.input_tensor)?;
        self.layer_norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

/// BERT encoder layer
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertLayer {
    #[param]
    pub attention: BertAttention,
    #[param]
    pub intermediate: BertIntermediate,
    #[param]
    pub output: BertOutput,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let attention = BertAttention::new(config)?;
        let intermediate = BertIntermediate::new(config)?;
        let output = BertOutput::new(config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
}

impl Module<BertAttentionInput<'_>> for BertLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let attention_output = self.attention.forward(input)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(BertOutputInput {
            hidden_states: &intermediate_output,
            input_tensor: &attention_output,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
        self.intermediate.training_mode(mode);
        self.output.training_mode(mode);
    }
}

/// BERT encoder (stack of transformer layers)
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertEncoder {
    #[param]
    pub layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            layers.push(BertLayer::new(config)?);
        }
        Ok(Self { layers })
    }
}

/// Input for BERT encoder
pub struct BertEncoderInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

/// Output from BERT encoder with all hidden states
pub struct BertEncoderOutput {
    /// Final hidden states [batch, seq_len, hidden_dim]
    pub last_hidden_state: Array,
    /// All hidden states from each layer (including embedding layer)
    /// Length = num_layers + 1
    pub hidden_states: Vec<Array>,
}

impl BertEncoder {
    /// Forward pass returning all hidden states from each layer
    pub fn forward_with_hidden_states(
        &mut self,
        input: BertEncoderInput<'_>,
    ) -> Result<BertEncoderOutput, Exception> {
        let mut h = input.hidden_states.clone();
        let mut all_hidden_states = Vec::with_capacity(self.layers.len() + 1);

        // Store embedding layer output as first hidden state
        all_hidden_states.push(h.clone());

        for layer in &mut self.layers {
            h = layer.forward(BertAttentionInput {
                hidden_states: &h,
                attention_mask: input.attention_mask,
            })?;
            all_hidden_states.push(h.clone());
        }

        Ok(BertEncoderOutput {
            last_hidden_state: h,
            hidden_states: all_hidden_states,
        })
    }
}

impl Module<BertEncoderInput<'_>> for BertEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertEncoderInput<'_>) -> Result<Self::Output, Self::Error> {
        let mut h = input.hidden_states.clone();
        for layer in &mut self.layers {
            h = layer.forward(BertAttentionInput {
                hidden_states: &h,
                attention_mask: input.attention_mask,
            })?;
        }
        Ok(h)
    }

    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
    }
}

/// Full BERT model for feature extraction
#[derive(Debug, Clone, ModuleParameters)]
pub struct BertModel {
    pub config: BertConfig,

    #[param]
    pub embeddings: BertEmbeddings,
    #[param]
    pub encoder: BertEncoder,
}

/// Output from BERT model with all hidden states
pub struct BertModelOutput {
    /// Final hidden states [batch, seq_len, hidden_dim]
    pub last_hidden_state: Array,
    /// All hidden states from each layer (including embedding layer)
    /// Length = num_layers + 1
    pub hidden_states: Vec<Array>,
}

impl BertModel {
    pub fn new(config: BertConfig) -> Result<Self, Exception> {
        let embeddings = BertEmbeddings::new(&config)?;
        let encoder = BertEncoder::new(&config)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
        })
    }

    /// Forward pass returning all hidden states from each layer
    pub fn forward_with_hidden_states(
        &mut self,
        input: BertModelInput<'_>,
    ) -> Result<BertModelOutput, Exception> {
        // Get embeddings
        let embeddings = self.embeddings.forward(BertEmbeddingInput {
            input_ids: input.input_ids,
            token_type_ids: input.token_type_ids,
            position_ids: None,
        })?;

        // Run encoder with hidden states
        let encoder_output = self.encoder.forward_with_hidden_states(BertEncoderInput {
            hidden_states: &embeddings,
            attention_mask: input.attention_mask,
        })?;

        Ok(BertModelOutput {
            last_hidden_state: encoder_output.last_hidden_state,
            hidden_states: encoder_output.hidden_states,
        })
    }

    /// Extract features for TTS from a specific layer
    ///
    /// This is designed to match the Python GPT-SoVITS behavior:
    /// 1. Run BERT and get hidden states from the 3rd-from-last layer
    /// 2. Remove CLS and SEP tokens (first and last)
    /// 3. Expand features according to word2ph to align with phonemes
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len] from BERT tokenizer
    /// * `word2ph` - Number of phonemes per character (len = seq_len - 2 for CLS/SEP)
    /// * `layer_idx` - Which layer to use (-3 means 3rd from last)
    ///
    /// # Returns
    /// Features [batch, total_phonemes, hidden_dim]
    pub fn extract_features_for_tts(
        &mut self,
        input_ids: &Array,
        word2ph: &[i32],
        layer_idx: i32,
    ) -> Result<Array, Exception> {
        use mlx_rs::ops::indexing::IndexOp;

        // Forward with hidden states
        let output = self.forward_with_hidden_states(BertModelInput {
            input_ids,
            token_type_ids: None,
            attention_mask: None,
        })?;

        // Get the specified layer's hidden states
        // layer_idx of -3 means 3rd from last
        let num_layers = output.hidden_states.len() as i32;
        let actual_idx = if layer_idx < 0 {
            (num_layers + layer_idx) as usize
        } else {
            layer_idx as usize
        };

        let hidden = output.hidden_states.get(actual_idx)
            .ok_or_else(|| Exception::from("Layer index out of range"))?;

        // hidden shape: [batch, seq_len, hidden_dim]
        // Remove CLS (first) and SEP (last) tokens: [1:-1]
        let seq_len = hidden.shape()[1] as i32;
        // Use index with ranges: (.., 1..(seq_len-1), ..)
        let hidden_trimmed = hidden.index((.., 1..(seq_len - 1), ..));

        // Now expand features according to word2ph
        // hidden_trimmed: [batch, text_len, hidden_dim]
        // We need to repeat each position i by word2ph[i] times
        let bert_token_len = hidden_trimmed.shape()[1] as usize;
        let hidden_dim = hidden_trimmed.shape()[2] as i32;

        // Handle mismatch between BERT tokens and word2ph length
        // This happens with mixed Chinese/English text where BERT uses subword tokenization
        // for English but word2ph expects character-level alignment
        if word2ph.len() != bert_token_len {
            // Fall back to simple approach: pad/truncate BERT features to match phoneme count
            let total_phonemes: i32 = word2ph.iter().sum();

            if bert_token_len == 0 {
                // No BERT tokens, return zeros
                return Array::zeros::<f32>(&[1, total_phonemes, hidden_dim]);
            }

            // Simple expansion: repeat BERT features to match phoneme count
            // This is approximate but works for mixed text
            let mut gather_indices = Vec::with_capacity(total_phonemes as usize);
            let mut bert_idx = 0usize;
            for &count in word2ph.iter() {
                for _ in 0..count {
                    gather_indices.push((bert_idx % bert_token_len) as i32);
                }
                // Advance BERT index proportionally
                bert_idx += 1;
                if bert_idx >= bert_token_len {
                    bert_idx = bert_token_len - 1; // Clamp to last token
                }
            }

            let indices_arr = Array::from_slice(&gather_indices, &[total_phonemes]);
            let result = hidden_trimmed.take_along_axis(&indices_arr.reshape(&[1, total_phonemes, 1])?, 1)?;
            return Ok(result);
        }

        // Calculate total output length
        let total_phonemes: i32 = word2ph.iter().sum();

        // Build indices for gather operation
        // indices[i] = which original position to use for output position i
        let mut gather_indices = Vec::with_capacity(total_phonemes as usize);
        for (char_idx, &count) in word2ph.iter().enumerate() {
            for _ in 0..count {
                gather_indices.push(char_idx as i32);
            }
        }

        // Use take_axis to gather features along axis 0: [batch, total_phonemes, hidden_dim]
        // hidden_trimmed is [1, text_len, hidden_dim]
        // We need to gather along axis 1
        let indices_arr = Array::from_slice(&gather_indices, &[total_phonemes]);
        let result = hidden_trimmed.take_along_axis(&indices_arr.reshape(&[1, total_phonemes, 1])?, 1)?;

        Ok(result)
    }

    /// Create attention mask from input IDs (mask padding tokens)
    pub fn create_attention_mask(input_ids: &Array, pad_token_id: i32) -> Result<Array, Exception> {
        // Create mask: 1 for real tokens, 0 for padding
        let mask = input_ids.ne(array!(pad_token_id))?;
        // Convert to attention mask format: [batch, 1, 1, seq_len]
        // 0 for positions to attend, -1e9 for positions to mask
        let mask = mask.as_type::<f32>()?;
        let mask = mask.multiply(array!(-1.0f32))?.add(array!(1.0f32))?;
        let mask = mask.multiply(array!(-1e9f32))?;
        // Reshape to [batch, 1, 1, seq_len]
        let shape = mask.shape();
        mask.reshape(&[shape[0] as i32, 1, 1, shape[1] as i32])
    }
}

/// Input for BERT model
pub struct BertModelInput<'a> {
    /// Token IDs [batch, seq_len]
    pub input_ids: &'a Array,
    /// Token type IDs [batch, seq_len] (optional)
    pub token_type_ids: Option<&'a Array>,
    /// Attention mask [batch, 1, 1, seq_len] (optional)
    pub attention_mask: Option<&'a Array>,
}

impl Module<BertModelInput<'_>> for BertModel {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertModelInput<'_>) -> Result<Self::Output, Self::Error> {
        // Get embeddings
        let embeddings = self.embeddings.forward(BertEmbeddingInput {
            input_ids: input.input_ids,
            token_type_ids: input.token_type_ids,
            position_ids: None,
        })?;

        // Run encoder
        self.encoder.forward(BertEncoderInput {
            hidden_states: &embeddings,
            attention_mask: input.attention_mask,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        self.encoder.training_mode(mode);
    }
}

/// Load BERT weights from safetensors
pub fn load_bert_weights(
    model: &mut BertModel,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    // Helper to get weight with fallback names
    let get_weight = |keys: &[&str]| -> Option<Array> {
        for key in keys {
            if let Some(w) = weights.get(*key) {
                return Some(w.clone());
            }
        }
        None
    };

    // Load embeddings - handle both naming conventions
    if let Some(w) = get_weight(&["embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings.weight"]) {
        model.embeddings.word_embeddings.weight = Param::new(w);
    }
    if let Some(w) = get_weight(&["embeddings.position_embeddings.weight", "bert.embeddings.position_embeddings.weight"]) {
        model.embeddings.position_embeddings.weight = Param::new(w);
    }
    if let Some(w) = get_weight(&["embeddings.token_type_embeddings.weight", "bert.embeddings.token_type_embeddings.weight"]) {
        model.embeddings.token_type_embeddings.weight = Param::new(w);
    }
    if let Some(w) = get_weight(&["embeddings.layer_norm.weight", "bert.embeddings.LayerNorm.weight"]) {
        model.embeddings.layer_norm.weight = Param::new(Some(w));
    }
    if let Some(b) = get_weight(&["embeddings.layer_norm.bias", "bert.embeddings.LayerNorm.bias"]) {
        model.embeddings.layer_norm.bias = Param::new(Some(b));
    }

    // Load encoder layers
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        // New naming: encoder.layers.{i}
        // Old naming: bert.encoder.layer.{i}
        let new_prefix = format!("encoder.layers.{}", i);
        let old_prefix = format!("bert.encoder.layer.{}", i);

        // Self attention - new naming uses self_attn.{q,k,v}_proj, old uses self.{query,key,value}
        if let Some(w) = get_weight(&[
            &format!("{}.attention.self_attn.q_proj.weight", new_prefix),
            &format!("{}.attention.self.query.weight", old_prefix),
        ]) {
            layer.attention.self_attn.query.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.attention.self_attn.q_proj.bias", new_prefix),
            &format!("{}.attention.self.query.bias", old_prefix),
        ]) {
            layer.attention.self_attn.query.bias = Param::new(Some(b));
        }
        if let Some(w) = get_weight(&[
            &format!("{}.attention.self_attn.k_proj.weight", new_prefix),
            &format!("{}.attention.self.key.weight", old_prefix),
        ]) {
            layer.attention.self_attn.key.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.attention.self_attn.k_proj.bias", new_prefix),
            &format!("{}.attention.self.key.bias", old_prefix),
        ]) {
            layer.attention.self_attn.key.bias = Param::new(Some(b));
        }
        if let Some(w) = get_weight(&[
            &format!("{}.attention.self_attn.v_proj.weight", new_prefix),
            &format!("{}.attention.self.value.weight", old_prefix),
        ]) {
            layer.attention.self_attn.value.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.attention.self_attn.v_proj.bias", new_prefix),
            &format!("{}.attention.self.value.bias", old_prefix),
        ]) {
            layer.attention.self_attn.value.bias = Param::new(Some(b));
        }

        // Attention output
        if let Some(w) = get_weight(&[
            &format!("{}.attention.output.dense.weight", new_prefix),
            &format!("{}.attention.output.dense.weight", old_prefix),
        ]) {
            layer.attention.output.dense.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.attention.output.dense.bias", new_prefix),
            &format!("{}.attention.output.dense.bias", old_prefix),
        ]) {
            layer.attention.output.dense.bias = Param::new(Some(b));
        }
        if let Some(w) = get_weight(&[
            &format!("{}.attention.output.layer_norm.weight", new_prefix),
            &format!("{}.attention.output.LayerNorm.weight", old_prefix),
        ]) {
            layer.attention.output.layer_norm.weight = Param::new(Some(w));
        }
        if let Some(b) = get_weight(&[
            &format!("{}.attention.output.layer_norm.bias", new_prefix),
            &format!("{}.attention.output.LayerNorm.bias", old_prefix),
        ]) {
            layer.attention.output.layer_norm.bias = Param::new(Some(b));
        }

        // Intermediate
        if let Some(w) = get_weight(&[
            &format!("{}.intermediate.dense.weight", new_prefix),
            &format!("{}.intermediate.dense.weight", old_prefix),
        ]) {
            layer.intermediate.dense.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.intermediate.dense.bias", new_prefix),
            &format!("{}.intermediate.dense.bias", old_prefix),
        ]) {
            layer.intermediate.dense.bias = Param::new(Some(b));
        }

        // Output
        if let Some(w) = get_weight(&[
            &format!("{}.output.dense.weight", new_prefix),
            &format!("{}.output.dense.weight", old_prefix),
        ]) {
            layer.output.dense.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("{}.output.dense.bias", new_prefix),
            &format!("{}.output.dense.bias", old_prefix),
        ]) {
            layer.output.dense.bias = Param::new(Some(b));
        }
        if let Some(w) = get_weight(&[
            &format!("{}.output.layer_norm.weight", new_prefix),
            &format!("{}.output.LayerNorm.weight", old_prefix),
        ]) {
            layer.output.layer_norm.weight = Param::new(Some(w));
        }
        if let Some(b) = get_weight(&[
            &format!("{}.output.layer_norm.bias", new_prefix),
            &format!("{}.output.LayerNorm.bias", old_prefix),
        ]) {
            layer.output.layer_norm.bias = Param::new(Some(b));
        }
    }

    Ok(())
}

/// Load BERT model from safetensors file
pub fn load_bert_model(weights_path: impl AsRef<Path>) -> Result<BertModel, Error> {
    let path = weights_path.as_ref();

    // Use large config for Chinese RoBERTa
    let config = BertConfig::default();

    let mut model = BertModel::new(config)?;

    let weights = Array::load_safetensors(path)?;
    load_bert_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_bert_config_default() {
        let config = BertConfig::default();
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
    }

    #[test]
    fn test_bert_config_base() {
        let config = BertConfig::base();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
    }

    #[test]
    fn test_bert_embeddings() {
        let config = BertConfig::base();
        let mut embeddings = BertEmbeddings::new(&config).unwrap();

        let input_ids = Array::zeros::<i32>(&[1, 10]).unwrap();
        let output = embeddings.forward(BertEmbeddingInput {
            input_ids: &input_ids,
            token_type_ids: None,
            position_ids: None,
        }).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 768]);
    }

    #[test]
    fn test_bert_self_attention() {
        let config = BertConfig::base();
        let mut attn = BertSelfAttention::new(&config).unwrap();

        let x = Array::zeros::<f32>(&[1, 10, 768]).unwrap();
        let output = attn.forward(BertAttentionInput {
            hidden_states: &x,
            attention_mask: None,
        }).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 768]);
    }

    #[test]
    fn test_bert_layer() {
        let config = BertConfig::base();
        let mut layer = BertLayer::new(&config).unwrap();

        let x = Array::zeros::<f32>(&[1, 10, 768]).unwrap();
        let output = layer.forward(BertAttentionInput {
            hidden_states: &x,
            attention_mask: None,
        }).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 768]);
    }

    #[test]
    fn test_bert_model() {
        // Use smaller config for faster test
        let config = BertConfig {
            vocab_size: 1000,
            hidden_dim: 256,
            num_heads: 4,
            num_layers: 2,
            intermediate_dim: 512,
            max_seq_len: 128,
            ..Default::default()
        };
        let mut model = BertModel::new(config).unwrap();

        let input_ids = Array::zeros::<i32>(&[1, 10]).unwrap();
        let output = model.forward(BertModelInput {
            input_ids: &input_ids,
            token_type_ids: None,
            attention_mask: None,
        }).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 256]);
    }

    #[test]
    fn test_attention_mask_creation() {
        let input_ids = Array::from_slice(&[1i32, 2, 3, 0, 0], &[1, 5]);
        let mask = BertModel::create_attention_mask(&input_ids, 0).unwrap();
        eval([&mask]).unwrap();

        assert_eq!(mask.shape(), &[1, 1, 1, 5]);
    }
}
