//! HuBERT / CNHubert Audio Encoder for GPT-SoVITS
//!
//! This module provides audio feature extraction using the HuBERT architecture.
//! CNHubert is used in GPT-SoVITS to extract 768-dimensional features from audio
//! at approximately 50Hz (one feature vector every 20ms).
//!
//! # Architecture
//!
//! ```text
//! Audio (16kHz)
//!     ↓
//! [Audio Normalization] - Zero mean, unit variance
//!     ↓
//! [Feature Extractor] - 7 conv layers, ~320x downsample
//!     ↓
//! [Feature Projection] - LayerNorm + Linear (512 → 768)
//!     ↓
//! [Positional Conv Embedding] - Grouped conv with weight normalization
//!     ↓
//! [Transformer Encoder] - 12 layers, post-norm
//!     ↓
//! Features [batch, time, 768]
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use mlx_rs_lm::models::hubert::load_hubert_model;
//! use mlx_rs_lm::audio::load_audio_for_hubert;
//! use mlx_rs::{module::Module, transforms::eval};
//!
//! // Load model
//! let mut hubert = load_hubert_model("/path/to/hubert.safetensors")?;
//!
//! // Load and preprocess audio (resamples to 16kHz, normalizes)
//! let audio = load_audio_for_hubert("/path/to/audio.wav")?;
//! eval([&audio])?;
//!
//! // Extract features
//! let features = hubert.forward(&audio)?;
//! eval([&features])?;
//! // features shape: [1, num_frames, 768]
//! ```
//!
//! # Key Implementation Details
//!
//! 1. **Audio normalization**: `(audio - mean) / sqrt(var + 1e-7)` matching Wav2Vec2FeatureExtractor
//! 2. **Feature extractor**: Only layer 0 has LayerNorm, layers 1-6 don't
//! 3. **Positional conv**: Weight normalization with `w = g * v / ||v||`, groups=16, kernel=128
//! 4. **Encoder layers**: Post-norm (LayerNorm after residual addition)
//!
//! For detailed documentation, see `docs/hubert.md`

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp, softmax_axis, swap_axes, transpose_axes},
    Array,
};
use serde::Deserialize;

use crate::error::Error;

/// Configuration for HuBERT encoder
#[derive(Debug, Clone, Deserialize)]
pub struct HuBertConfig {
    /// Input sample rate (must be 16000)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: i32,
    /// Feature extractor output dimension
    #[serde(default = "default_conv_dim")]
    pub conv_dim: i32,
    /// Hidden dimension of transformer
    #[serde(default = "default_hidden_dim")]
    pub hidden_dim: i32,
    /// Number of attention heads
    #[serde(default = "default_num_heads")]
    pub num_heads: i32,
    /// Number of transformer layers
    #[serde(default = "default_num_layers")]
    pub num_layers: i32,
    /// FFN intermediate dimension
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: i32,
    /// Dropout rate
    #[serde(default = "default_dropout")]
    pub dropout: f32,
    /// Output feature dimension
    #[serde(default = "default_output_dim")]
    pub output_dim: i32,
}

fn default_sample_rate() -> i32 { 16000 }
fn default_conv_dim() -> i32 { 512 }
fn default_hidden_dim() -> i32 { 768 }
fn default_num_heads() -> i32 { 12 }
fn default_num_layers() -> i32 { 12 }
fn default_ffn_dim() -> i32 { 3072 }
fn default_dropout() -> f32 { 0.1 }
fn default_output_dim() -> i32 { 768 }

impl Default for HuBertConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            conv_dim: default_conv_dim(),
            hidden_dim: default_hidden_dim(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            ffn_dim: default_ffn_dim(),
            dropout: default_dropout(),
            output_dim: default_output_dim(),
        }
    }
}

/// Convolutional layer with optional group norm (for layer 0 only)
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvLayer {
    #[param]
    pub conv: nn::Conv1d,
    #[param]
    pub norm: Option<nn::LayerNorm>,  // Only layer 0 has norm
    pub use_gelu: bool,
}

impl ConvLayer {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        use_norm: bool,
        use_gelu: bool,
    ) -> Result<Self, Exception> {
        // No padding - HuBERT uses valid convolution
        let conv = nn::Conv1dBuilder::new(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(0)
            .build()?;

        let norm = if use_norm {
            Some(nn::LayerNormBuilder::new(out_channels).eps(1e-5).build()?)
        } else {
            None
        };

        Ok(Self { conv, norm, use_gelu })
    }
}

impl Module<&Array> for ConvLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let mut h = self.conv.forward(x)?;
        if let Some(ref mut norm) = self.norm {
            h = norm.forward(&h)?;
        }
        if self.use_gelu {
            h = nn::gelu(&h)?;
        }
        Ok(h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
        if let Some(ref mut norm) = self.norm {
            norm.training_mode(mode);
        }
    }
}

/// Convolutional feature extractor
///
/// Converts raw audio waveform to feature sequence.
/// 7 convolutional layers with progressive downsampling.
/// Only layer 0 has LayerNorm, others don't.
#[derive(Debug, Clone, ModuleParameters)]
pub struct FeatureExtractor {
    #[param]
    pub layers: Vec<ConvLayer>,
}

impl FeatureExtractor {
    pub fn new(config: &HuBertConfig) -> Result<Self, Exception> {
        // HuBERT feature extractor layers:
        // Layer 0: kernel=10, stride=5, with LayerNorm + GELU
        // Layers 1-4: kernel=3, stride=2, GELU only (no norm)
        // Layers 5-6: kernel=2, stride=2, GELU only (no norm)
        // Total: ~320x downsample (16kHz -> 50Hz)

        let conv_layers = [
            (1, config.conv_dim, 10, 5, true),    // Layer 0: has norm
            (config.conv_dim, config.conv_dim, 3, 2, false),  // Layers 1-6: no norm
            (config.conv_dim, config.conv_dim, 3, 2, false),
            (config.conv_dim, config.conv_dim, 3, 2, false),
            (config.conv_dim, config.conv_dim, 3, 2, false),
            (config.conv_dim, config.conv_dim, 2, 2, false),
            (config.conv_dim, config.conv_dim, 2, 2, false),
        ];

        let mut layers = Vec::with_capacity(conv_layers.len());
        for &(in_ch, out_ch, kernel, stride, use_norm) in conv_layers.iter() {
            layers.push(ConvLayer::new(in_ch, out_ch, kernel, stride, use_norm, true)?);
        }

        Ok(Self { layers })
    }
}

impl Module<&Array> for FeatureExtractor {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Input: [batch, samples] or [batch, samples, 1]
        let mut h = if x.ndim() == 2 {
            // Add channel dimension: [batch, samples] -> [batch, samples, 1]
            x.index((.., .., mlx_rs::ops::indexing::NewAxis))
        } else {
            x.clone()
        };

        // Apply conv layers
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }

        // Output: [batch, time, conv_dim]
        Ok(h)
    }

    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
    }
}

/// Feature projection with LayerNorm
#[derive(Debug, Clone, ModuleParameters)]
pub struct FeatureProjection {
    #[param]
    pub layer_norm: nn::LayerNorm,
    #[param]
    pub projection: nn::Linear,
}

impl FeatureProjection {
    pub fn new(in_dim: i32, out_dim: i32) -> Result<Self, Exception> {
        let layer_norm = nn::LayerNormBuilder::new(in_dim).eps(1e-5).build()?;
        let projection = nn::LinearBuilder::new(in_dim, out_dim).bias(true).build()?;
        Ok(Self { layer_norm, projection })
    }
}

impl Module<&Array> for FeatureProjection {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.layer_norm.forward(x)?;
        self.projection.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layer_norm.training_mode(mode);
        self.projection.training_mode(mode);
    }
}

/// Positional convolution embedding
/// Uses grouped convolution with weight normalization
/// groups=16, kernel_size=128 for HuBERT
#[derive(Debug, Clone, ModuleParameters)]
pub struct PosConvEmbed {
    #[param]
    pub conv: nn::Conv1d,
    // Weight normalization parameters
    // weight_g: [1, 1, kernel_size] - magnitude per output neuron
    // weight_v: [out_channels, in_channels/groups, kernel_size] - direction
    #[param]
    pub weight_g: Param<Array>,
    #[param]
    pub weight_v: Param<Array>,
    pub kernel_size: i32,
    pub groups: i32,
}

impl PosConvEmbed {
    pub fn new(hidden_dim: i32, kernel_size: i32, groups: i32) -> Result<Self, Exception> {
        // Grouped convolution - padding to maintain sequence length
        let padding = kernel_size / 2;
        let conv = nn::Conv1dBuilder::new(hidden_dim, hidden_dim, kernel_size)
            .padding(padding)
            .groups(groups)
            .build()?;

        // Weight normalization parameters (will be loaded from weights)
        // PyTorch weight_v: [out_channels, in_channels/groups, kernel_size] = [768, 48, 128]
        // weight_g: [1, 1, kernel_size] = [1, 1, 128]
        let in_per_group = hidden_dim / groups;
        let weight_g = Param::new(Array::ones::<f32>(&[1, 1, kernel_size])?);
        let weight_v = Param::new(Array::zeros::<f32>(&[hidden_dim, in_per_group, kernel_size])?);

        Ok(Self { conv, weight_g, weight_v, kernel_size, groups })
    }

    /// Compute normalized weight from weight_g and weight_v
    /// Returns weight in MLX format: [out_channels, kernel_size, in_channels/groups]
    fn compute_weight(&self) -> Result<Array, Exception> {
        // Weight normalization: w = g * (v / ||v||)
        // PyTorch weight_v: [out_channels=768, in_channels/groups=48, kernel_size=128]
        // weight_g: [1, 1, kernel_size=128]
        // Norm is computed over dim 0 (out_channels), keeping dims [1, 48, 128]

        let v = self.weight_v.as_ref();
        let g = self.weight_g.as_ref();

        // Compute L2 norm over out_channels (dim 0)
        let v_sq = v.multiply(v)?;
        let v_norm_sq = ops::sum_axis(&v_sq, 0, true)?;  // [1, 48, 128]
        let v_norm = ops::sqrt(&v_norm_sq.add(array!(1e-7))?)?;

        // Normalize and scale: w = g * (v / ||v||)
        let v_normalized = v.divide(&v_norm)?;  // [768, 48, 128]
        let weight = v_normalized.multiply(g)?;  // broadcast g [1,1,128] -> [768, 48, 128]

        // Transpose from PyTorch [out, in/groups, kernel] to MLX [out, kernel, in/groups]
        let weight_mlx = swap_axes(&weight, 1, 2)?;  // [768, 128, 48]

        Ok(weight_mlx)
    }
}

impl Module<&Array> for PosConvEmbed {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Compute weight-normalized conv weight
        let weight = self.compute_weight()?;
        self.conv.weight = Param::new(weight);

        let h = self.conv.forward(x)?;

        // Remove the extra frame caused by padding
        // With kernel_size=128 and padding=64, output has input_len+1 frames
        // Slice to match input length: h[:, :-1, :]
        let seq_len = x.shape()[1];
        let h = h.index((.., ..seq_len as i32, ..));

        nn::gelu(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
    }
}

/// Multi-head attention for encoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl EncoderAttention {
    pub fn new(hidden_dim: i32, num_heads: i32) -> Result<Self, Exception> {
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let q_proj = nn::LinearBuilder::new(hidden_dim, hidden_dim).bias(true).build()?;
        let k_proj = nn::LinearBuilder::new(hidden_dim, hidden_dim).bias(true).build()?;
        let v_proj = nn::LinearBuilder::new(hidden_dim, hidden_dim).bias(true).build()?;
        let out_proj = nn::LinearBuilder::new(hidden_dim, hidden_dim).bias(true).build()?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }
}

impl Module<&Array> for EncoderAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let seq_len = shape[1] as i32;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let q = transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = transpose_axes(&v, &[0, 2, 1, 3])?;

        // Attention scores: [batch, heads, seq, head_dim] x [batch, heads, head_dim, seq]
        let k_t = transpose_axes(&k, &[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;

        // Softmax
        let attn_weights = softmax_axis(&scores, -1, None)?;

        // Apply attention
        let context = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let context = transpose_axes(&context, &[0, 2, 1, 3])?;
        let context = context.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Output projection
        self.out_proj.forward(&context)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

/// Feed-forward network
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderFFN {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
}

impl EncoderFFN {
    pub fn new(hidden_dim: i32, ffn_dim: i32) -> Result<Self, Exception> {
        let fc1 = nn::LinearBuilder::new(hidden_dim, ffn_dim).bias(true).build()?;
        let fc2 = nn::LinearBuilder::new(ffn_dim, hidden_dim).bias(true).build()?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module<&Array> for EncoderFFN {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.fc1.forward(x)?;
        let h = nn::gelu(&h)?;
        self.fc2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.fc1.training_mode(mode);
        self.fc2.training_mode(mode);
    }
}

/// Transformer encoder layer
/// Uses post-norm (LayerNorm after residual)
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderLayer {
    #[param]
    pub self_attn: EncoderAttention,
    #[param]
    pub self_attn_norm: nn::LayerNorm,  // layer_norm (after attention residual)
    #[param]
    pub ffn: EncoderFFN,
    #[param]
    pub ffn_norm: nn::LayerNorm,  // final_layer_norm (after FFN residual)
}

impl EncoderLayer {
    pub fn new(config: &HuBertConfig) -> Result<Self, Exception> {
        let self_attn = EncoderAttention::new(config.hidden_dim, config.num_heads)?;
        let self_attn_norm = nn::LayerNormBuilder::new(config.hidden_dim).eps(1e-5).build()?;
        let ffn = EncoderFFN::new(config.hidden_dim, config.ffn_dim)?;
        let ffn_norm = nn::LayerNormBuilder::new(config.hidden_dim).eps(1e-5).build()?;

        Ok(Self {
            self_attn,
            self_attn_norm,
            ffn,
            ffn_norm,
        })
    }
}

impl Module<&Array> for EncoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Post-norm: residual + attn, then norm
        let h = self.self_attn.forward(x)?;
        let x = x.add(&h)?;
        let x = self.self_attn_norm.forward(&x)?;

        // FFN with post-norm
        let h = self.ffn.forward(&x)?;
        let x = x.add(&h)?;
        self.ffn_norm.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.self_attn_norm.training_mode(mode);
        self.ffn.training_mode(mode);
        self.ffn_norm.training_mode(mode);
    }
}

/// HuBERT encoder model
///
/// Extracts audio features from waveforms.
#[derive(Debug, Clone, ModuleParameters)]
pub struct HuBertEncoder {
    pub config: HuBertConfig,

    /// Convolutional feature extractor
    #[param]
    pub feature_extractor: FeatureExtractor,

    /// Feature projection (conv_dim -> hidden_dim) with LayerNorm
    #[param]
    pub feature_projection: FeatureProjection,

    /// Positional convolution embedding
    #[param]
    pub pos_conv_embed: PosConvEmbed,

    /// Transformer encoder layers
    #[param]
    pub encoder_layers: Vec<EncoderLayer>,

    /// Final layer norm
    #[param]
    pub layer_norm: nn::LayerNorm,

    /// Output projection (hidden_dim -> output_dim)
    #[param]
    pub output_projection: Option<nn::Linear>,
}

impl HuBertEncoder {
    pub fn new(config: HuBertConfig) -> Result<Self, Exception> {
        let feature_extractor = FeatureExtractor::new(&config)?;

        let feature_projection = FeatureProjection::new(config.conv_dim, config.hidden_dim)?;

        // Positional conv embedding: kernel=128, groups=16
        let pos_conv_embed = PosConvEmbed::new(config.hidden_dim, 128, 16)?;

        let mut encoder_layers = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            encoder_layers.push(EncoderLayer::new(&config)?);
        }

        let layer_norm = nn::LayerNormBuilder::new(config.hidden_dim).eps(1e-5).build()?;

        // Only add output projection if dimensions differ
        let output_projection = if config.hidden_dim != config.output_dim {
            Some(nn::LinearBuilder::new(config.hidden_dim, config.output_dim)
                .bias(true)
                .build()?)
        } else {
            None
        };

        Ok(Self {
            config,
            feature_extractor,
            feature_projection,
            pos_conv_embed,
            encoder_layers,
            layer_norm,
            output_projection,
        })
    }

    /// Normalize audio to zero mean and unit variance
    /// This matches Wav2Vec2FeatureExtractor preprocessing
    fn normalize_audio(&self, audio: &Array) -> Result<Array, Exception> {
        // Compute mean and variance along the sample dimension (last axis)
        let mean = ops::mean_axis(audio, -1, true)?;
        let var = ops::var_axis(audio, -1, true, None)?;

        // Normalize: (x - mean) / sqrt(var + 1e-7)
        let std = ops::sqrt(&var.add(array!(1e-7))?)?;
        audio.subtract(&mean)?.divide(&std)
    }
}

impl Module<&Array> for HuBertEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, audio: &Array) -> Result<Self::Output, Self::Error> {
        // Normalize audio (like Wav2Vec2FeatureExtractor)
        let audio = self.normalize_audio(audio)?;

        // Extract convolutional features
        let features = self.feature_extractor.forward(&audio)?;

        // Project to hidden dimension (with LayerNorm)
        let mut h = self.feature_projection.forward(&features)?;

        // Add positional embedding
        let pos_embed = self.pos_conv_embed.forward(&h)?;
        h = h.add(&pos_embed)?;

        // Apply transformer encoder
        for layer in &mut self.encoder_layers {
            h = layer.forward(&h)?;
        }

        // Final layer norm
        h = self.layer_norm.forward(&h)?;

        // Output projection if needed
        if let Some(ref mut proj) = self.output_projection {
            h = proj.forward(&h)?;
        }

        Ok(h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.feature_extractor.training_mode(mode);
        self.feature_projection.training_mode(mode);
        self.pos_conv_embed.training_mode(mode);
        for layer in &mut self.encoder_layers {
            layer.training_mode(mode);
        }
        self.layer_norm.training_mode(mode);
        if let Some(ref mut proj) = self.output_projection {
            proj.training_mode(mode);
        }
    }
}

/// Load HuBERT weights from safetensors
pub fn load_hubert_weights(
    model: &mut HuBertEncoder,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    // Helper to get weight
    let get_weight = |key: &str| -> Result<Array, Error> {
        weights
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Message(format!("Weight not found: {}", key)))
    };

    // Load feature extractor weights
    // PyTorch conv weights are [out, in, kernel], MLX conv1d expects [out, kernel, in]
    for (i, layer) in model.feature_extractor.layers.iter_mut().enumerate() {
        if let Ok(w) = get_weight(&format!("feature_extractor.conv_layers.{}.conv.weight", i)) {
            // Transpose from [out, in, kernel] to [out, kernel, in]
            let w = swap_axes(&w, 1, 2)?;
            layer.conv.weight = Param::new(w);
        }
        // Layers don't have bias in the weights file

        // Only layer 0 has layer_norm
        if i == 0 {
            if let Some(ref mut norm) = layer.norm {
                if let Ok(w) = get_weight(&format!("feature_extractor.conv_layers.{}.layer_norm.weight", i)) {
                    norm.weight = Param::new(Some(w));
                }
                if let Ok(b) = get_weight(&format!("feature_extractor.conv_layers.{}.layer_norm.bias", i)) {
                    norm.bias = Param::new(Some(b));
                }
            }
        }
    }

    // Load feature projection (with LayerNorm)
    if let Ok(w) = get_weight("feature_projection.layer_norm.weight") {
        model.feature_projection.layer_norm.weight = Param::new(Some(w));
    }
    if let Ok(b) = get_weight("feature_projection.layer_norm.bias") {
        model.feature_projection.layer_norm.bias = Param::new(Some(b));
    }
    if let Ok(w) = get_weight("feature_projection.projection.weight") {
        model.feature_projection.projection.weight = Param::new(w);
    }
    if let Ok(b) = get_weight("feature_projection.projection.bias") {
        model.feature_projection.projection.bias = Param::new(Some(b));
    }

    // Load positional conv embedding
    // Weight normalized conv has weight_g (magnitude) and weight_v (direction)
    if let Ok(g) = get_weight("encoder.pos_conv_embed.conv.weight_g") {
        model.pos_conv_embed.weight_g = Param::new(g);
    }
    if let Ok(v) = get_weight("encoder.pos_conv_embed.conv.weight_v") {
        model.pos_conv_embed.weight_v = Param::new(v);
    }
    if let Ok(b) = get_weight("encoder.pos_conv_embed.conv.bias") {
        model.pos_conv_embed.conv.bias = Param::new(Some(b));
    }

    // Load encoder layers
    for (i, layer) in model.encoder_layers.iter_mut().enumerate() {
        let prefix = format!("encoder.layers.{}", i);

        // Self attention
        if let Ok(w) = get_weight(&format!("{}.attention.q_proj.weight", prefix)) {
            layer.self_attn.q_proj.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.attention.q_proj.bias", prefix)) {
            layer.self_attn.q_proj.bias = Param::new(Some(b));
        }
        if let Ok(w) = get_weight(&format!("{}.attention.k_proj.weight", prefix)) {
            layer.self_attn.k_proj.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.attention.k_proj.bias", prefix)) {
            layer.self_attn.k_proj.bias = Param::new(Some(b));
        }
        if let Ok(w) = get_weight(&format!("{}.attention.v_proj.weight", prefix)) {
            layer.self_attn.v_proj.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.attention.v_proj.bias", prefix)) {
            layer.self_attn.v_proj.bias = Param::new(Some(b));
        }
        if let Ok(w) = get_weight(&format!("{}.attention.out_proj.weight", prefix)) {
            layer.self_attn.out_proj.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.attention.out_proj.bias", prefix)) {
            layer.self_attn.out_proj.bias = Param::new(Some(b));
        }

        // Layer norm after attention
        if let Ok(w) = get_weight(&format!("{}.layer_norm.weight", prefix)) {
            layer.self_attn_norm.weight = Param::new(Some(w));
        }
        if let Ok(b) = get_weight(&format!("{}.layer_norm.bias", prefix)) {
            layer.self_attn_norm.bias = Param::new(Some(b));
        }

        // FFN
        if let Ok(w) = get_weight(&format!("{}.feed_forward.intermediate_dense.weight", prefix)) {
            layer.ffn.fc1.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.feed_forward.intermediate_dense.bias", prefix)) {
            layer.ffn.fc1.bias = Param::new(Some(b));
        }
        if let Ok(w) = get_weight(&format!("{}.feed_forward.output_dense.weight", prefix)) {
            layer.ffn.fc2.weight = Param::new(w);
        }
        if let Ok(b) = get_weight(&format!("{}.feed_forward.output_dense.bias", prefix)) {
            layer.ffn.fc2.bias = Param::new(Some(b));
        }

        // Final layer norm (after FFN)
        if let Ok(w) = get_weight(&format!("{}.final_layer_norm.weight", prefix)) {
            layer.ffn_norm.weight = Param::new(Some(w));
        }
        if let Ok(b) = get_weight(&format!("{}.final_layer_norm.bias", prefix)) {
            layer.ffn_norm.bias = Param::new(Some(b));
        }
    }

    // Final layer norm
    if let Ok(w) = get_weight("encoder.layer_norm.weight") {
        model.layer_norm.weight = Param::new(Some(w));
    }
    if let Ok(b) = get_weight("encoder.layer_norm.bias") {
        model.layer_norm.bias = Param::new(Some(b));
    }

    // Output projection
    if let Some(ref mut proj) = model.output_projection {
        if let Ok(w) = get_weight("output_projection.weight") {
            proj.weight = Param::new(w);
        }
        if let Ok(b) = get_weight("output_projection.bias") {
            proj.bias = Param::new(Some(b));
        }
    }

    Ok(())
}

/// Load HuBERT model from safetensors file
pub fn load_hubert_model(weights_path: impl AsRef<Path>) -> Result<HuBertEncoder, Error> {
    let path = weights_path.as_ref();

    let config = HuBertConfig::default();
    let mut model = HuBertEncoder::new(config)?;

    let weights = Array::load_safetensors(path)?;
    load_hubert_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_hubert_config_default() {
        let config = HuBertConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.output_dim, 768);
    }

    #[test]
    fn test_audio_normalization() {
        let config = HuBertConfig::default();
        let encoder = HuBertEncoder::new(config).unwrap();

        // Test normalization
        let audio = Array::from_slice(&[0.1f32, -0.5, 0.3, 0.2, -0.1], &[1, 5]);
        let normalized = encoder.normalize_audio(&audio).unwrap();
        eval([&normalized]).unwrap();

        // Should have zero mean
        let mean: f32 = ops::mean_axis(&normalized, -1, false).unwrap().item();
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Should have unit variance
        let var: f32 = ops::var_axis(&normalized, -1, false, None).unwrap().item();
        assert!((var - 1.0).abs() < 0.1, "Var should be ~1, got {}", var);
    }

    #[test]
    fn test_conv_layer() {
        let mut layer = ConvLayer::new(1, 512, 10, 5, true, true).unwrap();

        // Input: [batch=1, samples=16000, channels=1]
        let x = Array::zeros::<f32>(&[1, 16000, 1]).unwrap();
        let output = layer.forward(&x).unwrap();
        eval([&output]).unwrap();

        // Output should be downsampled by stride 5
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[2], 512);
    }

    #[test]
    fn test_feature_extractor() {
        let config = HuBertConfig::default();
        let mut extractor = FeatureExtractor::new(&config).unwrap();

        // Input: [batch=1, samples=16000] - 1 second of audio
        let audio = Array::zeros::<f32>(&[1, 16000]).unwrap();
        let output = extractor.forward(&audio).unwrap();
        eval([&output]).unwrap();

        // Output should be approximately 49 frames for 1 second
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[2], config.conv_dim);
        let time_dim = output.shape()[1] as i32;
        assert!(time_dim >= 40 && time_dim <= 60, "Expected ~49 frames, got {}", time_dim);
    }

    #[test]
    fn test_encoder_attention() {
        let mut attn = EncoderAttention::new(768, 12).unwrap();

        let x = Array::zeros::<f32>(&[1, 50, 768]).unwrap();
        let output = attn.forward(&x).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 50, 768]);
    }

    #[test]
    fn test_encoder_layer() {
        let config = HuBertConfig::default();
        let mut layer = EncoderLayer::new(&config).unwrap();

        let x = Array::zeros::<f32>(&[1, 50, 768]).unwrap();
        let output = layer.forward(&x).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 50, 768]);
    }

    #[test]
    fn test_hubert_encoder() {
        // Use smaller config for faster test
        let config = HuBertConfig {
            num_layers: 2,  // Fewer layers for testing
            ..Default::default()
        };
        let mut encoder = HuBertEncoder::new(config).unwrap();

        // Input: 1 second of audio at 16kHz
        let audio = Array::zeros::<f32>(&[1, 16000]).unwrap();
        let output = encoder.forward(&audio).unwrap();
        eval([&output]).unwrap();

        // Output: [batch, time, 768]
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[2], 768);
        // Time dimension should be ~49 for 1 second
        let time_dim = output.shape()[1] as i32;
        assert!(time_dim >= 40 && time_dim <= 60, "Expected ~49 frames, got {}", time_dim);
    }
}
