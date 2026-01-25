//! VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
//!
//! This implements the SynthesizerTrn model from GPT-SoVITS for vocoding.
//!
//! Key components:
//! - ResidualVectorQuantizer: Decodes semantic codes to continuous representations
//! - TextEncoder (enc_p): Combines SSL features with text features via MRTE
//! - ResidualCouplingBlock (flow): Normalizing flow for latent transformation
//! - Generator (dec): HiFiGAN-style decoder for audio synthesis
//! - MelStyleEncoder (ref_enc): Extracts style embedding from reference mel

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis, exp, expand_dims, indexing::IndexOp, matmul, maximum, minimum,
        softmax_axis, split, sqrt, swap_axes, tanh, zeros_like,
    },
    random,
    Array,
};

use crate::error::Error;

/// Configuration for VITS model
#[derive(Debug, Clone)]
pub struct VITSConfig {
    /// Hidden channels (192 in GPT-SoVITS)
    pub hidden_channels: i32,
    /// SSL feature dimension (768 from CNHubert)
    pub ssl_dim: i32,
    /// Number of attention heads
    pub n_heads: i32,
    /// Number of encoder layers
    pub n_layers: i32,
    /// Filter channels in FFN
    pub filter_channels: i32,
    /// Kernel size in encoder
    pub kernel_size: i32,
    /// Number of flow layers
    pub n_flows: i32,
    /// Gin channels (style conditioning)
    pub gin_channels: i32,
    /// Text vocabulary size
    pub vocab_size: i32,
    /// Codebook size
    pub codebook_size: i32,
    /// Codebook dimension
    pub codebook_dim: i32,
    /// Upsample rates
    pub upsample_rates: Vec<i32>,
    /// Upsample kernel sizes
    pub upsample_kernel_sizes: Vec<i32>,
    /// Upsample initial channel
    pub upsample_initial_channel: i32,
    /// ResBlock kernel sizes
    pub resblock_kernel_sizes: Vec<i32>,
    /// ResBlock dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
}

impl Default for VITSConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 192,
            ssl_dim: 768,
            n_heads: 2,
            n_layers: 6,
            filter_channels: 768,
            kernel_size: 3,
            n_flows: 4,
            gin_channels: 512,
            vocab_size: 732,
            codebook_size: 1024,
            codebook_dim: 768,
            upsample_rates: vec![10, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 8, 2, 2],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        }
    }
}

// ============================================================================
// Residual Vector Quantizer
// ============================================================================

/// RVQ Codebook for decoding semantic codes
#[derive(Debug, Clone, ModuleParameters)]
pub struct RVQCodebook {
    #[param]
    pub embed: Param<Array>,
    pub codebook_size: i32,
    pub codebook_dim: i32,
}

impl RVQCodebook {
    pub fn new(codebook_size: i32, codebook_dim: i32) -> Result<Self, Exception> {
        let embed = Array::zeros::<f32>(&[codebook_size, codebook_dim])?;
        Ok(Self {
            embed: Param::new(embed),
            codebook_size,
            codebook_dim,
        })
    }

    /// Decode indices to embeddings
    /// Input: codes [n_q, batch, seq] or [batch, n_q, seq]
    /// Output: quantized [batch, dim, seq]
    pub fn decode(&self, codes: &Array) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;

        // codes shape: [1, 1, seq] from GPT-SoVITS typically
        let shape = codes.shape();

        // Flatten to get indices
        let indices = codes.flatten(None, None)?;
        let indices = indices.as_type::<i32>()?;

        // Gather embeddings using take_axis (embedding lookup)
        // embed: [codebook_size, codebook_dim], indices: [seq]
        // result: [seq, codebook_dim]
        let quantized = self.embed.take_axis(&indices, 0)?;
        eval([&quantized])?; // Force evaluation to materialize

        // Reshape: if input was [1, 1, seq], output should be [1, dim, seq]
        if shape.len() == 3 {
            let seq_len = shape[2] as i32;
            // quantized is [seq, dim] - we need [1, dim, seq]
            // First add batch dim: [1, seq, dim]
            let batched = quantized.reshape(&[1, seq_len, self.codebook_dim])?;
            // Then transpose last two dims: [1, seq, dim] -> [1, dim, seq]
            // Use transpose_axes for explicit permutation
            batched.transpose_axes(&[0, 2, 1])
        } else {
            Ok(quantized)
        }
    }

    /// Encode features to codebook indices (for few-shot mode)
    /// Input: features [batch, dim, seq]
    /// Output: codes [batch, 1, seq]
    ///
    /// This finds the nearest codebook entry for each feature vector.
    pub fn encode(&self, features: &Array) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;
        use mlx_rs::ops::{sum_axis, indexing::argmin_axis};

        let shape = features.shape();
        let batch = shape[0] as i32;
        let dim = shape[1] as i32;
        let seq = shape[2] as i32;

        // Transpose features: [batch, dim, seq] -> [batch, seq, dim]
        let features_t = features.transpose_axes(&[0, 2, 1])?;
        // Reshape to [batch * seq, dim]
        let flat_features = features_t.reshape(&[batch * seq, dim])?;

        // Compute L2 distances to each codebook entry
        // embed: [codebook_size, dim]
        // flat_features: [batch * seq, dim]
        //
        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        //
        // features_sq: [batch * seq, 1]
        let features_sq = sum_axis(&flat_features.multiply(&flat_features)?, -1, true)?;

        // embed_sq: [1, codebook_size]
        let embed_sq = sum_axis(&self.embed.multiply(&self.embed)?, -1, true)?;
        let embed_sq = embed_sq.transpose()?;

        // dot product: [batch * seq, dim] @ [dim, codebook_size] = [batch * seq, codebook_size]
        let embed_t = self.embed.transpose()?;
        let dot = matmul(&flat_features, &embed_t)?;

        // distances: [batch * seq, codebook_size]
        let distances = features_sq
            .add(&embed_sq)?
            .subtract(&dot.multiply(array!(2.0f32))?)?;

        eval([&distances])?;

        // Find argmin for each position
        // codes: [batch * seq]
        let codes = argmin_axis(&distances, -1, false)?;
        let codes = codes.as_type::<i32>()?;

        // Reshape to [batch, 1, seq]
        codes.reshape(&[batch, 1, seq])
    }
}

// ============================================================================
// Attention Layer (for transformer encoder)
// ============================================================================

/// Multi-head attention with relative positional encoding
#[derive(Debug, Clone, ModuleParameters)]
pub struct RelativeAttention {
    #[param]
    pub conv_q: nn::Conv1d,
    #[param]
    pub conv_k: nn::Conv1d,
    #[param]
    pub conv_v: nn::Conv1d,
    #[param]
    pub conv_o: nn::Conv1d,
    #[param]
    pub emb_rel_k: Param<Array>,
    #[param]
    pub emb_rel_v: Param<Array>,
    pub n_heads: i32,
    pub head_dim: i32,
    pub window_size: i32,
}

impl RelativeAttention {
    pub fn new(channels: i32, n_heads: i32) -> Result<Self, Exception> {
        Self::new_with_window(channels, n_heads, 4) // default window_size=4
    }

    pub fn new_with_window(channels: i32, n_heads: i32, window_size: i32) -> Result<Self, Exception> {
        let head_dim = channels / n_heads;

        let conv_q = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_k = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_v = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_o = nn::Conv1dBuilder::new(channels, channels, 1).build()?;

        // Relative position embeddings: [1, window*2+1, head_dim]
        let emb_size = window_size * 2 + 1;
        let emb_rel_k = Array::zeros::<f32>(&[1, emb_size, head_dim])?;
        let emb_rel_v = Array::zeros::<f32>(&[1, emb_size, head_dim])?;

        Ok(Self {
            conv_q,
            conv_k,
            conv_v,
            conv_o,
            emb_rel_k: Param::new(emb_rel_k),
            emb_rel_v: Param::new(emb_rel_v),
            n_heads,
            head_dim,
            window_size,
        })
    }

    /// Get relative embeddings for the given sequence length
    fn get_relative_embeddings(&self, rel_emb: &Array, length: i32) -> Result<Array, Exception> {
        let _max_rel_pos = 2 * self.window_size + 1;
        let pad_length = (length - (self.window_size + 1)).max(0);
        let slice_start = ((self.window_size + 1) - length).max(0);
        let slice_end = slice_start + 2 * length - 1;

        let padded = if pad_length > 0 {
            // Pad along the sequence dimension (dim 1)
            // rel_emb shape: [1, max_rel_pos, head_dim]
            let widths: &[(i32, i32)] = &[(0, 0), (pad_length, pad_length), (0, 0)];
            mlx_rs::ops::pad(rel_emb, widths, None, None)?
        } else {
            rel_emb.clone()
        };

        // Slice: padded[:, slice_start:slice_end, :]
        Ok(padded.index((.., slice_start..slice_end, ..)))
    }

    /// Matmul with relative keys: x[b,h,l,d] @ y[1,m,d].T -> [b,h,l,m]
    fn matmul_with_relative_keys(&self, x: &Array, y: &Array) -> Result<Array, Exception> {
        // y shape: [1, m, d] -> [1, 1, m, d] -> transpose to [1, 1, d, m]
        let y_exp = y.index((mlx_rs::ops::indexing::NewAxis, .., .., ..));
        let y_t = swap_axes(&y_exp, 2, 3)?;
        matmul(x, &y_t)
    }

    /// Matmul with relative values: x[b,h,l,m] @ y[1,m,d] -> [b,h,l,d]
    fn matmul_with_relative_values(&self, x: &Array, y: &Array) -> Result<Array, Exception> {
        // y shape: [1, m, d] -> [1, 1, m, d]
        let y_exp = y.index((mlx_rs::ops::indexing::NewAxis, .., .., ..));
        matmul(x, &y_exp)
    }

    /// Convert relative position to absolute position
    /// x: [b, h, l, 2*l-1] -> [b, h, l, l]
    fn relative_position_to_absolute_position(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let heads = shape[1] as i32;
        let length = shape[2] as i32;

        // Pad along last dim: [b, h, l, 2*l-1] -> [b, h, l, 2*l]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, 0), (0, 1)];
        let x_padded = mlx_rs::ops::pad(x, widths, None, None)?;

        // Reshape to [b, h, l * 2 * l]
        let x_flat = x_padded.reshape(&[batch, heads, length * 2 * length])?;

        // Pad: [b, h, l*2*l] -> [b, h, l*2*l + l - 1]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, length - 1)];
        let x_flat = mlx_rs::ops::pad(&x_flat, widths, None, None)?;

        // Reshape to [b, h, l+1, 2*l-1]
        let x_reshaped = x_flat.reshape(&[batch, heads, length + 1, 2 * length - 1])?;

        // Slice: [:, :, :length, length-1:]
        Ok(x_reshaped.index((.., .., ..length, (length - 1)..)))
    }

    /// Convert absolute position to relative position
    /// x: [b, h, l, l] -> [b, h, l, 2*l-1]
    fn absolute_position_to_relative_position(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let heads = shape[1] as i32;
        let length = shape[2] as i32;

        // Pad along last dim: [b, h, l, l] -> [b, h, l, 2*l-1]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, 0), (0, length - 1)];
        let x_padded = mlx_rs::ops::pad(x, widths, None, None)?;

        // Reshape to [b, h, l^2 + l*(l-1)]
        let flat_size = length * length + length * (length - 1);
        let x_flat = x_padded.reshape(&[batch, heads, flat_size])?;

        // Pad at beginning: [b, h, flat_size] -> [b, h, flat_size + length]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (length, 0)];
        let x_flat = mlx_rs::ops::pad(&x_flat, widths, None, None)?;

        // Reshape to [b, h, l, 2*l]
        let x_reshaped = x_flat.reshape(&[batch, heads, length, 2 * length])?;

        // Slice: [:, :, :, 1:]
        Ok(x_reshaped.index((.., .., .., 1..)))
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let channels = shape[1] as i32;
        let seq_len = shape[2] as i32;

        // Convert NCL to NLC for Conv1d (mlx-rs expects NLC)
        let x_nlc = swap_axes(x, 1, 2)?;

        // Q, K, V projections (input/output in NLC)
        let q = self.conv_q.forward(&x_nlc)?;
        let k = self.conv_k.forward(&x_nlc)?;
        let v = self.conv_v.forward(&x_nlc)?;

        // Convert NLC to NCL: [batch, seq, channels] -> [batch, channels, seq]
        let q = swap_axes(&q, 1, 2)?;
        let k = swap_axes(&k, 1, 2)?;
        let v = swap_axes(&v, 1, 2)?;

        // Reshape for multi-head: [batch, channels, seq] -> [batch, heads, head_dim, seq]
        let q = q.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;
        let k = k.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;
        let v = v.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q = swap_axes(&q, 2, 3)?;
        let k = swap_axes(&k, 2, 3)?;
        let v = swap_axes(&v, 2, 3)?;

        // Attention scores: [batch, heads, seq, seq]
        let scale = (self.head_dim as f32).sqrt();
        let q_scaled = q.divide(array!(scale))?;
        let scores = matmul(&q_scaled, &swap_axes(&k, 2, 3)?)?;

        // TODO: Re-enable relative position encoding after verifying baseline
        // Add relative position bias for keys
        // let rel_emb_k = self.get_relative_embeddings(&self.emb_rel_k, seq_len)?;
        // let rel_logits = self.matmul_with_relative_keys(&q_scaled, &rel_emb_k)?;
        // let scores_local = self.relative_position_to_absolute_position(&rel_logits)?;
        // let scores = scores.add(&scores_local)?;

        // Apply attention mask if provided
        // mask shape: [batch, 1, seq, seq] - positions with 0 are masked out
        let scores = if let Some(m) = mask {
            // scores.masked_fill(mask == 0, -1e4)
            let neg_inf = array!(-1e4f32);
            let zero = array!(0.0f32);
            let mask_zero = m.eq(&zero)?;
            mlx_rs::ops::r#where(&mask_zero, &neg_inf, &scores)?
        } else {
            scores
        };

        // Softmax
        let attn = softmax_axis(&scores, -1, false)?;

        // Apply to values: [batch, heads, seq, head_dim]
        let out = matmul(&attn, &v)?;

        // TODO: Re-enable relative position encoding for values
        // Add relative position bias for values
        // let rel_weights = self.absolute_position_to_relative_position(&attn)?;
        // let rel_emb_v = self.get_relative_embeddings(&self.emb_rel_v, seq_len)?;
        // let rel_values = self.matmul_with_relative_values(&rel_weights, &rel_emb_v)?;
        // out = out.add(&rel_values)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, channels, seq]
        let out = swap_axes(&out, 2, 3)?;
        let out = out.reshape(&[batch, channels, seq_len])?;

        // Convert NCL to NLC for output projection
        let out_nlc = swap_axes(&out, 1, 2)?;
        let out_nlc = self.conv_o.forward(&out_nlc)?;

        // Convert back to NCL
        swap_axes(&out_nlc, 1, 2)
    }

    /// Cross-attention: Q from x, K/V from c (both NCL format)
    /// attn_mask shape: [batch, 1, q_len, kv_len] - positions with 0 are masked out
    pub fn cross_forward(&mut self, x: &Array, c: &Array, attn_mask: Option<&Array>) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let c_shape = c.shape();
        let batch = x_shape[0] as i32;
        let channels = x_shape[1] as i32;
        let q_len = x_shape[2] as i32;  // SSL sequence length
        let kv_len = c_shape[2] as i32;  // Text sequence length

        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let c_nlc = swap_axes(c, 1, 2)?;

        // Q from x (query), K/V from c (key/value)
        let q = self.conv_q.forward(&x_nlc)?;
        let k = self.conv_k.forward(&c_nlc)?;
        let v = self.conv_v.forward(&c_nlc)?;

        // Convert NLC to NCL
        let q = swap_axes(&q, 1, 2)?;
        let k = swap_axes(&k, 1, 2)?;
        let v = swap_axes(&v, 1, 2)?;

        // Reshape for multi-head
        let q = q.reshape(&[batch, self.n_heads, self.head_dim, q_len])?;
        let k = k.reshape(&[batch, self.n_heads, self.head_dim, kv_len])?;
        let v = v.reshape(&[batch, self.n_heads, self.head_dim, kv_len])?;

        // Transpose: [batch, heads, seq, head_dim]
        let q = swap_axes(&q, 2, 3)?;
        let k = swap_axes(&k, 2, 3)?;
        let v = swap_axes(&v, 2, 3)?;

        // Cross-attention scores: [batch, heads, q_len, kv_len]
        let scale = (self.head_dim as f32).sqrt();
        let scores = matmul(&q, &swap_axes(&k, 2, 3)?)?;
        let scores = scores.divide(array!(scale))?;

        // Apply attention mask: scores.masked_fill(mask == 0, -1e4)
        let scores = if let Some(mask) = attn_mask {
            // mask shape: [batch, 1, q_len, kv_len]
            // Create large negative value where mask == 0
            let neg_inf = array!(-1e4f32);
            let zero = array!(0.0f32);
            // where(mask == 0, -1e4, scores)
            let mask_bool = mask.eq(&zero)?;
            mlx_rs::ops::r#where(&mask_bool, &neg_inf, &scores)?
        } else {
            scores
        };

        // Softmax
        let attn = softmax_axis(&scores, -1, false)?;

        // Apply to values: [batch, heads, q_len, head_dim]
        let out = matmul(&attn, &v)?;

        // Reshape back: [batch, heads, q_len, head_dim] -> [batch, channels, q_len]
        let out = swap_axes(&out, 2, 3)?;
        let out = out.reshape(&[batch, channels, q_len])?;

        // Convert NCL to NLC for output projection
        let out_nlc = swap_axes(&out, 1, 2)?;
        let out_nlc = self.conv_o.forward(&out_nlc)?;

        // Convert back to NCL
        swap_axes(&out_nlc, 1, 2)
    }
}

// ============================================================================
// FFN Layer (Feed-Forward Network)
// ============================================================================

/// Feed-forward network with Conv1d
#[derive(Debug, Clone, ModuleParameters)]
pub struct FFN {
    #[param]
    pub conv_1: nn::Conv1d,
    #[param]
    pub conv_2: nn::Conv1d,
    pub kernel_size: i32,
}

impl FFN {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        filter_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        let padding = (kernel_size - 1) / 2;
        let conv_1 = nn::Conv1dBuilder::new(in_channels, filter_channels, kernel_size)
            .padding(padding)
            .build()?;
        let conv_2 = nn::Conv1dBuilder::new(filter_channels, out_channels, kernel_size)
            .padding(padding)
            .build()?;

        Ok(Self {
            conv_1,
            conv_2,
            kernel_size,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let mask_nlc = swap_axes(mask, 1, 2)?;

        let x = self.conv_1.forward(&x_nlc)?;
        let x = nn::relu(&x)?;
        let x = x.multiply(&mask_nlc)?;
        let x = self.conv_2.forward(&x)?;
        let x = x.multiply(&mask_nlc)?;

        // Convert back to NCL
        swap_axes(&x, 1, 2)
    }
}

// ============================================================================
// Transformer Encoder
// ============================================================================

/// Layer normalization for conv inputs (channels-first)
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvLayerNorm {
    #[param]
    pub gamma: Param<Array>,
    #[param]
    pub beta: Param<Array>,
    pub channels: i32,
    pub eps: f32,
}

impl ConvLayerNorm {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        let gamma = Array::ones::<f32>(&[channels])?;
        let beta = Array::zeros::<f32>(&[channels])?;
        Ok(Self {
            gamma: Param::new(gamma),
            beta: Param::new(beta),
            channels,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // x: [batch, channels, seq]
        // Transpose to [batch, seq, channels], normalize, transpose back
        let x = swap_axes(x, 1, 2)?;

        // Manual layer norm along last dimension
        let mean = x.mean_axis(-1, true)?;
        let x_centered = x.subtract(&mean)?;
        let var = x_centered.square()?.mean_axis(-1, true)?;
        let x_norm = x_centered.divide(&sqrt(&var.add(array!(self.eps))?)?)?;

        // Apply scale and bias
        // gamma and beta are [channels], need [1, 1, channels] for broadcasting
        let gamma = self.gamma.reshape(&[1, 1, self.channels])?;
        let beta = self.beta.reshape(&[1, 1, self.channels])?;
        let out = x_norm.multiply(&gamma)?.add(&beta)?;

        // Transpose back
        swap_axes(&out, 1, 2)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderLayer {
    #[param]
    pub attn: RelativeAttention,
    #[param]
    pub ffn: FFN,
    #[param]
    pub norm1: ConvLayerNorm,
    #[param]
    pub norm2: ConvLayerNorm,
}

impl EncoderLayer {
    pub fn new(
        channels: i32,
        n_heads: i32,
        filter_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        let attn = RelativeAttention::new(channels, n_heads)?;
        let ffn = FFN::new(channels, channels, filter_channels, kernel_size)?;
        let norm1 = ConvLayerNorm::new(channels)?;
        let norm2 = ConvLayerNorm::new(channels)?;

        Ok(Self {
            attn,
            ffn,
            norm1,
            norm2,
        })
    }

    /// Forward pass - POST-NORM version (matching Python GPT-SoVITS)
    /// Using norm(x + attn(x)) instead of x + attn(norm(x))
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        // POST-NORM: x = norm1(x + attn(x))
        let attn_out = self.attn.forward(x, None)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;

        // x = norm2(x + ffn(x))
        let ffn_out = self.ffn.forward(&x, mask)?;
        self.norm2.forward(&x.add(&ffn_out)?)
    }
}

/// Transformer encoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerEncoder {
    #[param]
    pub layers: Vec<EncoderLayer>,
    pub n_layers: i32,
}

impl TransformerEncoder {
    pub fn new(
        channels: i32,
        n_heads: i32,
        filter_channels: i32,
        kernel_size: i32,
        n_layers: i32,
    ) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(n_layers as usize);
        for _ in 0..n_layers {
            layers.push(EncoderLayer::new(
                channels,
                n_heads,
                filter_channels,
                kernel_size,
            )?);
        }
        Ok(Self { layers, n_layers })
    }

    /// Forward pass - simple version without explicit attention mask
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask)?;
        }
        Ok(h)
    }
}

// ============================================================================
// MRTE (Multi-Resolution Temporal Encoder) for cross-attention
// ============================================================================

/// Cross-attention for combining SSL and text features
#[derive(Debug, Clone, ModuleParameters)]
pub struct MRTECrossAttention {
    #[param]
    pub c_pre: nn::Conv1d,
    #[param]
    pub text_pre: nn::Conv1d,
    #[param]
    pub cross_attention: RelativeAttention,
    #[param]
    pub c_post: nn::Conv1d,
    pub channels: i32,
    pub hidden: i32,
}

impl MRTECrossAttention {
    pub fn new(channels: i32, hidden: i32, n_heads: i32) -> Result<Self, Exception> {
        let c_pre = nn::Conv1dBuilder::new(channels, hidden, 1).build()?;
        let text_pre = nn::Conv1dBuilder::new(channels, hidden, 1).build()?;
        let cross_attention = RelativeAttention::new(hidden, n_heads)?;
        let c_post = nn::Conv1dBuilder::new(hidden, channels, 1).build()?;

        Ok(Self {
            c_pre,
            text_pre,
            cross_attention,
            c_post,
            channels,
            hidden,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    /// Cross-attention: SSL features (query) attend to text features (key/value)
    ///
    /// Following actual GPT-SoVITS implementation:
    /// 1. Apply mask BEFORE c_pre/text_pre convolutions
    /// 2. Create attention mask from ssl_mask and text_mask
    /// 3. Apply mask BEFORE c_post convolution
    pub fn forward(
        &mut self,
        ssl_features: &Array,
        ssl_mask: &Array,
        text_features: &Array,
        text_mask: &Array,
        style: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Create attention mask: text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        // text_mask: [batch, 1, text_len] -> [batch, 1, 1, text_len]
        // ssl_mask: [batch, 1, ssl_len] -> [batch, 1, ssl_len, 1]
        // attn_mask: [batch, 1, ssl_len, text_len]
        let text_mask_4d = expand_dims(text_mask, 2)?;  // [batch, 1, 1, text_len]
        let ssl_mask_4d = expand_dims(ssl_mask, -1)?;   // [batch, 1, ssl_len, 1]
        let attn_mask = text_mask_4d.multiply(&ssl_mask_4d)?;  // [batch, 1, ssl_len, text_len]

        // Apply mask BEFORE c_pre (following actual GPT-SoVITS)
        let ssl_masked_input = ssl_features.multiply(ssl_mask)?;
        let text_masked_input = text_features.multiply(text_mask)?;

        // Convert NCL to NLC for Conv1d
        let ssl_nlc = swap_axes(&ssl_masked_input, 1, 2)?;
        let text_nlc = swap_axes(&text_masked_input, 1, 2)?;

        // Project features (NLC format for mlx-rs Conv1d)
        let ssl_proj = self.c_pre.forward(&ssl_nlc)?;
        let text_proj = self.text_pre.forward(&text_nlc)?;

        // Convert back to NCL for attention
        let ssl_ncl = swap_axes(&ssl_proj, 1, 2)?;  // [batch, hidden, ssl_seq]
        let text_ncl = swap_axes(&text_proj, 1, 2)?;  // [batch, hidden, text_seq]

        // Apply masks again for cross-attention input (following actual GPT-SoVITS)
        let ssl_masked = ssl_ncl.multiply(ssl_mask)?;
        let text_masked = text_ncl.multiply(text_mask)?;

        // Cross-attention: Q from SSL, K/V from text, with attention mask
        let attn_out = self.cross_attention.cross_forward(&ssl_masked, &text_masked, Some(&attn_mask))?;

        // Add residual from projected SSL
        let attn_out = attn_out.add(&ssl_masked)?;

        // Add style embedding if provided (ge=0 if None in Python)
        let attn_out = if let Some(ge) = style {
            attn_out.add(ge)?
        } else {
            attn_out
        };

        // Apply mask BEFORE c_post (following actual GPT-SoVITS)
        let attn_masked = attn_out.multiply(ssl_mask)?;

        // Convert NCL to NLC for output projection
        let attn_nlc = swap_axes(&attn_masked, 1, 2)?;
        let out = self.c_post.forward(&attn_nlc)?;
        // Convert back to NCL
        swap_axes(&out, 1, 2)
    }
}

// ============================================================================
// TextEncoder (enc_p)
// ============================================================================

/// TextEncoder: Combines SSL features with text phoneme features
#[derive(Debug, Clone, ModuleParameters)]
pub struct TextEncoder {
    #[param]
    pub ssl_proj: nn::Conv1d,
    #[param]
    pub encoder_ssl: TransformerEncoder,
    #[param]
    pub text_embedding: nn::Embedding,
    #[param]
    pub encoder_text: TransformerEncoder,
    #[param]
    pub mrte: MRTECrossAttention,
    #[param]
    pub encoder2: TransformerEncoder,
    #[param]
    pub proj: nn::Conv1d,
    pub out_channels: i32,
}

impl TextEncoder {
    pub fn new(config: &VITSConfig) -> Result<Self, Exception> {
        let ssl_proj = nn::Conv1dBuilder::new(config.ssl_dim, config.hidden_channels, 1).build()?;

        let encoder_ssl = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers / 2,
        )?;

        let text_embedding = nn::Embedding::new(config.vocab_size, config.hidden_channels)?;

        let encoder_text = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers,
        )?;

        let mrte = MRTECrossAttention::new(config.hidden_channels, config.gin_channels, 4)?;

        let encoder2 = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers / 2,
        )?;

        // Output: mean and log_var (2 * hidden_channels)
        let proj =
            nn::Conv1dBuilder::new(config.hidden_channels, config.hidden_channels * 2, 1).build()?;

        Ok(Self {
            ssl_proj,
            encoder_ssl,
            text_embedding,
            encoder_text,
            mrte,
            encoder2,
            proj,
            out_channels: config.hidden_channels,
        })
    }

    /// Forward pass (matching actual GPT-SoVITS TextEncoder.forward)
    /// - quantized: [batch, ssl_dim, seq] from RVQ decode (NCL format)
    /// - text: [batch, text_seq] phoneme indices
    /// - style: [batch, gin_channels, 1] style embedding
    /// Returns: (encoded, mean, log_var, mask) all in NCL format
    pub fn forward(
        &mut self,
        quantized: &Array,
        text: &Array,
        style: Option<&Array>,
    ) -> Result<(Array, Array, Array, Array), Exception> {
        let batch = quantized.shape()[0] as i32;
        let seq_len = quantized.shape()[2] as i32;

        // Create masks
        // NCL format mask for convolutions and encoder
        let y_mask = Array::ones::<f32>(&[batch, 1, seq_len])?;

        // Step 1: ssl_proj with mask before AND after (matching Python)
        // Python: y = self.ssl_proj(y * y_mask) * y_mask
        let quantized_masked = quantized.multiply(&y_mask)?;  // mask BEFORE ssl_proj
        let quantized_nlc = swap_axes(&quantized_masked, 1, 2)?;
        let ssl = self.ssl_proj.forward(&quantized_nlc)?;
        let mask_nlc = swap_axes(&y_mask, 1, 2)?;
        let ssl = ssl.multiply(&mask_nlc)?;  // mask AFTER ssl_proj
        let ssl_ncl = swap_axes(&ssl, 1, 2)?;

        // Step 2: encoder_ssl with mask before (matching Python)
        // Python: y = self.encoder_ssl(y * y_mask, y_mask)
        let ssl_masked = ssl_ncl.multiply(&y_mask)?;  // mask BEFORE encoder_ssl
        let ssl_enc = self.encoder_ssl.forward(&ssl_masked, &y_mask)?;

        // Step 3: text embedding and encoder_text with mask before
        // Python: text = self.encoder_text(text * text_mask, text_mask)
        let text_seq_len = text.shape()[1] as i32;
        let text_mask = Array::ones::<f32>(&[batch, 1, text_seq_len])?;
        let text_embed = self.text_embedding.forward(text)?;
        // [batch, seq, channels] -> [batch, channels, seq]
        let text_embed = swap_axes(&text_embed, 1, 2)?;
        let text_masked = text_embed.multiply(&text_mask)?;  // mask BEFORE encoder_text
        let text_enc = self.encoder_text.forward(&text_masked, &text_mask)?;

        // Step 4: MRTE (already fixed to match actual GPT-SoVITS)
        let mrte_out = self.mrte.forward(&ssl_enc, &y_mask, &text_enc, &text_mask, style)?;

        // Step 5: encoder2 with mask before (matching Python)
        // Python: y = self.encoder2(y * y_mask, y_mask)
        let mrte_masked = mrte_out.multiply(&y_mask)?;  // mask BEFORE encoder2
        let encoded = self.encoder2.forward(&mrte_masked, &y_mask)?;

        // Step 6: output projection
        // Python: stats = self.proj(y) * y_mask
        let encoded_nlc = swap_axes(&encoded, 1, 2)?;
        let stats = self.proj.forward(&encoded_nlc)?;
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(&y_mask)?;

        // Split into mean and log_var
        let halves = split(&stats, 2, 1)?;
        let mean = halves[0].clone();
        let log_var = halves[1].clone();

        Ok((encoded, mean, log_var, y_mask))
    }

    /// Debug forward that returns all intermediate outputs
    pub fn forward_debug(
        &mut self,
        quantized: &Array,
        text: &Array,
        style: Option<&Array>,
    ) -> Result<Vec<(String, Array)>, Exception> {
        let mut outputs = Vec::new();

        let batch = quantized.shape()[0] as i32;
        let seq_len = quantized.shape()[2] as i32;

        // Create masks
        let mask_nlc = Array::ones::<f32>(&[batch, seq_len, 1])?;
        let mask_ncl = Array::ones::<f32>(&[batch, 1, seq_len])?;
        outputs.push(("step0_y_mask".to_string(), mask_ncl.clone()));

        // Step 1: ssl_proj
        let quantized_nlc = swap_axes(quantized, 1, 2)?;
        outputs.push(("step1_ssl_proj_input".to_string(), quantized.clone()));

        let ssl = self.ssl_proj.forward(&quantized_nlc)?;
        let ssl = ssl.multiply(&mask_nlc)?;
        let ssl_ncl = swap_axes(&ssl, 1, 2)?;
        outputs.push(("step1_ssl_proj_output".to_string(), ssl_ncl.clone()));

        // Step 2: encoder_ssl
        let mask_enc = Array::ones::<f32>(&[batch, 1, seq_len])?;
        outputs.push(("step2_encoder_ssl_input".to_string(), ssl_ncl.clone()));

        let ssl_ncl = self.encoder_ssl.forward(&ssl_ncl, &mask_enc)?;
        outputs.push(("step2_encoder_ssl_output".to_string(), ssl_ncl.clone()));

        // Step 3: text_embedding and encoder_text
        let text_seq_len = text.shape()[1] as i32;
        let text_mask = Array::ones::<f32>(&[batch, 1, text_seq_len])?;
        outputs.push(("step3_text_mask".to_string(), text_mask.clone()));

        let text_embed = self.text_embedding.forward(text)?;
        let text_embed = swap_axes(&text_embed, 1, 2)?;
        outputs.push(("step3_text_embed".to_string(), text_embed.clone()));

        let text_encoded = self.encoder_text.forward(&text_embed, &text_mask)?;
        outputs.push(("step3_text_encoded".to_string(), text_encoded.clone()));

        // Step 4: mrte
        let combined = self.mrte.forward(&ssl_ncl, &mask_enc, &text_encoded, &text_mask, style)?;
        outputs.push(("step4_mrte_output".to_string(), combined.clone()));

        // Step 5: encoder2
        let enc2_input = combined.multiply(&mask_enc)?;
        outputs.push(("step5_encoder2_input".to_string(), enc2_input.clone()));

        let encoded = self.encoder2.forward(&enc2_input, &mask_enc)?;
        outputs.push(("step5_encoder2_output".to_string(), encoded.clone()));

        // Step 6: proj
        let encoded_nlc = swap_axes(&encoded, 1, 2)?;
        let stats = self.proj.forward(&encoded_nlc)?;
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(&mask_ncl)?;
        outputs.push(("step6_proj_output".to_string(), stats.clone()));

        let halves = split(&stats, 2, 1)?;
        outputs.push(("step6_m_p".to_string(), halves[0].clone()));
        outputs.push(("step6_logs_p".to_string(), halves[1].clone()));

        Ok(outputs)
    }
}

// ============================================================================
// WN (WaveNet-style) encoder for flow
// ============================================================================

/// WaveNet-style network for flow coupling layers
#[derive(Debug, Clone, ModuleParameters)]
pub struct WNEncoder {
    #[param]
    pub in_layers: Vec<nn::Conv1d>,
    #[param]
    pub res_skip_layers: Vec<nn::Conv1d>,
    #[param]
    pub cond_layer: nn::Conv1d,
    pub n_layers: i32,
    pub hidden_channels: i32,
}

impl WNEncoder {
    pub fn new(
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        gin_channels: i32,
    ) -> Result<Self, Exception> {
        let padding = (kernel_size - 1) / 2;
        let mut in_layers = Vec::with_capacity(n_layers as usize);
        let mut res_skip_layers = Vec::with_capacity(n_layers as usize);

        for i in 0..n_layers {
            let dilation = 1; // Simplified: use dilation 1
            let in_layer = nn::Conv1dBuilder::new(hidden_channels, hidden_channels * 2, kernel_size)
                .padding(padding * dilation)
                .dilation(dilation)
                .build()?;
            in_layers.push(in_layer);

            // Last layer outputs hidden_channels, others output hidden_channels * 2
            let out_ch = if i < n_layers - 1 {
                hidden_channels * 2
            } else {
                hidden_channels
            };
            let res_skip = nn::Conv1dBuilder::new(hidden_channels, out_ch, 1).build()?;
            res_skip_layers.push(res_skip);
        }

        let cond_layer =
            nn::Conv1dBuilder::new(gin_channels, hidden_channels * 2 * n_layers, 1).build()?;

        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            n_layers,
            hidden_channels,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut output = zeros_like(x)?;

        // Condition on style (NCL -> NLC -> NCL for conv)
        let g_cond = if let Some(style) = g {
            let style_nlc = swap_axes(style, 1, 2)?;
            let cond = self.cond_layer.forward(&style_nlc)?;
            Some(swap_axes(&cond, 1, 2)?) // Back to NCL
        } else {
            None
        };

        let mask_nlc = swap_axes(mask, 1, 2)?;
        let mut h = x.clone();

        for (i, (in_layer, res_skip)) in self
            .in_layers
            .iter_mut()
            .zip(self.res_skip_layers.iter_mut())
            .enumerate()
        {
            // Convert to NLC for conv
            let h_nlc = swap_axes(&h, 1, 2)?;
            let h_in_nlc = in_layer.forward(&h_nlc)?;
            let h_in = swap_axes(&h_in_nlc, 1, 2)?; // Back to NCL

            // Add conditioning if available (both in NCL)
            let h_in = if let Some(ref g) = g_cond {
                let g_slice =
                    g.index((.., i as i32 * self.hidden_channels * 2..(i as i32 + 1) * self.hidden_channels * 2, ..));
                h_in.add(&g_slice)?
            } else {
                h_in
            };

            // Gated activation (NCL format, split on channel dim 1)
            let halves = split(&h_in, 2, 1)?;
            let h_tanh = tanh(&halves[0])?;
            let h_sigmoid = nn::sigmoid(&halves[1])?;
            let acts = h_tanh.multiply(&h_sigmoid)?; // NCL

            // Residual and skip connection (convert to NLC for conv)
            let acts_nlc = swap_axes(&acts, 1, 2)?;
            let res_skip_out_nlc = res_skip.forward(&acts_nlc)?;
            let res_skip_out = swap_axes(&res_skip_out_nlc, 1, 2)?; // Back to NCL

            if i < (self.n_layers - 1) as usize {
                let res_skip_halves = split(&res_skip_out, 2, 1)?;
                // Python: x = (x + res_acts) * x_mask
                h = h.add(&res_skip_halves[0])?.multiply(mask)?;
                output = output.add(&res_skip_halves[1])?;
            } else {
                output = output.add(&res_skip_out)?;
            }
        }

        output.multiply(mask)
    }
}

// ============================================================================
// ResidualCouplingLayer
// ============================================================================

/// Residual coupling layer for normalizing flow
#[derive(Debug, Clone, ModuleParameters)]
pub struct ResidualCouplingLayer {
    #[param]
    pub pre: nn::Conv1d,
    #[param]
    pub enc: WNEncoder,
    #[param]
    pub post: nn::Conv1d,
    pub half_channels: i32,
    pub mean_only: bool,
}

impl ResidualCouplingLayer {
    pub fn new(
        channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        gin_channels: i32,
        mean_only: bool,
    ) -> Result<Self, Exception> {
        let half_channels = channels / 2;

        let pre = nn::Conv1dBuilder::new(half_channels, hidden_channels, 1).build()?;

        let enc = WNEncoder::new(hidden_channels, kernel_size, n_layers, gin_channels)?;

        let post_out = if mean_only {
            half_channels
        } else {
            half_channels * 2
        };
        let post = nn::Conv1dBuilder::new(hidden_channels, post_out, 1).build()?;

        Ok(Self {
            pre,
            enc,
            post,
            half_channels,
            mean_only,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
        reverse: bool,
    ) -> Result<Array, Exception> {
        // Split input (NCL format)
        let x0 = x.index((.., ..self.half_channels, ..));
        let x1 = x.index((.., self.half_channels.., ..));

        // Convert NCL to NLC for pre conv
        let x0_nlc = swap_axes(&x0, 1, 2)?;
        let h = self.pre.forward(&x0_nlc)?;
        // Back to NCL
        let h = swap_axes(&h, 1, 2)?;
        let h = h.multiply(mask)?;

        // WNEncoder forward (expects/returns NCL)
        let h = self.enc.forward(&h, mask, g)?;

        // Convert NCL to NLC for post conv
        let h_nlc = swap_axes(&h, 1, 2)?;
        let stats = self.post.forward(&h_nlc)?;
        // Back to NCL
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(mask)?;

        let m = if self.mean_only {
            stats
        } else {
            let halves = split(&stats, 2, 1)?;
            halves[0].clone()
        };

        // Apply coupling
        let x1 = if reverse {
            x1.subtract(&m)?.multiply(mask)?
        } else {
            x1.add(&m)?.multiply(mask)?
        };

        // Concatenate
        concatenate_axis(&[&x0, &x1], 1)
    }
}

// ============================================================================
// ResidualCouplingBlock (flow)
// ============================================================================

/// Flow model with residual coupling layers
#[derive(Debug, Clone, ModuleParameters)]
pub struct ResidualCouplingBlock {
    #[param]
    pub flows: Vec<ResidualCouplingLayer>,
    pub n_flows: i32,
}

impl ResidualCouplingBlock {
    pub fn new(
        channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        n_flows: i32,
        gin_channels: i32,
    ) -> Result<Self, Exception> {
        let mut flows = Vec::with_capacity(n_flows as usize);
        for _ in 0..n_flows {
            flows.push(ResidualCouplingLayer::new(
                channels,
                hidden_channels,
                kernel_size,
                n_layers,
                gin_channels,
                true, // mean_only
            )?);
        }
        Ok(Self { flows, n_flows })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
        reverse: bool,
    ) -> Result<Array, Exception> {
        let mut h = x.clone();

        // Helper to flip channels (reverse along dim 1)
        fn flip_channels(x: &Array) -> Result<Array, Exception> {
            let n_channels = x.shape()[1] as i32;
            // Create reversed indices: [n-1, n-2, ..., 1, 0]
            let indices = Array::from_iter((0..n_channels).rev(), &[n_channels]);
            x.take_axis(&indices, 1)
        }

        if reverse {
            for flow in self.flows.iter_mut().rev() {
                // Flip: reverse entire channel dimension (like torch.flip(x, [1]))
                h = flip_channels(&h)?;
                // Apply coupling
                h = flow.forward(&h, mask, g, true)?;
            }
        } else {
            for flow in &mut self.flows {
                h = flow.forward(&h, mask, g, false)?;
                // Flip: reverse entire channel dimension
                h = flip_channels(&h)?;
            }
        }

        Ok(h)
    }
}

// ============================================================================
// HiFiGAN Generator (dec)
// ============================================================================

/// ResBlock for HiFiGAN
#[derive(Debug, Clone, ModuleParameters)]
pub struct HiFiGANResBlock {
    #[param]
    pub convs1: Vec<nn::Conv1d>,
    #[param]
    pub convs2: Vec<nn::Conv1d>,
}

impl HiFiGANResBlock {
    pub fn new(channels: i32, kernel_size: i32, dilations: &[i32]) -> Result<Self, Exception> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        for &d in dilations {
            let padding = (kernel_size - 1) * d / 2;
            convs1.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding(padding)
                    .dilation(d)
                    .build()?,
            );
            convs2.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding((kernel_size - 1) / 2)
                    .build()?,
            );
        }

        Ok(Self { convs1, convs2 })
    }

    /// Forward pass (expects NLC input, returns NLC output)
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Process through all dilations with skip connection at each step
        // Matching Python: x = xt + x inside the loop
        let mut h = x.clone();
        for (c1, c2) in self.convs1.iter_mut().zip(self.convs2.iter_mut()) {
            let xt = nn::leaky_relu(&h, 0.1)?;
            let xt = c1.forward(&xt)?;
            let xt = nn::leaky_relu(&xt, 0.1)?;
            let xt = c2.forward(&xt)?;
            h = xt.add(&h)?;  // Skip connection inside loop
        }
        Ok(h)
    }
}

/// HiFiGAN Generator
#[derive(Debug, Clone, ModuleParameters)]
pub struct HiFiGANGenerator {
    #[param]
    pub conv_pre: nn::Conv1d,
    #[param]
    pub ups: Vec<nn::ConvTranspose1d>,
    #[param]
    pub resblocks: Vec<HiFiGANResBlock>,
    #[param]
    pub conv_post: nn::Conv1d,
    #[param]
    pub cond: nn::Conv1d,
    pub num_kernels: i32,
    pub num_upsamples: i32,
}

impl HiFiGANGenerator {
    pub fn new(config: &VITSConfig) -> Result<Self, Exception> {
        let conv_pre = nn::Conv1dBuilder::new(
            config.hidden_channels,
            config.upsample_initial_channel,
            7,
        )
        .padding(3)
        .build()?;

        let mut ups = Vec::new();
        let mut ch = config.upsample_initial_channel;
        for (i, (&u, &k)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_ch = ch / 2;
            ups.push(
                nn::ConvTranspose1dBuilder::new(ch, out_ch, k)
                    .stride(u)
                    .padding((k - u) / 2)
                    .build()?,
            );
            ch = out_ch;
        }

        let mut resblocks = Vec::new();
        ch = config.upsample_initial_channel;
        for i in 0..config.upsample_rates.len() {
            ch = ch / 2;
            for (j, (k, d)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(HiFiGANResBlock::new(ch, *k, d)?);
            }
        }

        let final_ch = config.upsample_initial_channel
            / (2_i32.pow(config.upsample_rates.len() as u32));
        let conv_post = nn::Conv1dBuilder::new(final_ch, 1, 7)
            .padding(3)
            .build()?;

        let cond =
            nn::Conv1dBuilder::new(config.gin_channels, config.upsample_initial_channel, 1)
                .build()?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond,
            num_kernels: config.resblock_kernel_sizes.len() as i32,
            num_upsamples: config.upsample_rates.len() as i32,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, g: Option<&Array>) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let mut h = self.conv_pre.forward(&x_nlc)?;

        // Add style conditioning (also in NLC)
        if let Some(style) = g {
            let style_nlc = swap_axes(style, 1, 2)?;
            let cond = self.cond.forward(&style_nlc)?;
            h = h.add(&cond)?;
        }

        let mut resblock_idx = 0;
        for up in self.ups.iter_mut() {
            h = nn::leaky_relu(&h, 0.1)?;
            h = up.forward(&h)?;

            // Apply resblocks (all in NLC)
            let mut xs = None::<Array>;
            for _ in 0..self.num_kernels {
                if resblock_idx < self.resblocks.len() {
                    let rb_out = self.resblocks[resblock_idx].forward(&h)?;
                    xs = Some(match xs {
                        Some(acc) => acc.add(&rb_out)?,
                        None => rb_out,
                    });
                    resblock_idx += 1;
                }
            }

            if let Some(x_sum) = xs {
                h = x_sum.divide(array!(self.num_kernels as f32))?;
            }
        }

        h = nn::leaky_relu(&h, 0.1)?;
        h = self.conv_post.forward(&h)?;
        let h = tanh(&h)?;

        // Convert NLC back to NCL
        swap_axes(&h, 1, 2)
    }
}

// ============================================================================
// MelStyleEncoder (ref_enc)
// ============================================================================

/// MelStyleEncoder for extracting style from reference mel spectrogram
#[derive(Debug, Clone, ModuleParameters)]
pub struct MelStyleEncoder {
    #[param]
    pub spectral_0: nn::Linear,
    #[param]
    pub spectral_1: nn::Linear,
    #[param]
    pub temporal_0: nn::Conv1d,
    #[param]
    pub temporal_1: nn::Conv1d,
    #[param]
    pub slf_attn_q: nn::Linear,
    #[param]
    pub slf_attn_k: nn::Linear,
    #[param]
    pub slf_attn_v: nn::Linear,
    #[param]
    pub slf_attn_fc: nn::Linear,
    #[param]
    pub fc: nn::Linear,
    pub hidden_dim: i32,
    pub out_dim: i32,
}

impl MelStyleEncoder {
    pub fn new(mel_channels: i32, hidden_dim: i32, out_dim: i32) -> Result<Self, Exception> {
        let spectral_0 = nn::LinearBuilder::new(mel_channels, hidden_dim)
            .bias(true)
            .build()?;
        let spectral_1 = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;

        // GLU convolutions
        let temporal_0 = nn::Conv1dBuilder::new(hidden_dim, hidden_dim * 2, 5)
            .padding(2)
            .build()?;
        let temporal_1 = nn::Conv1dBuilder::new(hidden_dim, hidden_dim * 2, 5)
            .padding(2)
            .build()?;

        // Self-attention
        let slf_attn_q = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_k = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_v = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_fc = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;

        let fc = nn::LinearBuilder::new(hidden_dim, out_dim)
            .bias(true)
            .build()?;

        Ok(Self {
            spectral_0,
            spectral_1,
            temporal_0,
            temporal_1,
            slf_attn_q,
            slf_attn_k,
            slf_attn_v,
            slf_attn_fc,
            fc,
            hidden_dim,
            out_dim,
        })
    }

    fn mish(x: &Array) -> Result<Array, Exception> {
        // mish(x) = x * tanh(softplus(x))
        let softplus = x.exp()?.add(array!(1.0f32))?.log()?;
        x.multiply(&tanh(&softplus)?)
    }

    fn glu(x: &Array) -> Result<Array, Exception> {
        // GLU: x * sigmoid(gate)
        let halves = split(x, 2, -1)?;
        halves[0].multiply(&nn::sigmoid(&halves[1])?)
    }

    /// Forward pass (expects NCL input mel, returns [batch, out_dim, 1] style)
    pub fn forward(&mut self, mel: &Array) -> Result<Array, Exception> {
        // mel: [batch, mel_channels, time] NCL -> [batch, time, mel_channels] NLC
        let x = swap_axes(mel, 1, 2)?;

        // Spectral processing (Linear operates on last dim, so NLC is correct)
        let x = self.spectral_0.forward(&x)?;
        let x = Self::mish(&x)?;
        let x = self.spectral_1.forward(&x)?;
        let x = Self::mish(&x)?;

        // Temporal processing with GLU and RESIDUAL connection
        // Python Conv1dGLU: residual = x; x = conv(x); x = glu(x); x = residual + x
        // Conv1d in mlx-rs expects NLC format
        let residual = x.clone();
        let x = self.temporal_0.forward(&x)?; // NLC -> NLC (but doubled channels)
        let x = Self::glu(&x)?; // Split on last dim and apply GLU
        let x = residual.add(&x)?; // RESIDUAL connection

        let residual = x.clone();
        let x = self.temporal_1.forward(&x)?;
        let x = Self::glu(&x)?;
        let x = residual.add(&x)?; // RESIDUAL connection

        // Self-attention with RESIDUAL connection
        // Python: residual = x; ... output = fc(output) + residual
        let residual = x.clone();

        // Multi-head attention: n_head=2, d_k=d_v=hidden_dim/2=64
        // Q, K, V: [batch, time, hidden] -> [batch, time, n_head, d_k]
        let n_head = 2;
        let d_k = self.hidden_dim / n_head;
        let batch = x.dim(0);
        let seq_len = x.dim(1);

        let q = self.slf_attn_q.forward(&x)?;
        let k = self.slf_attn_k.forward(&x)?;
        let v = self.slf_attn_v.forward(&x)?;

        // Reshape for multi-head: [batch, time, hidden] -> [batch, time, n_head, d_k] -> [batch*n_head, time, d_k]
        let q = q.reshape(&[batch, seq_len, n_head, d_k])?;
        let q = q.transpose_axes(&[2, 0, 1, 3])?; // [n_head, batch, time, d_k]
        let q = q.reshape(&[n_head * batch, seq_len, d_k])?;

        let k = k.reshape(&[batch, seq_len, n_head, d_k])?;
        let k = k.transpose_axes(&[2, 0, 1, 3])?;
        let k = k.reshape(&[n_head * batch, seq_len, d_k])?;

        let v = v.reshape(&[batch, seq_len, n_head, d_k])?;
        let v = v.transpose_axes(&[2, 0, 1, 3])?;
        let v = v.reshape(&[n_head * batch, seq_len, d_k])?;

        // Attention scores: [n_head*batch, time, time]
        let scale = (self.hidden_dim as f32).sqrt(); // d_model not d_k for temperature
        let scores = matmul(&q, &swap_axes(&k, 1, 2)?)?;
        let attn = softmax_axis(&scores.divide(array!(scale))?, -1, false)?;
        let attn_out = matmul(&attn, &v)?; // [n_head*batch, time, d_k]

        // Reshape back: [n_head*batch, time, d_k] -> [n_head, batch, time, d_k] -> [batch, time, n_head, d_k] -> [batch, time, hidden]
        let attn_out = attn_out.reshape(&[n_head, batch, seq_len, d_k])?;
        let attn_out = attn_out.transpose_axes(&[1, 2, 0, 3])?; // [batch, time, n_head, d_k]
        let attn_out = attn_out.reshape(&[batch, seq_len, self.hidden_dim])?;

        let x = self.slf_attn_fc.forward(&attn_out)?;
        let x = x.add(&residual)?; // RESIDUAL connection for attention

        // Temporal average pooling: [batch, time, hidden] -> [batch, hidden]
        let x = x.mean_axis(1, false)?;

        // Final projection: [batch, out_dim]
        let style = self.fc.forward(&x)?;

        // Add trailing dimension for broadcasting: [batch, out_dim, 1]
        Ok(style.index((.., .., mlx_rs::ops::indexing::NewAxis)))
    }
}

// ============================================================================
// SynthesizerTrn (full VITS model)
// ============================================================================

/// SynthesizerTrn: Full VITS model for GPT-SoVITS
#[derive(Debug, Clone, ModuleParameters)]
pub struct SynthesizerTrn {
    pub config: VITSConfig,
    #[param]
    pub quantizer: RVQCodebook,
    #[param]
    pub enc_p: TextEncoder,
    #[param]
    pub flow: ResidualCouplingBlock,
    #[param]
    pub dec: HiFiGANGenerator,
    #[param]
    pub ref_enc: MelStyleEncoder,
    #[param]
    pub ssl_proj: nn::Conv1d,
}

impl SynthesizerTrn {
    pub fn new(config: VITSConfig) -> Result<Self, Exception> {
        let quantizer = RVQCodebook::new(config.codebook_size, config.codebook_dim)?;

        let enc_p = TextEncoder::new(&config)?;

        let flow = ResidualCouplingBlock::new(
            config.hidden_channels,
            config.hidden_channels,
            5, // kernel_size
            4, // n_layers in WN
            config.n_flows,
            config.gin_channels,
        )?;

        let dec = HiFiGANGenerator::new(&config)?;

        let ref_enc = MelStyleEncoder::new(704, 128, config.gin_channels)?;

        // SSL projection before quantizer
        let ssl_proj = nn::Conv1dBuilder::new(config.ssl_dim, config.ssl_dim, 2)
            .padding(0)
            .build()?;

        Ok(Self {
            config,
            quantizer,
            enc_p,
            flow,
            dec,
            ref_enc,
            ssl_proj,
        })
    }

    /// Decode semantic codes to audio
    ///
    /// Args:
    /// - codes: Semantic codes [1, 1, seq] from T2S
    /// - text: Phoneme indices [batch, text_seq]
    /// - refer: Reference mel spectrogram [batch, mel_channels, time] (optional)
    /// - noise_scale: Noise scale for sampling (default 0.5)
    /// - speed: Speed factor (default 1.0)
    pub fn decode(
        &mut self,
        codes: &Array,
        text: &Array,
        refer: Option<&Array>,
        noise_scale: f32,
        _speed: f32,
    ) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;

        // Get style embedding from reference
        // For v2, slice to first 704 channels: refer[:, :704, :]
        let ge = if let Some(r) = refer {
            let r_sliced = r.index((.., ..704, ..));
            Some(self.ref_enc.forward(&r_sliced)?)
        } else {
            None
        };

        // Decode quantized features from codes
        let quantized = self.quantizer.decode(codes)?;

        // Interpolate if needed (25hz -> 50hz for semantic_frame_rate="25hz")
        // Input: [1, dim, seq] -> Output: [1, dim, seq*2]
        // Each position is repeated: [a0, a1, a2] -> [a0, a0, a1, a1, a2, a2]
        let seq_len = quantized.shape()[2] as i32;
        let target_len = seq_len * 2;
        // Add axis at end: [1, dim, seq] -> [1, dim, seq, 1]
        let q_expanded = quantized.index((.., .., .., mlx_rs::ops::indexing::NewAxis));
        // Repeat along the new axis: [1, dim, seq, 2]
        let q_rep = Array::repeat_axis::<f32>(q_expanded, 2, 3)?;
        // Reshape: [1, dim, seq*2]
        let quantized = q_rep.reshape(&[1, self.config.codebook_dim, target_len])?;

        // TextEncoder forward
        let (_, m_p, logs_p, y_mask) =
            self.enc_p.forward(&quantized, text, ge.as_ref())?;

        // Sample from posterior
        // Clamp logs_p to prevent numerical overflow in exp()
        let logs_p_clamped = maximum(&minimum(&logs_p, &array!(10.0f32))?, &array!(-10.0f32))?;
        let z_p = if noise_scale > 0.0 {
            let noise = random::normal::<f32>(m_p.shape(), None, None, None)?;
            m_p.add(&noise.multiply(&exp(&logs_p_clamped)?)?.multiply(array!(noise_scale))?)?
        } else {
            m_p.clone()
        };

        // Flow reverse
        let z = self.flow.forward(&z_p, &y_mask, ge.as_ref(), true)?;

        // Decode to audio (Python: o = vits.dec(z * y_mask, g=ge))
        let audio = self.dec.forward(&z.multiply(&y_mask)?, ge.as_ref())?;

        Ok(audio)
    }

    /// Extract latent codes from SSL features (for reference audio encoding)
    /// Input: ssl_features in NCL format [batch, ssl_dim, time]
    /// Output: projected features in NCL format [batch, ssl_dim, time']
    pub fn extract_latent(&mut self, ssl_features: &Array) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let ssl_nlc = swap_axes(ssl_features, 1, 2)?;
        let ssl = self.ssl_proj.forward(&ssl_nlc)?;
        // Convert back to NCL
        swap_axes(&ssl, 1, 2)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Compute weight from weight normalization components.
/// Weight normalization: weight = g * v / ||v||
/// g: [out_channels, 1, 1]
/// v: [out_channels, in_channels, kernel_size]
fn weight_norm_conv(g: &Array, v: &Array) -> Result<Array, Exception> {
    use mlx_rs::transforms::eval;

    // Compute L2 norm of v along in_channels and kernel dimensions
    // v shape: [out, in, kernel]
    let v_squared = v.square()?;
    // Sum along last two dims, keep dims for broadcasting
    let norm_sq = v_squared.sum_axes(&[-2, -1], true)?;
    let norm = sqrt(&norm_sq.add(array!(1e-12f32))?)?;

    // weight = g * v / norm
    let weight = g.multiply(v)?.divide(&norm)?;
    eval([&weight])?;
    Ok(weight)
}

/// Compute weight from weight normalization for ConvTranspose.
/// g: [in_channels, 1, 1]
/// v: [in_channels, out_channels, kernel_size]
fn weight_norm_convt(g: &Array, v: &Array) -> Result<Array, Exception> {
    use mlx_rs::transforms::eval;

    // Compute L2 norm of v along out_channels and kernel dimensions
    let v_squared = v.square()?;
    let norm_sq = v_squared.sum_axes(&[-2, -1], true)?;
    let norm = sqrt(&norm_sq.add(array!(1e-12f32))?)?;

    // weight = g * v / norm
    let weight = g.multiply(v)?.divide(&norm)?;
    eval([&weight])?;
    Ok(weight)
}

/// Load VITS/SynthesizerTrn weights from safetensors
pub fn load_vits_weights(
    model: &mut SynthesizerTrn,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    let get_weight = |key: &str| -> Option<Array> { weights.get(key).cloned() };

    // Helper to transpose Conv1d weights from PyTorch [out, in, kernel] to mlx-rs [out, kernel, in]
    let transpose_conv = |w: Array| -> Result<Array, Exception> { swap_axes(&w, 1, 2) };

    // Helper to transpose ConvTranspose1d weights from PyTorch [in, out, kernel] to mlx-rs [out, kernel, in]
    let transpose_convt = |w: Array| -> Result<Array, Exception> {
        let w = swap_axes(&w, 0, 1)?; // [out, in, kernel]
        swap_axes(&w, 1, 2) // [out, kernel, in]
    };

    // Helper to load weight-normalized Conv1d
    // Returns transposed weight ready for mlx-rs
    let load_weight_norm_conv = |prefix: &str| -> Option<Result<Array, Exception>> {
        let g = weights.get(&format!("{}.weight_g", prefix))?;
        let v = weights.get(&format!("{}.weight_v", prefix))?;
        Some(weight_norm_conv(g, v).and_then(|w| transpose_conv(w)))
    };

    // Helper to load weight-normalized ConvTranspose1d
    let load_weight_norm_convt = |prefix: &str| -> Option<Result<Array, Exception>> {
        let g = weights.get(&format!("{}.weight_g", prefix))?;
        let v = weights.get(&format!("{}.weight_v", prefix))?;
        Some(weight_norm_convt(g, v).and_then(|w| transpose_convt(w)))
    };

    // Quantizer codebook
    if let Some(w) = get_weight("quantizer.vq.layers.0._codebook.embed") {
        model.quantizer.embed = Param::new(w);
    }

    // SSL projection
    if let Some(w) = get_weight("ssl_proj.weight") {
        model.ssl_proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ssl_proj.bias") {
        model.ssl_proj.bias = Param::new(Some(b));
    }

    // TextEncoder (enc_p)
    if let Some(w) = get_weight("enc_p.ssl_proj.weight") {
        model.enc_p.ssl_proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.ssl_proj.bias") {
        model.enc_p.ssl_proj.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("enc_p.text_embedding.weight") {
        model.enc_p.text_embedding.weight = Param::new(w);
    }

    if let Some(w) = get_weight("enc_p.proj.weight") {
        model.enc_p.proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.proj.bias") {
        model.enc_p.proj.bias = Param::new(Some(b));
    }

    // MRTE
    if let Some(w) = get_weight("enc_p.mrte.c_pre.weight") {
        model.enc_p.mrte.c_pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.c_pre.bias") {
        model.enc_p.mrte.c_pre.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.c_post.weight") {
        model.enc_p.mrte.c_post.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.c_post.bias") {
        model.enc_p.mrte.c_post.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.text_pre.weight") {
        model.enc_p.mrte.text_pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.text_pre.bias") {
        model.enc_p.mrte.text_pre.bias = Param::new(Some(b));
    }

    // MRTE cross attention
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_q.weight") {
        model.enc_p.mrte.cross_attention.conv_q.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_q.bias") {
        model.enc_p.mrte.cross_attention.conv_q.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_k.weight") {
        model.enc_p.mrte.cross_attention.conv_k.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_k.bias") {
        model.enc_p.mrte.cross_attention.conv_k.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_v.weight") {
        model.enc_p.mrte.cross_attention.conv_v.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_v.bias") {
        model.enc_p.mrte.cross_attention.conv_v.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_o.weight") {
        model.enc_p.mrte.cross_attention.conv_o.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_o.bias") {
        model.enc_p.mrte.cross_attention.conv_o.bias = Param::new(Some(b));
    }

    // Helper to load transformer encoder weights
    let load_encoder_weights = |encoder: &mut TransformerEncoder,
                                prefix: &str,
                                weights: &HashMap<String, Array>|
     -> Result<(), Error> {
        for (i, layer) in encoder.layers.iter_mut().enumerate() {
            // Attention layers
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i)) {
                layer.attn.conv_q.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_q.bias", prefix, i)) {
                layer.attn.conv_q.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_k.weight", prefix, i)) {
                layer.attn.conv_k.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_k.bias", prefix, i)) {
                layer.attn.conv_k.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_v.weight", prefix, i)) {
                layer.attn.conv_v.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_v.bias", prefix, i)) {
                layer.attn.conv_v.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_o.weight", prefix, i)) {
                layer.attn.conv_o.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_o.bias", prefix, i)) {
                layer.attn.conv_o.bias = Param::new(Some(b.clone()));
            }

            // Relative position embeddings
            if let Some(emb) = weights.get(&format!("{}.attn_layers.{}.emb_rel_k", prefix, i)) {
                layer.attn.emb_rel_k = Param::new(emb.clone());
            }
            if let Some(emb) = weights.get(&format!("{}.attn_layers.{}.emb_rel_v", prefix, i)) {
                layer.attn.emb_rel_v = Param::new(emb.clone());
            }

            // FFN layers
            if let Some(w) = weights.get(&format!("{}.ffn_layers.{}.conv_1.weight", prefix, i)) {
                layer.ffn.conv_1.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.ffn_layers.{}.conv_1.bias", prefix, i)) {
                layer.ffn.conv_1.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.ffn_layers.{}.conv_2.weight", prefix, i)) {
                layer.ffn.conv_2.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.ffn_layers.{}.conv_2.bias", prefix, i)) {
                layer.ffn.conv_2.bias = Param::new(Some(b.clone()));
            }

            // Layer norms
            if let Some(g) = weights.get(&format!("{}.norm_layers_1.{}.gamma", prefix, i)) {
                layer.norm1.gamma = Param::new(g.clone());
            }
            if let Some(b) = weights.get(&format!("{}.norm_layers_1.{}.beta", prefix, i)) {
                layer.norm1.beta = Param::new(b.clone());
            }
            if let Some(g) = weights.get(&format!("{}.norm_layers_2.{}.gamma", prefix, i)) {
                layer.norm2.gamma = Param::new(g.clone());
            }
            if let Some(b) = weights.get(&format!("{}.norm_layers_2.{}.beta", prefix, i)) {
                layer.norm2.beta = Param::new(b.clone());
            }
        }
        Ok(())
    };

    // Load encoder_ssl weights
    load_encoder_weights(&mut model.enc_p.encoder_ssl, "enc_p.encoder_ssl", weights)?;

    // Load encoder_text weights
    load_encoder_weights(&mut model.enc_p.encoder_text, "enc_p.encoder_text", weights)?;

    // Load encoder2 weights
    load_encoder_weights(&mut model.enc_p.encoder2, "enc_p.encoder2", weights)?;

    // Flow layers
    for i in [0, 2, 4, 6].iter() {
        let flow_idx = *i / 2;
        if flow_idx < model.flow.flows.len() {
            let flow = &mut model.flow.flows[flow_idx];

            if let Some(w) = get_weight(&format!("flow.flows.{}.pre.weight", i)) {
                flow.pre.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("flow.flows.{}.pre.bias", i)) {
                flow.pre.bias = Param::new(Some(b));
            }
            if let Some(w) = get_weight(&format!("flow.flows.{}.post.weight", i)) {
                flow.post.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("flow.flows.{}.post.bias", i)) {
                flow.post.bias = Param::new(Some(b));
            }

            // WN encoder - try weight normalization first, fall back to regular
            let cond_prefix = format!("flow.flows.{}.enc.cond_layer", i);
            if let Some(w_result) = load_weight_norm_conv(&cond_prefix) {
                flow.enc.cond_layer.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", cond_prefix)) {
                flow.enc.cond_layer.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", cond_prefix)) {
                flow.enc.cond_layer.bias = Param::new(Some(b));
            }

            for j in 0..flow.enc.in_layers.len() {
                // in_layers - try weight normalization first, fall back to regular
                let in_prefix = format!("flow.flows.{}.enc.in_layers.{}", i, j);
                if let Some(w_result) = load_weight_norm_conv(&in_prefix) {
                    flow.enc.in_layers[j].weight = Param::new(w_result?);
                } else if let Some(w) = get_weight(&format!("{}.weight", in_prefix)) {
                    flow.enc.in_layers[j].weight = Param::new(transpose_conv(w)?);
                }
                if let Some(b) = get_weight(&format!("{}.bias", in_prefix)) {
                    flow.enc.in_layers[j].bias = Param::new(Some(b));
                }

                // res_skip_layers - try weight normalization first, fall back to regular
                let skip_prefix = format!("flow.flows.{}.enc.res_skip_layers.{}", i, j);
                if let Some(w_result) = load_weight_norm_conv(&skip_prefix) {
                    flow.enc.res_skip_layers[j].weight = Param::new(w_result?);
                } else if let Some(w) = get_weight(&format!("{}.weight", skip_prefix)) {
                    flow.enc.res_skip_layers[j].weight = Param::new(transpose_conv(w)?);
                }
                if let Some(b) = get_weight(&format!("{}.bias", skip_prefix)) {
                    flow.enc.res_skip_layers[j].bias = Param::new(Some(b));
                }
            }
        }
    }

    // HiFiGAN Generator (dec)
    if let Some(w) = get_weight("dec.conv_pre.weight") {
        model.dec.conv_pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("dec.conv_pre.bias") {
        model.dec.conv_pre.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("dec.conv_post.weight") {
        model.dec.conv_post.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(w) = get_weight("dec.cond.weight") {
        model.dec.cond.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("dec.cond.bias") {
        model.dec.cond.bias = Param::new(Some(b));
    }

    // Upsample layers (ConvTranspose1d) - try weight normalization first, fall back to regular
    for (i, up) in model.dec.ups.iter_mut().enumerate() {
        let prefix = format!("dec.ups.{}", i);
        if let Some(w_result) = load_weight_norm_convt(&prefix) {
            // Weight-normalized (luoxiang style)
            up.weight = Param::new(w_result?);
        } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
            // Regular weights (doubao style)
            up.weight = Param::new(transpose_convt(w)?);
        }
        if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
            up.bias = Param::new(Some(b));
        }
    }

    // ResBlocks - try weight normalization first, fall back to regular
    for (i, rb) in model.dec.resblocks.iter_mut().enumerate() {
        for (j, conv) in rb.convs1.iter_mut().enumerate() {
            let prefix = format!("dec.resblocks.{}.convs1.{}", i, j);
            if let Some(w_result) = load_weight_norm_conv(&prefix) {
                conv.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
                conv.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
                conv.bias = Param::new(Some(b));
            }
        }
        for (j, conv) in rb.convs2.iter_mut().enumerate() {
            let prefix = format!("dec.resblocks.{}.convs2.{}", i, j);
            if let Some(w_result) = load_weight_norm_conv(&prefix) {
                conv.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
                conv.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
                conv.bias = Param::new(Some(b));
            }
        }
    }

    // MelStyleEncoder (ref_enc)
    if let Some(w) = get_weight("ref_enc.spectral.0.fc.weight") {
        model.ref_enc.spectral_0.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.spectral.0.fc.bias") {
        model.ref_enc.spectral_0.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.spectral.3.fc.weight") {
        model.ref_enc.spectral_1.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.spectral.3.fc.bias") {
        model.ref_enc.spectral_1.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.temporal.0.conv1.conv.weight") {
        model.ref_enc.temporal_0.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ref_enc.temporal.0.conv1.conv.bias") {
        model.ref_enc.temporal_0.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.temporal.1.conv1.conv.weight") {
        model.ref_enc.temporal_1.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ref_enc.temporal.1.conv1.conv.bias") {
        model.ref_enc.temporal_1.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.slf_attn.w_qs.weight") {
        model.ref_enc.slf_attn_q.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_qs.bias") {
        model.ref_enc.slf_attn_q.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.w_ks.weight") {
        model.ref_enc.slf_attn_k.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_ks.bias") {
        model.ref_enc.slf_attn_k.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.w_vs.weight") {
        model.ref_enc.slf_attn_v.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_vs.bias") {
        model.ref_enc.slf_attn_v.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.fc.weight") {
        model.ref_enc.slf_attn_fc.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.fc.bias") {
        model.ref_enc.slf_attn_fc.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.fc.fc.weight") {
        model.ref_enc.fc.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.fc.fc.bias") {
        model.ref_enc.fc.bias = Param::new(Some(b));
    }

    Ok(())
}

/// Load VITS model from safetensors file
pub fn load_vits_model(weights_path: impl AsRef<Path>) -> Result<SynthesizerTrn, Error> {
    let path = weights_path.as_ref();

    let config = VITSConfig::default();
    let mut model = SynthesizerTrn::new(config)?;

    let weights = Array::load_safetensors(path)?;
    load_vits_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_rvq_codebook() {
        let codebook = RVQCodebook::new(1024, 768).unwrap();
        let codes = Array::zeros::<i32>(&[1, 1, 10]).unwrap();
        let quantized = codebook.decode(&codes).unwrap();
        eval([&quantized]).unwrap();
        assert_eq!(quantized.shape(), &[1, 768, 10]);
    }

    #[test]
    fn test_vits_config() {
        let config = VITSConfig::default();
        assert_eq!(config.hidden_channels, 192);
        assert_eq!(config.gin_channels, 512);
    }
}
