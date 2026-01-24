//! FunASR Paraformer Model for Chinese ASR
//!
//! This module implements the Paraformer-large (220M) model for non-autoregressive
//! Chinese speech recognition using MLX for GPU acceleration.
//!
//! # Architecture
//!
//! ```text
//! Audio (16kHz)
//!     ↓
//! [Mel Frontend] - 80 bins, 25ms window, 10ms hop, LFR 7/6
//!     ↓
//! [SAN-M Encoder] - 50 layers, 512 hidden, 4 heads
//!     ↓
//! [CIF Predictor] - Continuous Integrate-and-Fire
//!     ↓
//! [Bidirectional Decoder] - 16 layers, 512 hidden, 4 heads
//!     ↓
//! Tokens [batch, num_tokens]
//! ```
//!
//! # Key Features
//!
//! - **Non-autoregressive**: Predicts all tokens in parallel (3-5x faster than Whisper)
//! - **SAN-M Attention**: Self-attention with memory enhancement (FSMN block)
//! - **CIF Mechanism**: Continuous integrate-and-fire for length prediction
//! - **GPU Accelerated**: Metal GPU via MLX for all operations

use std::f32::consts::PI;
use std::path::Path;

use mlx_rs::{
    argmax_axis,
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp, softmax_axis},
    Array,
};

use crate::error::Error;
use std::fs;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Paraformer model
#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    // Audio frontend
    /// Sample rate (must be 16000)
    pub sample_rate: i32,
    /// Number of mel bins
    pub n_mels: i32,
    /// FFT window size in samples (400 = 25ms at 16kHz)
    pub n_fft: i32,
    /// Hop length in samples (160 = 10ms at 16kHz)
    pub hop_length: i32,
    /// LFR multiply factor (stack this many frames)
    pub lfr_m: i32,
    /// LFR divide factor (subsample by this factor)
    pub lfr_n: i32,

    // Encoder
    /// Encoder hidden dimension
    pub encoder_dim: i32,
    /// Number of encoder layers
    pub encoder_layers: i32,
    /// Number of attention heads
    pub encoder_heads: i32,
    /// FFN intermediate dimension
    pub encoder_ffn_dim: i32,
    /// SAN-M kernel size
    pub sanm_kernel_size: i32,
    /// Dropout rate
    pub dropout: f32,

    // CIF Predictor
    /// CIF threshold for firing
    pub cif_threshold: f32,
    /// CIF tail threshold
    pub cif_tail_threshold: f32,
    /// CIF conv left order
    pub cif_l_order: i32,
    /// CIF conv right order
    pub cif_r_order: i32,

    // Decoder
    /// Decoder hidden dimension (same as encoder)
    pub decoder_dim: i32,
    /// Number of decoder layers
    pub decoder_layers: i32,
    /// Number of decoder attention heads
    pub decoder_heads: i32,
    /// Decoder FFN intermediate dimension
    pub decoder_ffn_dim: i32,

    // Output
    /// Vocabulary size
    pub vocab_size: i32,
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            // Audio frontend (16kHz, 80 mel, LFR 7/6)
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,      // 25ms window
            hop_length: 160, // 10ms hop
            lfr_m: 7,        // Stack 7 frames
            lfr_n: 6,        // Subsample by 6

            // Encoder (Paraformer-large): 1 first_layer + 49 regular = 50 total
            encoder_dim: 512,
            encoder_layers: 50, // Total layers including first layer
            encoder_heads: 4,
            encoder_ffn_dim: 2048,
            sanm_kernel_size: 11,
            dropout: 0.1,

            // CIF Predictor
            cif_threshold: 1.0,
            cif_tail_threshold: 0.45,
            cif_l_order: 1,
            cif_r_order: 1,

            // Decoder (16 layers)
            decoder_dim: 512,
            decoder_layers: 16,
            decoder_heads: 4,
            decoder_ffn_dim: 2048,

            // Output
            vocab_size: 8404,
        }
    }
}

// ============================================================================
// Audio Frontend
// ============================================================================

/// Mel spectrogram frontend for Paraformer
///
/// Computes 80-bin mel spectrogram with LFR (Low Frame Rate) stacking
#[derive(Debug, Clone)]
pub struct MelFrontend {
    config: ParaformerConfig,
    /// Precomputed mel filterbank [n_mels, n_fft/2+1]
    mel_filters: Vec<f32>,
    /// Hann window
    window: Vec<f32>,
    /// CMVN addshift (negative mean) for LFR features [560]
    cmvn_addshift: Option<Vec<f32>>,
    /// CMVN rescale (inverse std) for LFR features [560]
    cmvn_rescale: Option<Vec<f32>>,
}

impl MelFrontend {
    pub fn new(config: &ParaformerConfig) -> Self {
        let n_fft = config.n_fft as usize;
        let n_mels = config.n_mels as usize;
        let sample_rate = config.sample_rate as f32;

        // Create Hamming window (as per FunASR config)
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let t = i as f32 / (n_fft - 1) as f32;
                0.54 - 0.46 * (2.0 * PI * t).cos()
            })
            .collect();

        // Create mel filterbank
        let mel_filters = Self::create_mel_filterbank(n_fft, n_mels, sample_rate);

        Self {
            config: config.clone(),
            mel_filters,
            window,
            cmvn_addshift: None,
            cmvn_rescale: None,
        }
    }

    /// Set CMVN normalization parameters (FunASR format)
    /// addshift: negative mean values (added to features)
    /// rescale: inverse std values (multiplied with features)
    pub fn set_cmvn(&mut self, addshift: Vec<f32>, rescale: Vec<f32>) {
        self.cmvn_addshift = Some(addshift);
        self.cmvn_rescale = Some(rescale);
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: f32) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let fmin = 0.0f32;
        let fmax = sample_rate / 2.0;

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Mel points
        let mut mel_points = Vec::with_capacity(n_mels + 2);
        for i in 0..=(n_mels + 1) {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_points.push(Self::mel_to_hz(mel));
        }

        // FFT frequencies
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        // Create filterbank [n_mels, n_freqs]
        let mut filterbank = vec![0.0f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = mel_points[m];
            let f_center = mel_points[m + 1];
            let f_right = mel_points[m + 2];

            for k in 0..n_freqs {
                let freq = fft_freqs[k];

                if freq >= f_left && freq <= f_center {
                    filterbank[m * n_freqs + k] = (freq - f_left) / (f_center - f_left);
                } else if freq > f_center && freq <= f_right {
                    filterbank[m * n_freqs + k] = (f_right - freq) / (f_right - f_center);
                }
            }
        }

        filterbank
    }

    /// Compute mel spectrogram from audio samples
    ///
    /// Returns Array [batch, time, n_mels * lfr_m] after LFR stacking
    ///
    /// FunASR pipeline (verified against kaldi_native_fbank):
    /// 1. Scale audio by 32768 (16-bit normalization)
    /// 2. Apply pre-emphasis with coeff=0.97
    /// 3. Hamming window, no center padding (snip_edges=True)
    /// 4. Power spectrum -> mel filterbank -> log
    /// 5. LFR stacking (7 frames, stride 6)
    /// 6. CMVN normalization
    pub fn forward(&self, audio: &Array) -> Result<Array, Exception> {
        let audio_data: Vec<f32> = audio
            .try_as_slice::<f32>()
            .map_err(|_| Exception::from("Failed to get audio slice"))?
            .to_vec();

        // Validate input
        if audio_data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(Exception::from("Audio contains NaN or Inf values"));
        }

        // Step 1: Scale audio by 2^15 (FunASR/Kaldi convention for 16-bit normalization)
        let audio_scaled: Vec<f32> = audio_data.iter().map(|&x| x * 32768.0).collect();

        // Step 2: Apply pre-emphasis (coeff=0.97, matching Kaldi default)
        let preemph_coeff = 0.97f32;
        let mut audio_preemph = Vec::with_capacity(audio_scaled.len());
        for i in 0..audio_scaled.len() {
            if i == 0 {
                audio_preemph.push(audio_scaled[i]);
            } else {
                audio_preemph.push(audio_scaled[i] - preemph_coeff * audio_scaled[i - 1]);
            }
        }

        // Compute STFT power spectrum (no center padding, snip_edges=True)
        let stft_mag = self.compute_stft(&audio_preemph);
        let n_freqs = (self.config.n_fft / 2 + 1) as usize;
        let n_frames = stft_mag.len() / n_freqs;

        if n_frames == 0 {
            return Err(Exception::from("Audio too short for mel spectrogram"));
        }

        // Apply mel filterbank
        let n_mels = self.config.n_mels as usize;
        let mut mel_spec = vec![0.0f32; n_frames * n_mels];

        for t in 0..n_frames {
            for m in 0..n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += stft_mag[t * n_freqs + k] * self.mel_filters[m * n_freqs + k];
                }
                // Log mel with floor
                mel_spec[t * n_mels + m] = (sum.max(1e-10)).ln();
            }
        }

        // Apply LFR (Low Frame Rate) stacking
        // FunASR: Prepend (lfr_m - 1) / 2 copies of first frame, then stack
        let lfr_m = self.config.lfr_m as usize;
        let lfr_n = self.config.lfr_n as usize;
        let left_padding = (lfr_m - 1) / 2;  // 3 for lfr_m=7
        let padded_frames = n_frames + left_padding;
        let lfr_frames = (padded_frames + lfr_n - 1) / lfr_n;
        let lfr_dim = n_mels * lfr_m;

        let mut lfr_spec = vec![0.0f32; lfr_frames * lfr_dim];

        for t in 0..lfr_frames {
            let start = t * lfr_n;
            for m in 0..lfr_m {
                // Calculate source frame index accounting for left padding
                let padded_idx = start + m;
                let src_frame = if padded_idx < left_padding {
                    0  // Left padding: repeat first frame
                } else if padded_idx - left_padding < n_frames {
                    padded_idx - left_padding
                } else {
                    n_frames - 1  // Right padding: repeat last frame
                };

                for f in 0..n_mels {
                    lfr_spec[t * lfr_dim + m * n_mels + f] = mel_spec[src_frame * n_mels + f];
                }
            }
        }

        // Apply CMVN after LFR stacking (FunASR format: x = (x + addshift) * rescale)
        if let (Some(addshift), Some(rescale)) = (&self.cmvn_addshift, &self.cmvn_rescale) {
            for t in 0..lfr_frames {
                for d in 0..lfr_dim {
                    let idx = t * lfr_dim + d;
                    lfr_spec[idx] = (lfr_spec[idx] + addshift[d]) * rescale[d];
                }
            }
        }

        // Create Array [1, lfr_frames, lfr_dim]
        let spec_array = Array::from_slice(&lfr_spec, &[1, lfr_frames as i32, lfr_dim as i32]);

        Ok(spec_array)
    }

    /// Compute STFT power spectrum
    ///
    /// Matches Kaldi's default: snip_edges=True (no center padding)
    /// Only frames that fit completely within the audio are computed.
    fn compute_stft(&self, samples: &[f32]) -> Vec<f32> {
        let n_fft = self.config.n_fft as usize;
        let hop_length = self.config.hop_length as usize;
        let n_freqs = n_fft / 2 + 1;

        // Kaldi snip_edges=True: no padding, only complete frames
        // Number of frames = floor((len - frame_length) / hop) + 1
        let n_frames = if samples.len() >= n_fft {
            (samples.len() - n_fft) / hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return vec![0.0f32; n_freqs];
        }

        // Output: [n_frames, n_freqs] - power spectrum
        let mut power_spec = vec![0.0f32; n_frames * n_freqs];

        for frame in 0..n_frames {
            let start = frame * hop_length;

            // Apply window (Hamming)
            let mut windowed = vec![0.0f32; n_fft];
            for i in 0..n_fft {
                windowed[i] = samples[start + i] * self.window[i];
            }

            // DFT power spectrum - O(n²), TODO: replace with FFT
            for k in 0..n_freqs {
                let mut real = 0.0f32;
                let mut imag = 0.0f32;

                for n in 0..n_fft {
                    let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                    real += windowed[n] * angle.cos();
                    imag -= windowed[n] * angle.sin();
                }

                // Power spectrum = |FFT|² (Kaldi uses power, not magnitude)
                power_spec[frame * n_freqs + k] = real * real + imag * imag;
            }
        }

        power_spec
    }
}

// ============================================================================
// Sinusoidal Positional Encoding
// ============================================================================

/// Create sinusoidal positional encoding
/// Sinusoidal position encoding matching FunASR's SinusoidalPositionEncoder
///
/// FunASR formula:
/// - positions: 1 to timesteps (1-indexed)
/// - log_timescale_increment = log(10000) / (depth/2 - 1)
/// - inv_timescales = exp(arange(depth/2) * (-log_timescale_increment))
/// - scaled_time = positions * inv_timescales
/// - encoding = concat([sin(scaled_time), cos(scaled_time)], dim=-1)
fn sinusoidal_position_encoding(max_len: i32, dim: i32) -> Result<Array, Exception> {
    let half_dim = dim / 2;
    let mut pe = vec![0.0f32; (max_len * dim) as usize];

    // log(10000) / (depth/2 - 1)
    let log_timescale_increment = 10000.0_f32.ln() / (half_dim as f32 - 1.0);

    // inv_timescales[i] = exp(-i * log_timescale_increment)
    let inv_timescales: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f32) * log_timescale_increment).exp())
        .collect();

    for pos in 0..max_len {
        // 1-indexed positions as in FunASR
        let position = (pos + 1) as f32;

        for i in 0..half_dim {
            let scaled_time = position * inv_timescales[i as usize];
            // First half: sin values, second half: cos values (concatenated)
            pe[(pos * dim + i) as usize] = scaled_time.sin();
            pe[(pos * dim + half_dim + i) as usize] = scaled_time.cos();
        }
    }

    Ok(Array::from_slice(&pe, &[max_len, dim]))
}

// ============================================================================
// SAN-M Attention (Self-Attention with Memory)
// ============================================================================

/// SAN-M Attention layer with FSMN memory block
/// Uses combined QKV projection as in FunASR
#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmAttention {
    #[param]
    pub linear_q_k_v: nn::Linear, // Combined QKV projection
    #[param]
    pub out_proj: nn::Linear,
    #[param]
    pub fsmn_block: nn::Conv1d,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub input_dim: i32, // May differ from output dim for first layer
}

impl SanmAttention {
    pub fn new(input_dim: i32, dim: i32, num_heads: i32, kernel_size: i32) -> Result<Self, Exception> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // Combined QKV projection: [input_dim] -> [3 * dim]
        let linear_q_k_v = nn::LinearBuilder::new(input_dim, 3 * dim).bias(true).build()?;
        let out_proj = nn::LinearBuilder::new(dim, dim).bias(true).build()?;

        // FSMN memory block (depthwise conv) - groups=dim for depthwise convolution
        let padding = kernel_size / 2;
        let fsmn_block = nn::Conv1dBuilder::new(dim, dim, kernel_size)
            .stride(1)
            .padding(padding)
            .groups(dim)  // Depthwise: each channel convolved independently
            .bias(false)  // FSMN has no bias
            .build()?;

        Ok(Self {
            linear_q_k_v,
            out_proj,
            fsmn_block,
            num_heads,
            head_dim,
            scale,
            input_dim,
        })
    }
}

impl Module<&Array> for SanmAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let (batch, seq_len, _dim) = (shape[0], shape[1], shape[2]);

        // Combined QKV projection
        let qkv = self.linear_q_k_v.forward(x)?;

        // Split into Q, K, V
        let dim = self.num_heads * self.head_dim;
        let q = qkv.index((.., .., ..dim));
        let k = qkv.index((.., .., dim..2*dim));
        let v = qkv.index((.., .., 2*dim..));

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back to [batch, seq, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // FSMN memory enhancement on the value vectors
        // FunASR: FSMN(v) = conv(v) + v (residual connection INSIDE FSMN)
        let v_proj = qkv.index((.., .., 2*dim..));  // [batch, seq, dim=512]
        let fsmn_conv = self.fsmn_block.forward(&v_proj)?;
        let fsmn_out = ops::add(&fsmn_conv, &v_proj)?;  // Residual connection within FSMN

        // FunASR: output = linear_out(attention) + FSMN
        // The output projection is applied ONLY to attention, not to FSMN
        let attn_proj = self.out_proj.forward(&attn_out)?;
        ops::add(&attn_proj, &fsmn_out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_q_k_v.training_mode(mode);
        self.out_proj.training_mode(mode);
        self.fsmn_block.training_mode(mode);
    }
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

/// Feed-forward network with GELU activation
#[derive(Debug, Clone, ModuleParameters)]
pub struct FeedForward {
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl FeedForward {
    pub fn new(dim: i32, ffn_dim: i32) -> Result<Self, Exception> {
        let up_proj = nn::LinearBuilder::new(dim, ffn_dim).bias(true).build()?;
        let down_proj = nn::LinearBuilder::new(ffn_dim, dim).bias(true).build()?;

        Ok(Self { up_proj, down_proj })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.up_proj.forward(x)?;
        let h = nn::relu(&h)?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

// ============================================================================
// SAN-M Encoder Layer
// ============================================================================

/// Single SAN-M encoder layer with pre-norm
#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmEncoderLayer {
    #[param]
    pub self_attn: SanmAttention,
    #[param]
    pub ffn: FeedForward,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
}

impl SanmEncoderLayer {
    pub fn new(input_dim: i32, dim: i32, config: &ParaformerConfig) -> Result<Self, Exception> {
        let self_attn = SanmAttention::new(
            input_dim,
            dim,
            config.encoder_heads,
            config.sanm_kernel_size,
        )?;
        let ffn = FeedForward::new(dim, config.encoder_ffn_dim)?;
        // norm1 normalizes input (may be different dim for first layer)
        let norm1 = nn::LayerNormBuilder::new(input_dim)
            .eps(1e-5)
            .build()?;
        let norm2 = nn::LayerNormBuilder::new(dim)
            .eps(1e-5)
            .build()?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
        })
    }
}

impl Module<&Array> for SanmEncoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Pre-norm self-attention
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h)?;

        // For first layer, input dim (560) != output dim (512), so no residual
        // For other layers, add residual
        let attn_input_dim = self.self_attn.input_dim;
        let attn_output_dim = self.self_attn.num_heads * self.self_attn.head_dim;

        let x = if attn_input_dim == attn_output_dim {
            ops::add(x, &h)?
        } else {
            // First layer: no residual (dimensions don't match)
            h
        };

        // Pre-norm FFN (always has matching dimensions)
        let h = self.norm2.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        ops::add(&x, &h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.ffn.training_mode(mode);
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
    }
}

// ============================================================================
// SAN-M Encoder
// ============================================================================

/// SAN-M Encoder stack
/// FunASR Paraformer-large has:
/// - encoders0: 1 special first layer (560 -> 512)
/// - layers: 49 regular layers (512 -> 512)
#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmEncoder {
    #[param]
    pub first_layer: SanmEncoderLayer, // encoders0.0: input LFR (560) -> 512
    #[param]
    pub layers: Vec<SanmEncoderLayer>, // layers.0-48: 512 -> 512
    #[param]
    pub after_norm: nn::LayerNorm,
    pub max_len: i32,
}

impl SanmEncoder {
    pub fn new(config: &ParaformerConfig) -> Result<Self, Exception> {
        // Input dimension: LFR stacked mel features (80 * 7 = 560)
        let input_dim = config.n_mels * config.lfr_m;

        // Special first layer: 560 -> 512
        let first_layer = SanmEncoderLayer::new(input_dim, config.encoder_dim, config)?;

        // Regular encoder layers (49 layers): 512 -> 512
        let num_regular_layers = config.encoder_layers - 1;
        let mut layers = Vec::with_capacity(num_regular_layers as usize);
        for _ in 0..num_regular_layers {
            layers.push(SanmEncoderLayer::new(config.encoder_dim, config.encoder_dim, config)?);
        }

        let after_norm = nn::LayerNormBuilder::new(config.encoder_dim)
            .eps(1e-5)
            .build()?;

        Ok(Self {
            first_layer,
            layers,
            after_norm,
            max_len: 5000,
        })
    }
}

impl Module<&Array> for SanmEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let seq_len = shape[1];
        let input_dim = shape[2];

        // FunASR: Scale input by sqrt(output_size) = sqrt(512) ≈ 22.627
        let scale_factor = (512.0_f32).sqrt();
        let mut h = x.multiply(array!(scale_factor))?;

        // Add positional encoding to scaled input (only ONCE, before encoders0)
        let pe = sinusoidal_position_encoding(seq_len, input_dim)?;
        let pe = pe.reshape(&[1, seq_len, input_dim])?;
        h = ops::add(&h, &pe)?;

        // First special layer (560 -> 512)
        h = self.first_layer.forward(&h)?;

        // Regular encoder layers (no additional PE - FunASR doesn't add PE after first layer)
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }

        // Final norm
        self.after_norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.first_layer.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.after_norm.training_mode(mode);
    }
}

// ============================================================================
// CIF Predictor
// ============================================================================

/// CIF (Continuous Integrate-and-Fire) Predictor
///
/// Predicts the number of output tokens and extracts acoustic embeddings
/// using a soft, monotonic alignment mechanism.
///
/// FunASR CifPredictorV3 implementation:
/// 1. Pad + Conv1d + ReLU (no residual!)
/// 2. Linear projection to scalar
/// 3. Sigmoid to get alphas
/// 4. CIF fire mechanism to extract acoustic embeddings
#[derive(Debug, Clone, ModuleParameters)]
pub struct CIFPredictor {
    #[param]
    pub conv: nn::Conv1d,
    #[param]
    pub output_proj: nn::Linear,
    pub threshold: f32,
    pub tail_threshold: f32,
    pub l_order: i32,
    pub r_order: i32,
}

impl CIFPredictor {
    pub fn new(config: &ParaformerConfig) -> Result<Self, Exception> {
        let kernel_size = config.cif_l_order + config.cif_r_order + 1;

        // FunASR uses asymmetric padding: l_order zeros on left, r_order zeros on right
        // MLX Conv1d padding is symmetric, so we use (l_order + r_order) / 2
        // This works correctly when l_order == r_order (both are 1 in Paraformer-large)
        // WARNING: If l_order != r_order, this will cause alignment issues!
        if config.cif_l_order != config.cif_r_order {
            return Err(Exception::from(
                "CIF asymmetric padding (l_order != r_order) not yet supported"
            ));
        }

        let conv = nn::Conv1dBuilder::new(config.encoder_dim, config.encoder_dim, kernel_size)
            .stride(1)
            .padding(config.cif_l_order)  // Symmetric padding (works when l_order == r_order)
            .build()?;

        let output_proj = nn::LinearBuilder::new(config.encoder_dim, 1)
            .bias(true)
            .build()?;

        Ok(Self {
            conv,
            output_proj,
            threshold: config.cif_threshold,
            tail_threshold: config.cif_tail_threshold,
            l_order: config.cif_l_order,
            r_order: config.cif_r_order,
        })
    }

    /// Compute alpha weights from encoder output
    /// FunASR: relu(conv(pad(x))) -> linear -> sigmoid
    fn compute_alphas(&mut self, encoder_out: &Array) -> Result<Array, Exception> {
        // FunASR does: transpose -> pad -> conv -> relu -> transpose -> linear -> sigmoid
        // MLX Conv1d expects NLC, and we use padding in conv builder
        let h = self.conv.forward(encoder_out)?;
        let h = nn::relu(&h)?;  // ReLU on conv output, NO residual!

        // Project to scalar and sigmoid
        let alphas = self.output_proj.forward(&h)?;
        let alphas = alphas.squeeze()?; // [batch, seq]
        ops::sigmoid(&alphas)
    }

    /// CIF fire mechanism - matching FunASR exactly
    /// FunASR accumulates weighted hidden states and fires when integrate >= threshold
    fn cif_fire(
        &self,
        hidden: &Array,
        alphas: &Array,
    ) -> Result<(Array, Array), Exception> {
        let shape = hidden.shape();
        let (batch, len_time, hidden_size) = (shape[0], shape[1], shape[2]);

        if batch != 1 {
            return Err(Exception::from("CIF currently only supports batch_size=1"));
        }

        let alphas_data: Vec<f32> = alphas
            .try_as_slice::<f32>()
            .map_err(|_| Exception::from("Failed to get alphas slice"))?
            .to_vec();
        let hidden_data: Vec<f32> = hidden
            .try_as_slice::<f32>()
            .map_err(|_| Exception::from("Failed to get hidden slice"))?
            .to_vec();

        // FunASR CIF algorithm
        let mut integrate = 0.0f32;
        let mut frame = vec![0.0f32; hidden_size as usize];
        let mut list_frames: Vec<Vec<f32>> = Vec::new();
        let mut fires: Vec<f32> = Vec::new();

        for t in 0..len_time as usize {
            let alpha = alphas_data[t];
            let distribution_completion = 1.0 - integrate;

            integrate += alpha;
            fires.push(integrate);

            let fire_place = integrate >= self.threshold;
            if fire_place {
                integrate -= 1.0;  // Subtract 1.0, not threshold!
            }

            let cur = if fire_place { distribution_completion } else { alpha };
            let remainds = alpha - cur;

            // Accumulate: frame += cur * hidden[t]
            for d in 0..hidden_size as usize {
                frame[d] += cur * hidden_data[t * hidden_size as usize + d];
            }

            // Save frame when fire
            if fire_place {
                list_frames.push(frame.clone());
                // Reset frame to remainds * hidden[t]
                for d in 0..hidden_size as usize {
                    frame[d] = remainds * hidden_data[t * hidden_size as usize + d];
                }
            }
        }

        // Handle tail: if remaining integrate > tail_threshold, add the accumulated frame
        if integrate > self.tail_threshold {
            list_frames.push(frame);
        }

        let num_tokens = list_frames.len();
        if num_tokens == 0 {
            return Ok((
                Array::zeros::<f32>(&[1, 0, hidden_size])?,
                Array::from_slice(&[0i32], &[1]),
            ));
        }

        // Flatten and create Array
        let flat_embeds: Vec<f32> = list_frames.into_iter().flatten().collect();
        let embeds_array = Array::from_slice(&flat_embeds, &[1, num_tokens as i32, hidden_size]);
        let token_num = Array::from_slice(&[num_tokens as i32], &[1]);

        Ok((embeds_array, token_num))
    }
}

impl Module<&Array> for CIFPredictor {
    type Output = (Array, Array); // (acoustic_embeds, token_num)
    type Error = Exception;

    fn forward(&mut self, encoder_out: &Array) -> Result<Self::Output, Self::Error> {
        let alphas = self.compute_alphas(encoder_out)?;
        self.cif_fire(encoder_out, &alphas)
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}

// ============================================================================
// Bidirectional Decoder Layer
// ============================================================================

/// Bidirectional decoder layer
/// Uses FSMN-only self-attention and cross-attention with combined KV
#[derive(Debug, Clone, ModuleParameters)]
pub struct ParaformerDecoderLayer {
    // Self-attention: FSMN block only (no Q/K/V projections)
    #[param]
    pub self_attn_fsmn: nn::Conv1d,
    // Cross-attention (src_attn): separate Q, combined KV
    #[param]
    pub src_attn_q: nn::Linear,
    #[param]
    pub src_attn_kv: nn::Linear, // Combined K+V projection
    #[param]
    pub src_attn_out: nn::Linear,
    #[param]
    pub ffn: FeedForward,
    #[param]
    pub ffn_norm: nn::LayerNorm, // feed_forward.norm (FFN gate)
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
    #[param]
    pub norm3: nn::LayerNorm,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl ParaformerDecoderLayer {
    pub fn new(config: &ParaformerConfig) -> Result<Self, Exception> {
        let head_dim = config.decoder_dim / config.decoder_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // Self-attention: FSMN block only (depthwise convolution)
        let padding = config.sanm_kernel_size / 2;
        let self_attn_fsmn = nn::Conv1dBuilder::new(config.decoder_dim, config.decoder_dim, config.sanm_kernel_size)
            .stride(1)
            .padding(padding)
            .groups(config.decoder_dim)  // Depthwise: each channel convolved independently
            .bias(false)  // FSMN has no bias
            .build()?;

        // Cross-attention with combined KV
        let src_attn_q = nn::LinearBuilder::new(config.decoder_dim, config.decoder_dim)
            .bias(true)
            .build()?;
        // Combined K+V: [encoder_dim] -> [2 * decoder_dim]
        let src_attn_kv = nn::LinearBuilder::new(config.encoder_dim, 2 * config.decoder_dim)
            .bias(true)
            .build()?;
        let src_attn_out = nn::LinearBuilder::new(config.decoder_dim, config.decoder_dim)
            .bias(true)
            .build()?;

        let ffn = FeedForward::new(config.decoder_dim, config.decoder_ffn_dim)?;

        // FFN gate normalization
        let ffn_norm = nn::LayerNormBuilder::new(config.decoder_ffn_dim)
            .eps(1e-5)
            .build()?;

        let norm1 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let norm2 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let norm3 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;

        Ok(Self {
            self_attn_fsmn,
            src_attn_q,
            src_attn_kv,
            src_attn_out,
            ffn,
            ffn_norm,
            norm1,
            norm2,
            norm3,
            num_heads: config.decoder_heads,
            head_dim,
            scale,
        })
    }

    fn cross_attention(
        &mut self,
        x: &Array,
        encoder_out: &Array,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, tgt_len, _) = (shape[0], shape[1], shape[2]);
        let src_len = encoder_out.shape()[1];

        // Project Q (separate) and KV (combined)
        let q = self.src_attn_q.forward(x)?;
        let kv = self.src_attn_kv.forward(encoder_out)?;

        // Split KV into K and V
        let dim = self.num_heads * self.head_dim;
        let k = kv.index((.., .., ..dim));
        let v = kv.index((.., .., dim..));

        // Reshape to multi-head
        let q = q
            .reshape(&[batch, tgt_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, src_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, src_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Attention
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])?;

        self.src_attn_out.forward(&attn_out)
    }
}

/// Input for decoder layer
pub struct DecoderLayerInput<'a> {
    pub x: &'a Array,
    pub encoder_out: &'a Array,
}

impl<'a> Module<DecoderLayerInput<'a>> for ParaformerDecoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DecoderLayerInput<'a>) -> Result<Self::Output, Self::Error> {
        let x = input.x;
        let encoder_out = input.encoder_out;
        let residual = x;

        // FunASR decoder layer order:
        // 1. norm1 -> FFN
        // 2. norm2 -> self_attn (FSMN)
        // 3. norm3 -> cross_attention

        // Step 1: FFN (PositionwiseFeedForwardDecoderSANM)
        // FunASR: w_2(norm(dropout(activation(w_1(x)))))
        let h = self.norm1.forward(x)?;
        let h = self.ffn.up_proj.forward(&h)?;  // w_1
        let h = nn::relu(&h)?;                   // activation
        // dropout skipped in eval mode
        let h = self.ffn_norm.forward(&h)?;      // norm (on intermediate dim!)
        let tgt = self.ffn.down_proj.forward(&h)?;  // w_2

        // Step 2: Self-attention (FSMN only)
        let h = self.norm2.forward(&tgt)?;
        // FunASR: FSMN with residual - conv(x) + x
        let h_fsmn = self.self_attn_fsmn.forward(&h)?;
        let h = ops::add(&h_fsmn, &h)?;  // FSMN residual
        let x = ops::add(residual, &h)?;  // Layer residual

        // Step 3: Cross-attention
        let residual = &x;
        let h = self.norm3.forward(&x)?;
        let h = self.cross_attention(&h, encoder_out)?;
        ops::add(residual, &h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn_fsmn.training_mode(mode);
        self.src_attn_q.training_mode(mode);
        self.src_attn_kv.training_mode(mode);
        self.src_attn_out.training_mode(mode);
        self.ffn.training_mode(mode);
        self.ffn_norm.training_mode(mode);
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
        self.norm3.training_mode(mode);
    }
}

// ============================================================================
// Bidirectional Decoder
// ============================================================================

/// Bidirectional decoder (16 layers + 1 final FFN layer for Paraformer-large)
#[derive(Debug, Clone, ModuleParameters)]
pub struct ParaformerDecoder {
    #[param]
    pub embed: nn::Embedding, // Token embedding
    #[param]
    pub layers: Vec<ParaformerDecoderLayer>,  // decoders (16 layers)
    // decoders3: final FFN-only layer
    #[param]
    pub final_ffn_norm1: nn::LayerNorm,
    #[param]
    pub final_ffn_up: nn::Linear,
    #[param]
    pub final_ffn_norm: nn::LayerNorm,  // norm on intermediate dim
    #[param]
    pub final_ffn_down: nn::Linear,
    #[param]
    pub after_norm: nn::LayerNorm,
    #[param]
    pub output_proj: nn::Linear,
}

impl ParaformerDecoder {
    pub fn new(config: &ParaformerConfig) -> Result<Self, Exception> {
        // Token embedding: vocab_size -> decoder_dim
        let embed = nn::Embedding::new(config.vocab_size, config.decoder_dim)?;

        let mut layers = Vec::with_capacity(config.decoder_layers as usize);
        for _ in 0..config.decoder_layers {
            layers.push(ParaformerDecoderLayer::new(config)?);
        }

        // decoders3: final FFN-only layer
        let final_ffn_norm1 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let final_ffn_up = nn::LinearBuilder::new(config.decoder_dim, config.decoder_ffn_dim)
            .bias(true)
            .build()?;
        let final_ffn_norm = nn::LayerNormBuilder::new(config.decoder_ffn_dim)
            .eps(1e-5)
            .build()?;
        let final_ffn_down = nn::LinearBuilder::new(config.decoder_ffn_dim, config.decoder_dim)
            .bias(false)  // no bias in w_2
            .build()?;

        let after_norm = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;

        let output_proj = nn::LinearBuilder::new(config.decoder_dim, config.vocab_size)
            .bias(true)
            .build()?;

        Ok(Self {
            embed,
            layers,
            final_ffn_norm1,
            final_ffn_up,
            final_ffn_norm,
            final_ffn_down,
            after_norm,
            output_proj,
        })
    }
}

/// Input for decoder
pub struct DecoderInput<'a> {
    pub acoustic_embeds: &'a Array,
    pub encoder_out: &'a Array,
}

impl<'a> Module<DecoderInput<'a>> for ParaformerDecoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DecoderInput<'a>) -> Result<Self::Output, Self::Error> {
        // Input is acoustic embeddings from CIF predictor (already in decoder_dim space)
        let mut h = input.acoustic_embeds.clone();

        // Decoder layers (decoders - 16 layers)
        for layer in &mut self.layers {
            h = layer.forward(DecoderLayerInput {
                x: &h,
                encoder_out: input.encoder_out,
            })?;
        }

        // Final FFN layer (decoders3)
        // FunASR: norm1 -> FFN (w_1 -> relu -> norm -> w_2), no residual
        let h = self.final_ffn_norm1.forward(&h)?;
        let h = self.final_ffn_up.forward(&h)?;
        let h = nn::relu(&h)?;
        let h = self.final_ffn_norm.forward(&h)?;
        let h = self.final_ffn_down.forward(&h)?;

        // Final norm and projection
        let h = self.after_norm.forward(&h)?;
        self.output_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.final_ffn_norm1.training_mode(mode);
        self.final_ffn_up.training_mode(mode);
        self.final_ffn_norm.training_mode(mode);
        self.final_ffn_down.training_mode(mode);
        self.after_norm.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}

// ============================================================================
// Full Paraformer Model
// ============================================================================

/// Paraformer ASR model
///
/// Non-autoregressive speech recognition model for Chinese
#[derive(Debug, Clone, ModuleParameters)]
pub struct Paraformer {
    pub frontend: MelFrontend,
    #[param]
    pub encoder: SanmEncoder,
    #[param]
    pub predictor: CIFPredictor,
    #[param]
    pub decoder: ParaformerDecoder,
    pub config: ParaformerConfig,
}

impl Paraformer {
    pub fn new(config: ParaformerConfig) -> Result<Self, Exception> {
        let frontend = MelFrontend::new(&config);
        let encoder = SanmEncoder::new(&config)?;
        let predictor = CIFPredictor::new(&config)?;
        let decoder = ParaformerDecoder::new(&config)?;

        Ok(Self {
            frontend,
            encoder,
            predictor,
            decoder,
            config,
        })
    }

    /// Transcribe audio to token IDs
    ///
    /// Input: audio Array [1, samples] (16kHz)
    /// Output: token IDs Array [1, num_tokens]
    pub fn forward(&mut self, audio: &Array) -> Result<Array, Exception> {
        // Frontend: audio -> mel features with LFR
        let mel = self.frontend.forward(audio)?;

        // Encoder
        let encoder_out = self.encoder.forward(&mel)?;

        // CIF predictor: extract acoustic embeddings
        let (acoustic_embeds, _token_num) = self.predictor.forward(&encoder_out)?;

        // Check if we have any tokens
        if acoustic_embeds.shape()[1] == 0 {
            return Ok(Array::from_slice::<i32>(&[], &[1, 0]));
        }

        // Decoder: predict token logits
        let logits = self.decoder.forward(DecoderInput {
            acoustic_embeds: &acoustic_embeds,
            encoder_out: &encoder_out,
        })?;

        // Argmax to get token IDs
        let token_ids = argmax_axis!(logits, -1)?;
        // Cast to int32 if needed (argmax may return uint32)
        token_ids.as_dtype(mlx_rs::Dtype::Int32)
    }

    /// Set CMVN normalization parameters (FunASR format)
    /// addshift: negative mean values (added to features)
    /// rescale: inverse std values (multiplied with features)
    pub fn set_cmvn(&mut self, addshift: Vec<f32>, rescale: Vec<f32>) {
        self.frontend.set_cmvn(addshift, rescale);
    }
}

impl Module<&Array> for Paraformer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, audio: &Array) -> Result<Self::Output, Self::Error> {
        Paraformer::forward(self, audio)
    }

    fn training_mode(&mut self, mode: bool) {
        self.encoder.training_mode(mode);
        self.predictor.training_mode(mode);
        self.decoder.training_mode(mode);
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

use std::collections::HashMap;

/// Helper to get a weight or return error
fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Message(format!("Missing weight: {}", key)))
}

/// Helper to get Conv1d weight and transpose from PyTorch [out, in, k] to MLX [out, k, in]
fn get_conv_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    let weight = get_weight(weights, key)?;
    // PyTorch Conv1d weights are stored as [out_channels, in_channels, kernel_size]
    // but mlx Conv1d expects [out_channels, kernel_size, in_channels]
    weight.transpose_axes(&[0, 2, 1])
        .map_err(|e| Error::Message(format!("Failed to transpose conv weight: {}", e)))
}

/// Load weights into Paraformer model
fn load_paraformer_weights(
    model: &mut Paraformer,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    eprintln!("Loading {} weight tensors...", weights.len());

    // ============ Encoder First Layer (encoders0.0) ============
    {
        let layer = &mut model.encoder.first_layer;
        let prefix = "encoder.encoders0.0";

        // SAN-M attention with combined QKV
        layer.self_attn.linear_q_k_v.weight = Param::new(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.weight", prefix))?);
        layer.self_attn.linear_q_k_v.bias = Param::new(Some(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.bias", prefix))?));
        layer.self_attn.out_proj.weight = Param::new(get_weight(weights, &format!("{}.self_attn.out_proj.weight", prefix))?);
        layer.self_attn.out_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.self_attn.out_proj.bias", prefix))?));
        layer.self_attn.fsmn_block.weight = Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        // FFN
        layer.ffn.up_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        layer.ffn.down_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.ffn.down_proj.bias", prefix))?));

        // Norms
        layer.norm1.weight = Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias = Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight = Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias = Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
    }

    // ============ Regular Encoder Layers (layers.0 to layers.48) ============
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        let prefix = format!("encoder.layers.{}", i);

        // SAN-M attention with combined QKV
        layer.self_attn.linear_q_k_v.weight = Param::new(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.weight", prefix))?);
        layer.self_attn.linear_q_k_v.bias = Param::new(Some(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.bias", prefix))?));
        layer.self_attn.out_proj.weight = Param::new(get_weight(weights, &format!("{}.self_attn.out_proj.weight", prefix))?);
        layer.self_attn.out_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.self_attn.out_proj.bias", prefix))?));
        layer.self_attn.fsmn_block.weight = Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        // FFN
        layer.ffn.up_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        layer.ffn.down_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.ffn.down_proj.bias", prefix))?));

        // Norms
        layer.norm1.weight = Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias = Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight = Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias = Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
    }

    // Encoder final norm
    model.encoder.after_norm.weight = Param::new(Some(get_weight(weights, "encoder.after_norm.weight")?));
    model.encoder.after_norm.bias = Param::new(Some(get_weight(weights, "encoder.after_norm.bias")?));

    // ============ CIF Predictor ============
    model.predictor.conv.weight = Param::new(get_conv_weight(weights, "predictor.conv.weight")?);
    model.predictor.conv.bias = Param::new(Some(get_weight(weights, "predictor.conv.bias")?));
    model.predictor.output_proj.weight = Param::new(get_weight(weights, "predictor.output_proj.weight")?);
    model.predictor.output_proj.bias = Param::new(Some(get_weight(weights, "predictor.output_proj.bias")?));

    // ============ Decoder Embedding ============
    model.decoder.embed.weight = Param::new(get_weight(weights, "decoder.embed.0.weight")?);

    // ============ Decoder Layers ============
    for (i, layer) in model.decoder.layers.iter_mut().enumerate() {
        let prefix = format!("decoder.layers.{}", i);

        // Self-attention (FSMN only) - transpose from [out, in, k] to [out, k, in]
        layer.self_attn_fsmn.weight = Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        // Cross-attention (src_attn)
        layer.src_attn_q.weight = Param::new(get_weight(weights, &format!("{}.src_attn.q_proj.weight", prefix))?);
        layer.src_attn_q.bias = Param::new(Some(get_weight(weights, &format!("{}.src_attn.q_proj.bias", prefix))?));
        layer.src_attn_kv.weight = Param::new(get_weight(weights, &format!("{}.src_attn.linear_k_v.weight", prefix))?);
        layer.src_attn_kv.bias = Param::new(Some(get_weight(weights, &format!("{}.src_attn.linear_k_v.bias", prefix))?));
        layer.src_attn_out.weight = Param::new(get_weight(weights, &format!("{}.src_attn.out_proj.weight", prefix))?);
        layer.src_attn_out.bias = Param::new(Some(get_weight(weights, &format!("{}.src_attn.out_proj.bias", prefix))?));

        // FFN
        layer.ffn.up_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias = Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight = Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        // Note: down_proj has no bias in decoder
        layer.ffn.down_proj.bias = Param::new(None);

        // FFN norm (gate)
        layer.ffn_norm.weight = Param::new(Some(get_weight(weights, &format!("{}.feed_forward.norm.weight", prefix))?));
        layer.ffn_norm.bias = Param::new(Some(get_weight(weights, &format!("{}.feed_forward.norm.bias", prefix))?));

        // Layer norms
        layer.norm1.weight = Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias = Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight = Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias = Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
        layer.norm3.weight = Param::new(Some(get_weight(weights, &format!("{}.norm3.weight", prefix))?));
        layer.norm3.bias = Param::new(Some(get_weight(weights, &format!("{}.norm3.bias", prefix))?));
    }

    // ============ Decoder Final FFN Layer (decoders3.0) ============
    // FunASR names: decoder.decoders3.0.{ffn.up_proj, ffn.down_proj, feed_forward.norm, norm1}
    model.decoder.final_ffn_norm1.weight = Param::new(Some(get_weight(weights, "decoder.decoders3.0.norm1.weight")?));
    model.decoder.final_ffn_norm1.bias = Param::new(Some(get_weight(weights, "decoder.decoders3.0.norm1.bias")?));
    model.decoder.final_ffn_up.weight = Param::new(get_weight(weights, "decoder.decoders3.0.ffn.up_proj.weight")?);
    model.decoder.final_ffn_up.bias = Param::new(Some(get_weight(weights, "decoder.decoders3.0.ffn.up_proj.bias")?));
    model.decoder.final_ffn_norm.weight = Param::new(Some(get_weight(weights, "decoder.decoders3.0.feed_forward.norm.weight")?));
    model.decoder.final_ffn_norm.bias = Param::new(Some(get_weight(weights, "decoder.decoders3.0.feed_forward.norm.bias")?));
    model.decoder.final_ffn_down.weight = Param::new(get_weight(weights, "decoder.decoders3.0.ffn.down_proj.weight")?);
    // Note: down_proj has no bias

    // Decoder final norm and output projection
    model.decoder.after_norm.weight = Param::new(Some(get_weight(weights, "decoder.after_norm.weight")?));
    model.decoder.after_norm.bias = Param::new(Some(get_weight(weights, "decoder.after_norm.bias")?));
    model.decoder.output_proj.weight = Param::new(get_weight(weights, "decoder.output_proj.weight")?);
    model.decoder.output_proj.bias = Param::new(Some(get_weight(weights, "decoder.output_proj.bias")?));

    eprintln!("Weights loaded successfully");
    Ok(())
}

/// Parse FunASR am.mvn file to extract CMVN parameters
/// Returns (addshift, rescale) vectors for 560-dim LFR features
///
/// Handles both single-line and multi-line formats:
/// - Single-line: `<LearnRateCoef> 0 [ -8.31 -8.60 ... ]`
/// - Multi-line: values may span multiple lines between `[` and `]`
pub fn parse_cmvn_file(path: impl AsRef<Path>) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let content = fs::read_to_string(path.as_ref())
        .map_err(|e| Error::Message(format!("Failed to read CMVN file: {}", e)))?;

    let mut addshift = Vec::new();
    let mut rescale = Vec::new();
    let mut in_addshift = false;
    let mut in_rescale = false;
    let mut in_values = false; // Currently parsing values between [ and ]

    for line in content.lines() {
        let line = line.trim();

        // Section markers
        if line.contains("<AddShift>") {
            in_addshift = true;
            in_rescale = false;
            in_values = false;
            continue;
        }
        if line.contains("<Rescale>") {
            in_addshift = false;
            in_rescale = true;
            in_values = false;
            continue;
        }
        if line.contains("</Nnet>") {
            break;
        }
        if line.contains("<Splice>") || line.contains("<Nnet>") {
            continue;
        }

        // Parse values - handle both single-line and multi-line formats
        if (in_addshift || in_rescale) && (line.contains('[') || in_values) {
            // Extract the relevant portion
            let mut parse_str = line;

            // Check for start of values block
            if let Some(start) = line.find('[') {
                in_values = true;
                parse_str = &line[start + 1..];
            }

            // Check for end of values block
            let at_end = parse_str.contains(']');
            if at_end {
                if let Some(end) = parse_str.find(']') {
                    parse_str = &parse_str[..end];
                }
                in_values = false;
            }

            // Parse float values from this line
            let values: Vec<f32> = parse_str
                .split_whitespace()
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();

            // Accumulate (not overwrite) values
            if in_addshift {
                addshift.extend(values);
            } else if in_rescale {
                rescale.extend(values);
            }
        }
    }

    if addshift.is_empty() || rescale.is_empty() {
        return Err(Error::Message(format!(
            "Failed to parse CMVN values from am.mvn (addshift={}, rescale={})",
            addshift.len(), rescale.len()
        )));
    }

    if addshift.len() != 560 || rescale.len() != 560 {
        return Err(Error::Message(format!(
            "CMVN dimension mismatch: expected 560, got addshift={}, rescale={}",
            addshift.len(), rescale.len()
        )));
    }

    Ok((addshift, rescale))
}

/// Load Paraformer model from safetensors weights
pub fn load_paraformer_model(weights_path: impl AsRef<Path>) -> Result<Paraformer, Error> {
    let path = weights_path.as_ref();
    let config = ParaformerConfig::default();
    let mut model = Paraformer::new(config).map_err(Error::Exception)?;

    // Load weights from safetensors
    let weights = Array::load_safetensors(path)
        .map_err(|e| Error::Message(format!("Failed to load weights: {:?}", e)))?;
    load_paraformer_weights(&mut model, &weights)?;

    Ok(model)
}

/// Load Paraformer model with custom config
pub fn load_paraformer_model_with_config(
    weights_path: impl AsRef<Path>,
    config: ParaformerConfig,
) -> Result<Paraformer, Error> {
    let path = weights_path.as_ref();
    let mut model = Paraformer::new(config).map_err(Error::Exception)?;

    // Load weights from safetensors
    let weights = Array::load_safetensors(path)
        .map_err(|e| Error::Message(format!("Failed to load weights: {:?}", e)))?;
    load_paraformer_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ParaformerConfig::default();
        assert_eq!(config.encoder_layers, 50);
        assert_eq!(config.decoder_layers, 16);
        assert_eq!(config.vocab_size, 8404);
    }

    #[test]
    fn test_sinusoidal_encoding() {
        let pe = sinusoidal_position_encoding(100, 512).unwrap();
        assert_eq!(pe.shape(), &[100, 512]);
    }

    #[test]
    fn test_model_creation() {
        let config = ParaformerConfig::default();
        let model = Paraformer::new(config);
        assert!(model.is_ok());
    }
}
