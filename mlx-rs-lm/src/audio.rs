//! Audio processing utilities for TTS
//!
//! Provides WAV file loading and mel spectrogram computation for reference audio.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use mlx_rs::{array, ops::{sqrt, swap_axes}, Array};
use mlx_rs::error::Exception;

/// Audio configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// FFT size (filter_length)
    pub n_fft: i32,
    /// Hop length between frames
    pub hop_length: i32,
    /// Window length
    pub win_length: i32,
    /// Sample rate
    pub sample_rate: i32,
    /// Number of mel channels
    pub n_mels: i32,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (None = sample_rate / 2)
    pub fmax: Option<f32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 640,
            win_length: 2048,
            sample_rate: 32000,
            n_mels: 704,  // v2 uses 704 mel bins
            fmin: 0.0,
            fmax: None,
        }
    }
}

/// Load WAV file and return samples as f32 in range [-1, 1]
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), std::io::Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read RIFF header
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    if &header != b"RIFF" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a RIFF file",
        ));
    }

    // Skip file size
    reader.seek(SeekFrom::Current(4))?;

    // Read WAVE header
    reader.read_exact(&mut header)?;
    if &header != b"WAVE" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a WAVE file",
        ));
    }

    let mut sample_rate = 0u32;
    let mut bits_per_sample = 16u16;
    let mut num_channels = 1u16;
    let mut audio_data: Vec<u8> = Vec::new();

    // Read chunks
    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            break;
        }

        let mut chunk_size_bytes = [0u8; 4];
        reader.read_exact(&mut chunk_size_bytes)?;
        let chunk_size = u32::from_le_bytes(chunk_size_bytes);

        match &chunk_id {
            b"fmt " => {
                let mut fmt_data = vec![0u8; chunk_size as usize];
                reader.read_exact(&mut fmt_data)?;

                // Audio format (should be 1 for PCM)
                let _audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
                num_channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
                sample_rate = u32::from_le_bytes([
                    fmt_data[4],
                    fmt_data[5],
                    fmt_data[6],
                    fmt_data[7],
                ]);
                // byte_rate = u32::from_le_bytes([fmt_data[8..12]])
                // block_align = u16::from_le_bytes([fmt_data[12], fmt_data[13]])
                bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
            }
            b"data" => {
                audio_data = vec![0u8; chunk_size as usize];
                reader.read_exact(&mut audio_data)?;
                break;
            }
            _ => {
                // Skip unknown chunk
                reader.seek(SeekFrom::Current(chunk_size as i64))?;
            }
        }
    }

    // Convert to f32 samples
    let samples: Vec<f32> = match bits_per_sample {
        16 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 2);
            for chunk in audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample as f32 / 32768.0);
            }
            samples
        }
        24 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 3);
            for chunk in audio_data.chunks_exact(3) {
                let sample = i32::from_le_bytes([0, chunk[0], chunk[1], chunk[2]]) >> 8;
                samples.push(sample as f32 / 8388608.0);
            }
            samples
        }
        32 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 4);
            for chunk in audio_data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                samples.push(sample);
            }
            samples
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported bits per sample: {}", bits_per_sample),
            ));
        }
    };

    // Mix to mono if stereo
    let samples = if num_channels > 1 {
        samples
            .chunks_exact(num_channels as usize)
            .map(|ch| ch.iter().sum::<f32>() / num_channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Resample audio from source rate to target rate using linear interpolation
pub fn resample(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / target_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(samples.len() - 1);
        let frac = (src_idx - idx_floor as f64) as f32;

        let sample = samples[idx_floor] * (1.0 - frac) + samples[idx_ceil] * frac;
        output.push(sample);
    }

    output
}

/// Create Hann window
fn hann_window(size: usize) -> Vec<f32> {
    let mut window = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f32 / (size - 1) as f32;
        window.push(0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos());
    }
    window
}

/// Convert frequency to mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix
fn mel_filterbank(n_fft: i32, n_mels: i32, sample_rate: i32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = (n_fft / 2 + 1) as usize;

    // Mel points
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels as usize + 2);
    for i in 0..=(n_mels + 1) as usize {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    // Convert to FFT bins
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    // Create filterbank
    let mut filterbank = vec![0.0f32; n_mels as usize * n_freqs];

    for m in 0..n_mels as usize {
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

/// Compute Short-Time Fourier Transform magnitude
/// Returns [n_freqs, n_frames] where n_freqs = n_fft/2 + 1
/// Uses center=False (no padding) to match GPT-SoVITS
fn stft_magnitude(
    samples: &[f32],
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
) -> Vec<f32> {
    use std::f32::consts::PI;

    let n_fft = n_fft as usize;
    let hop_length = hop_length as usize;
    let win_length = win_length as usize;
    let n_freqs = n_fft / 2 + 1;

    // Create window
    let window = hann_window(win_length);

    // No center padding (center=False like Python)
    // Number of frames with center=False
    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return vec![0.0f32; n_freqs];
    }

    // Output magnitude spectrogram [n_freqs, n_frames]
    let mut magnitude = vec![0.0f32; n_freqs * n_frames];

    for frame in 0..n_frames {
        let start = frame * hop_length;

        // Apply window directly to samples (no padding)
        // For n_fft == win_length, just apply window to the frame
        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..win_length.min(n_fft) {
            if start + i < samples.len() {
                windowed[i] = samples[start + i] * window[i];
            }
        }

        // DFT to compute magnitude
        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for n in 0..n_fft {
                let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                real += windowed[n] * angle.cos();
                imag -= windowed[n] * angle.sin();
            }

            magnitude[k * n_frames + frame] = (real * real + imag * imag).sqrt();
        }
    }

    magnitude
}

/// Compute STFT spectrogram from audio samples
///
/// For GPT-SoVITS v2, this returns the raw STFT magnitude (first n_mels frequency bins)
/// NOT mel-scale transformed.
///
/// Returns Array with shape [1, n_mels, n_frames] (NCL format)
pub fn compute_mel_spectrogram(
    samples: &[f32],
    config: &AudioConfig,
) -> Result<Array, Exception> {
    let n_freqs = (config.n_fft / 2 + 1) as usize;

    // Compute STFT magnitude [n_freqs, n_frames]
    let stft_mag = stft_magnitude(
        samples,
        config.n_fft,
        config.hop_length,
        config.win_length,
    );

    let n_frames = stft_mag.len() / n_freqs;

    // For v2, use raw STFT magnitude (first n_mels bins), NOT mel-scale
    // GPT-SoVITS v2 expects refer[:, :704] which is the first 704 frequency bins
    let n_bins = (config.n_mels as usize).min(n_freqs);
    let mut spec = vec![0.0f32; n_bins * n_frames];

    for f in 0..n_bins {
        for t in 0..n_frames {
            spec[f * n_frames + t] = stft_mag[f * n_frames + t];
        }
    }

    // Create Array [1, n_bins, n_frames]
    let spec_array = Array::from_slice(&spec, &[1, n_bins as i32, n_frames as i32]);

    Ok(spec_array)
}

/// Load audio for HuBERT (16kHz, normalized)
///
/// Returns Array with shape [1, samples] ready for HuBERT input
pub fn load_audio_for_hubert(
    path: impl AsRef<Path>,
) -> Result<Array, Box<dyn std::error::Error>> {
    // Load WAV
    let (samples, src_rate) = load_wav(&path)?;

    // Resample to 16kHz if needed
    let samples = if src_rate != 16000 {
        resample(&samples, src_rate, 16000)
    } else {
        samples
    };

    // Normalize audio
    let max_val = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let samples: Vec<f32> = if max_val > 0.0 {
        samples.iter().map(|x| x / max_val.max(1.0)).collect()
    } else {
        samples
    };

    // Create Array [1, samples]
    let audio_array = Array::from_slice(&samples, &[1, samples.len() as i32]);

    Ok(audio_array)
}

/// Load reference audio and compute mel spectrogram
///
/// Returns Array with shape [1, n_mels, n_frames] (NCL format)
pub fn load_reference_mel(
    path: impl AsRef<Path>,
    config: &AudioConfig,
) -> Result<Array, Box<dyn std::error::Error>> {
    // Load WAV
    let (samples, src_rate) = load_wav(&path)?;

    // Resample if needed
    let samples = if src_rate != config.sample_rate as u32 {
        resample(&samples, src_rate, config.sample_rate as u32)
    } else {
        samples
    };

    // Match Python normalization: if (maxx > 1): audio /= min(2, maxx)
    let max_val = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let samples: Vec<f32> = if max_val > 1.0 {
        let scale = max_val.min(2.0);
        samples.iter().map(|x| x / scale).collect()
    } else {
        samples
    };

    // Compute mel spectrogram
    let mel = compute_mel_spectrogram(&samples, config)?;

    Ok(mel)
}

// ============ FunASR/Paraformer Audio Frontend ============
//
// DEPRECATED: Use `crate::models::paraformer::MelFrontend` instead.
// This module contains a legacy frontend implementation that diverges from
// FunASR's actual preprocessing pipeline (missing 16-bit scaling, different
// LFR padding strategy). The authoritative implementation is in paraformer.rs.
//
// These functions are kept for backwards compatibility but will be removed
// in a future version.

/// DEPRECATED: Use `crate::models::paraformer::ParaformerConfig` instead
#[deprecated(note = "Use crate::models::paraformer::MelFrontend for FunASR-compatible features")]
#[derive(Debug, Clone)]
pub struct ParaformerAudioConfig {
    pub sample_rate: i32,
    pub window: String,
    pub n_mels: i32,
    pub frame_length_ms: i32,
    pub frame_shift_ms: i32,
    pub lfr_m: i32,
    pub lfr_n: i32,
    pub dither: f32,
}

#[allow(deprecated)]
impl Default for ParaformerAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            window: "hamming".to_string(),
            n_mels: 80,
            frame_length_ms: 25,
            frame_shift_ms: 10,
            lfr_m: 7,
            lfr_n: 6,
            dither: 0.0,
        }
    }
}

/// DEPRECATED: Use `crate::models::paraformer::parse_cmvn_file` instead
#[deprecated(note = "Use crate::models::paraformer::parse_cmvn_file for robust CMVN parsing")]
pub struct CmvnStats {
    pub mean: Vec<f32>,
    pub istd: Vec<f32>,
}

#[allow(deprecated)]
impl CmvnStats {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut mean = Vec::new();
        let mut var = Vec::new();
        let mut in_mean = false;
        let mut in_var = false;

        for line in content.lines() {
            let line = line.trim();
            if line.contains("<AddShift>") { in_mean = true; in_var = false; continue; }
            if line.contains("<Rescale>") { in_mean = false; in_var = true; continue; }
            if line.contains("</Nnet>") { break; }

            if (in_mean || in_var) && line.contains("[") {
                if let Some(start) = line.find('[') {
                    let after_bracket = &line[start + 1..];
                    let end = after_bracket.find(']').unwrap_or(after_bracket.len());
                    let values: Vec<f32> = after_bracket[..end]
                        .split_whitespace()
                        .filter_map(|s| s.parse::<f32>().ok())
                        .collect();
                    if in_mean { mean.extend(values); }
                    else if in_var { var.extend(values); }
                }
            }
        }

        let mean: Vec<f32> = mean.iter().map(|x| -x).collect();
        Ok(Self { mean, istd: var })
    }

    pub fn identity(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], istd: vec![1.0; dim] }
    }
}

/// DEPRECATED: Use `crate::models::paraformer::MelFrontend::forward` instead
///
/// This function does NOT match FunASR's preprocessing:
/// - Missing 16-bit audio scaling (x32768)
/// - Different LFR padding strategy
#[deprecated(note = "Use crate::models::paraformer::MelFrontend for FunASR-compatible features")]
#[allow(deprecated)]
pub fn extract_paraformer_features(
    _samples: &[f32],
    _config: &ParaformerAudioConfig,
    _cmvn: &CmvnStats,
) -> Result<Array, Exception> {
    Err(Exception::from(
        "extract_paraformer_features is deprecated. Use crate::models::paraformer::MelFrontend instead."
    ))
}

/// DEPRECATED: Use MelFrontend from paraformer module
#[deprecated(note = "Use crate::models::paraformer::MelFrontend for FunASR-compatible features")]
#[allow(deprecated)]
pub fn load_audio_for_paraformer(
    _path: impl AsRef<std::path::Path>,
    _cmvn: &CmvnStats,
) -> Result<Array, Box<dyn std::error::Error>> {
    Err("load_audio_for_paraformer is deprecated. Use crate::models::paraformer::MelFrontend instead.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(256);
        assert_eq!(window.len(), 256);
        assert!((window[0]).abs() < 1e-6); // Start at 0
        assert!((window[127] - 1.0).abs() < 0.01); // Peak near middle
    }

    #[test]
    fn test_hz_to_mel() {
        assert!((hz_to_mel(0.0)).abs() < 1e-6);
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Rough check
    }

    #[test]
    fn test_resample() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let resampled = resample(&samples, 16000, 32000);
        // Output should be roughly 2x length
        assert!(resampled.len() > samples.len());
    }
}
