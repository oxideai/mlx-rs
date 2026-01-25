//! Voice Cloning API for GPT-SoVITS
//!
//! Provides a high-level API for voice cloning with any reference audio.
//! Supports both zero-shot and few-shot voice cloning modes.
//!
//! # Modes
//!
//! - **Zero-shot**: Uses only reference audio mel spectrogram for voice style
//! - **Few-shot**: Uses reference audio + transcript for stronger conditioning via HuBERT
//!
//! # Zero-Shot Example
//!
//! ```ignore
//! use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};
//!
//! let config = VoiceClonerConfig::default();
//! let mut cloner = VoiceCloner::new(config)?;
//!
//! // Zero-shot: only reference audio
//! cloner.set_reference_audio("/path/to/reference.wav")?;
//!
//! let audio = cloner.synthesize("你好，世界！")?;
//! cloner.save_wav(&audio, "/tmp/output.wav")?;
//! ```
//!
//! # Few-Shot Example (Better Quality)
//!
//! ```ignore
//! use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};
//!
//! let config = VoiceClonerConfig::default();
//! let mut cloner = VoiceCloner::new(config)?;
//!
//! // Few-shot: reference audio + transcript
//! cloner.set_reference_audio_with_text(
//!     "/path/to/reference.wav",
//!     "这是参考音频的文本内容"
//! )?;
//!
//! let audio = cloner.synthesize("你好，世界！")?;
//! cloner.play_blocking(&audio)?;
//! ```
//!
//! # Command Line
//!
//! ```bash
//! # Zero-shot
//! cargo run --release --example voice_clone -- "你好" --ref voice.wav
//!
//! # Few-shot
//! cargo run --release --example voice_clone -- "你好" --ref voice.wav --ref-text "参考文本"
//!
//! # Interactive mode
//! cargo run --release --example voice_clone -- --interactive
//! ```
//!
//! For detailed documentation, see `docs/voice_clone.md`

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use mlx_rs::{Array, module::Module, ops::indexing::IndexOp, transforms::eval, random};

use crate::{
    audio::{AudioConfig, load_reference_mel, load_audio_for_hubert},
    cache::ConcatKeyValueCache,
    error::Error,
    inference::preprocess_text,
    models::{
        hubert::{HuBertEncoder, load_hubert_model},
        t2s::{T2SConfig, T2SInput, T2SModel, load_t2s_model},
        vits::{SynthesizerTrn, load_vits_model},
    },
    text::BertFeatureExtractor,
};

/// Configuration for voice cloner
#[derive(Debug, Clone)]
pub struct VoiceClonerConfig {
    /// Path to T2S model weights
    pub t2s_weights: String,
    /// Path to BERT model weights
    pub bert_weights: String,
    /// Path to BERT tokenizer
    pub bert_tokenizer: String,
    /// Path to VITS model weights
    pub vits_weights: String,
    /// Path to HuBERT model weights (for few-shot mode)
    pub hubert_weights: String,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Top-k sampling parameter
    pub top_k: i32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Repetition penalty (1.0 = no penalty, 1.35 = Python default)
    pub repetition_penalty: f32,
    /// Noise scale for VITS (0.0 = deterministic)
    pub noise_scale: f32,
    /// Speed factor (1.0 = normal)
    pub speed: f32,
}

impl Default for VoiceClonerConfig {
    fn default() -> Self {
        Self {
            // Use doubao-mixed fine-tuned models (converted from dora-primespeech)
            t2s_weights: "/tmp/gpt-sovits-mlx/doubao_mixed_gpt_new.safetensors".to_string(),
            bert_weights: "/tmp/gpt-sovits-mlx/bert.safetensors".to_string(),
            bert_tokenizer: "/tmp/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json".to_string(),
            vits_weights: "/tmp/gpt-sovits-mlx/doubao_mixed_sovits_new.safetensors".to_string(),
            hubert_weights: "/tmp/gpt-sovits-mlx/hubert.safetensors".to_string(),
            sample_rate: 32000,
            top_k: 15,  // Increased to include more candidates like Python
            temperature: 0.6,  // Lower temperature for more deterministic sampling
            repetition_penalty: 1.35,  // Match Python default
            noise_scale: 0.5,
            speed: 1.0,
        }
    }
}

/// Generated audio output
#[derive(Debug)]
pub struct AudioOutput {
    /// Raw audio samples (f32, range -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Number of semantic tokens generated
    pub num_tokens: usize,
}

impl AudioOutput {
    /// Get duration in seconds
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Convert to i16 samples for WAV output
    pub fn to_i16_samples(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Apply fade-in to reduce initial noise artifacts
    ///
    /// # Arguments
    /// * `fade_ms` - Fade-in duration in milliseconds (default: 50ms)
    pub fn apply_fade_in(&mut self, fade_ms: f32) {
        let fade_samples = ((fade_ms / 1000.0) * self.sample_rate as f32) as usize;
        let fade_samples = fade_samples.min(self.samples.len());

        for i in 0..fade_samples {
            let factor = i as f32 / fade_samples as f32;
            self.samples[i] *= factor;
        }
    }
}

/// Voice cloner for GPT-SoVITS
pub struct VoiceCloner {
    config: VoiceClonerConfig,
    t2s_config: T2SConfig,
    t2s: T2SModel,
    bert: BertFeatureExtractor,
    vits: SynthesizerTrn,
    hubert: Option<HuBertEncoder>,
    audio_config: AudioConfig,
    reference_mel: Option<Array>,
    reference_path: Option<String>,
    /// Prompt semantic codes for few-shot mode (extracted from reference audio)
    prompt_semantic: Option<Array>,
    /// Reference text for few-shot mode
    reference_text: Option<String>,
}

impl VoiceCloner {
    /// Create a new voice cloner with the given configuration
    pub fn new(config: VoiceClonerConfig) -> Result<Self, Error> {
        // Validate paths (HuBERT is optional for few-shot mode)
        for (name, path) in [
            ("T2S weights", &config.t2s_weights),
            ("BERT weights", &config.bert_weights),
            ("BERT tokenizer", &config.bert_tokenizer),
            ("VITS weights", &config.vits_weights),
        ] {
            if !Path::new(path).exists() {
                return Err(Error::Message(format!("{} not found: {}", name, path)));
            }
        }

        // Load models
        let bert = BertFeatureExtractor::new(&config.bert_tokenizer, &config.bert_weights, -3)?;
        let t2s_config = T2SConfig::default();
        let t2s = load_t2s_model(&config.t2s_weights)?;
        let vits = load_vits_model(&config.vits_weights)?;
        let audio_config = AudioConfig::default();

        // Try to load HuBERT (optional for few-shot mode)
        let hubert = if Path::new(&config.hubert_weights).exists() {
            match load_hubert_model(&config.hubert_weights) {
                Ok(h) => Some(h),
                Err(e) => {
                    eprintln!("Warning: Failed to load HuBERT model: {}. Few-shot mode will be unavailable.", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            t2s_config,
            t2s,
            bert,
            vits,
            hubert,
            audio_config,
            reference_mel: None,
            reference_path: None,
            prompt_semantic: None,
            reference_text: None,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, Error> {
        Self::new(VoiceClonerConfig::default())
    }

    /// Set reference audio for voice cloning (zero-shot mode)
    pub fn set_reference_audio(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", path)));
        }

        let mel = load_reference_mel(path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        self.reference_mel = Some(mel);
        self.reference_path = Some(path.to_string_lossy().to_string());
        // Clear few-shot data
        self.prompt_semantic = None;
        self.reference_text = None;

        Ok(())
    }

    /// Set reference audio with transcript for few-shot mode
    ///
    /// Few-shot mode extracts semantic tokens from the reference audio using HuBERT,
    /// which provides better voice cloning quality than zero-shot mode.
    ///
    /// # Arguments
    /// * `audio_path` - Path to reference audio file
    /// * `text` - Transcript of the reference audio
    pub fn set_reference_audio_with_text(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();
        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Extract prompt semantic codes if HuBERT is available
        let prompt_semantic = if let Some(ref mut hubert) = self.hubert {
            // Load audio at 16kHz for HuBERT
            let audio_16k = load_audio_for_hubert(audio_path)
                .map_err(|e| Error::Message(format!("Failed to load audio for HuBERT: {}", e)))?;
            eval([&audio_16k]).map_err(|e| Error::Message(e.to_string()))?;

            // Pad with 0.3s silence (matching Python's zero_wav padding)
            // This is important for matching the exact token count
            let audio_data: Vec<f32> = audio_16k.as_slice().to_vec();
            let pad_samples = (0.3 * 16000.0) as usize;
            let mut audio_padded = audio_data;
            audio_padded.extend(vec![0.0f32; pad_samples]);
            let audio_16k = Array::from_slice(&audio_padded, &[1, audio_padded.len() as i32]);

            // Extract HuBERT features: [batch, time, 768] (NLC format)
            // NOTE: The Rust HuBERT implementation may not produce the same features as
            // the Python CNHubert. If few-shot results are poor, try using pre-computed
            // prompt semantic codes from Python instead.
            let hubert_features = hubert.forward(&audio_16k)
                .map_err(|e| Error::Message(format!("HuBERT forward failed: {}", e)))?;
            eval([&hubert_features]).map_err(|e| Error::Message(e.to_string()))?;

            // ssl_proj expects NLC format, hubert_features is already NLC
            let projected_nlc = self.vits.ssl_proj.forward(&hubert_features)
                .map_err(|e| Error::Message(format!("ssl_proj forward failed: {}", e)))?;
            eval([&projected_nlc]).map_err(|e| Error::Message(e.to_string()))?;

            // Convert to NCL for quantizer.encode: [batch, 768, time]
            let projected_ncl = projected_nlc.transpose_axes(&[0, 2, 1])
                .map_err(|e| Error::Message(format!("Transpose failed: {}", e)))?;

            // Encode to semantic codes: [batch, 1, time]
            let codes = self.vits.quantizer.encode(&projected_ncl)
                .map_err(|e| Error::Message(format!("Quantizer encode failed: {}", e)))?;
            eval([&codes]).map_err(|e| Error::Message(e.to_string()))?;

            // Debug: print token count
            let token_count = codes.shape()[2];
            println!("DEBUG: Rust HuBERT extracted {} prompt_semantic tokens", token_count);

            Some(codes)
        } else {
            return Err(Error::Message(
                "Few-shot mode requires HuBERT model. Ensure hubert_weights path is valid.".to_string()
            ));
        };

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = prompt_semantic;
        self.reference_text = Some(text.to_string());

        Ok(())
    }

    /// Set reference audio with pre-computed prompt semantic codes
    ///
    /// Use this when the Rust HuBERT produces poor results. You can extract
    /// prompt semantic codes using Python and load them here.
    ///
    /// # Arguments
    /// * `audio_path` - Path to reference audio file (for mel spectrogram)
    /// * `text` - Transcript of the reference audio
    /// * `codes_path` - Path to binary file containing i32 codes (little-endian)
    ///
    /// # Example: Extract codes with Python
    /// ```python
    /// # See scripts/extract_prompt_semantic.py
    /// import torch
    /// from transformers import HubertModel, Wav2Vec2FeatureExtractor
    /// # ... extract codes and save as .bin file
    /// codes.numpy().astype(np.int32).tofile("prompt_semantic.bin")
    /// ```
    pub fn set_reference_with_precomputed_codes(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
        codes_path: impl AsRef<Path>,
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();
        let codes_path = codes_path.as_ref();

        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }
        if !codes_path.exists() {
            return Err(Error::Message(format!("Codes file not found: {:?}", codes_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Load pre-computed codes from file (supports both .npy and raw binary)
        let codes_data = std::fs::read(codes_path)
            .map_err(|e| Error::Message(format!("Failed to read codes file: {}", e)))?;

        // Check for NPY file format (magic bytes: \x93NUMPY)
        let codes: Vec<i32> = if codes_data.len() > 10 && &codes_data[..6] == b"\x93NUMPY" {
            // Parse NPY file: find header end (newline after dict)
            let mut header_end = 10;
            while header_end < codes_data.len() && codes_data[header_end] != b'\n' {
                header_end += 1;
            }
            header_end += 1; // Skip the newline

            // Extract data portion
            codes_data[header_end..]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            // Raw binary format
            codes_data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        if codes.is_empty() {
            return Err(Error::Message("Codes file is empty".to_string()));
        }

        // Create Array from codes: [1, 1, num_codes]
        let codes_array = Array::from_slice(&codes, &[1, 1, codes.len() as i32]);

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = Some(codes_array);
        self.reference_text = Some(text.to_string());

        Ok(())
    }

    /// Set reference using pre-extracted semantic codes (for debugging/testing)
    ///
    /// This allows using semantic codes extracted from Python for comparison.
    pub fn set_reference_with_semantic_codes(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
        semantic_codes: &[i32],
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();

        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Create Array from codes: [1, 1, num_codes]
        let codes_array = Array::from_slice(semantic_codes, &[1, 1, semantic_codes.len() as i32]);

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = Some(codes_array);
        self.reference_text = Some(text.to_string());

        eprintln!("DEBUG: Set reference with {} semantic codes from external source", semantic_codes.len());

        Ok(())
    }

    /// Check if few-shot mode is available
    pub fn few_shot_available(&self) -> bool {
        self.hubert.is_some()
    }

    /// Check if currently in few-shot mode
    pub fn is_few_shot_mode(&self) -> bool {
        self.prompt_semantic.is_some() && self.reference_text.is_some()
    }

    /// Get the current reference audio path
    pub fn reference_path(&self) -> Option<&str> {
        self.reference_path.as_deref()
    }

    /// Get the current reference text (for few-shot mode)
    pub fn reference_text(&self) -> Option<&str> {
        self.reference_text.as_deref()
    }

    /// Get the current prompt semantic codes (for debugging)
    pub fn get_prompt_semantic(&self) -> Option<Array> {
        self.prompt_semantic.clone()
    }

    /// Synthesize audio from external semantic tokens (for testing/debugging)
    ///
    /// This bypasses token generation and directly vocodes the provided tokens.
    /// Useful for comparing Rust VITS with Python's semantic tokens.
    pub fn synthesize_from_tokens(&mut self, text: &str, tokens: &[i32]) -> Result<AudioOutput, Error> {
        let ref_mel = self.reference_mel.clone()
            .ok_or_else(|| Error::Message("No reference audio set.".to_string()))?;

        // Preprocess text to get phoneme IDs
        let (phoneme_ids, _phonemes, _word2ph, _text_normalized) = preprocess_text(text);

        // Vocode using provided tokens
        let audio = self.vocode(tokens, &phoneme_ids, &ref_mel)?;
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: tokens.len(),
        })
    }

    /// Synthesize speech from text
    ///
    /// Text is automatically split at punctuation marks (like Python's cut5 method)
    /// and each segment is processed separately for better quality.
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, Error> {
        // Clone reference mel to avoid borrow issues
        let ref_mel = self.reference_mel.clone()
            .ok_or_else(|| Error::Message("No reference audio set. Call set_reference_audio() first.".to_string()))?;

        // Split text by language like Python's LangSegment
        // Keeps English phrases together, only splits at language boundaries
        let segments = split_text_by_language(text);
        eprintln!("DEBUG: Split text into {} segments:", segments.len());
        for (i, seg) in segments.iter().enumerate() {
            eprintln!("  Segment {}: {}", i+1, seg);
        }

        // Process each segment separately
        let mut all_samples = Vec::new();
        let mut total_tokens = 0;

        for segment in segments {
            if segment.trim().is_empty() {
                continue;
            }

            // Check if we're in few-shot mode
            let segment_output = if self.is_few_shot_mode() {
                self.synthesize_few_shot(&segment, &ref_mel)?
            } else {
                self.synthesize_zero_shot(&segment, &ref_mel)?
            };

            total_tokens += segment_output.num_tokens;
            all_samples.extend(segment_output.samples);
        }

        // Combine all samples into final output
        let duration = all_samples.len() as f32 / self.config.sample_rate as f32;
        let mut output = AudioOutput {
            samples: all_samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: total_tokens,
        };

        // Apply fade-in to reduce initial noise artifacts (30ms)
        output.apply_fade_in(30.0);

        Ok(output)
    }

    /// Zero-shot synthesis (no reference text, only reference audio for style)
    fn synthesize_zero_shot(&mut self, text: &str, ref_mel: &Array) -> Result<AudioOutput, Error> {
        // Strip leading/trailing punctuation - trailing commas cause audio alignment issues
        let text = text.trim_matches(|c: char| {
            matches!(c, ',' | '.' | '!' | '?' | '，' | '。' | '！' | '？' | '、' | '；' | '：' | ' ')
        });

        // 1. Text preprocessing (word2ph comes from preprocessor for correct handling of mixed text)
        let (phoneme_ids, phonemes, word2ph, text_normalized) = preprocess_text(text);
        let ids: Vec<i32> = phoneme_ids.as_slice().to_vec();
        eprintln!("RUST phonemes: {:?}", phonemes);
        eprintln!("RUST IDs: {:?}", ids);

        // 2. BERT encoding - use normalized text (quotes/parentheses removed)
        let text_chars = text_normalized.chars().count();
        let word2ph_for_bert = &word2ph[..text_chars.min(word2ph.len())];
        let bert_features = self.extract_bert_features(&text_normalized, word2ph_for_bert, phonemes.len())?;

        // 3. Generate semantic tokens
        // For zero-shot, all tokens are newly generated (no prompt)
        let (all_tokens, generated_count) = self.generate_semantic_tokens(&phoneme_ids, &bert_features, phonemes.len(), None)?;
        // Use last generated_count tokens (for zero-shot, this equals all_tokens since no prompt)
        let tokens = &all_tokens[all_tokens.len().saturating_sub(generated_count)..];

        // 4. VITS vocoding
        let audio = self.vocode(tokens, &phoneme_ids, ref_mel)?;

        // 5. Convert to output
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: tokens.len(),
        })
    }

    /// Few-shot synthesis (with reference text and prompt semantic codes)
    fn synthesize_few_shot(&mut self, text: &str, ref_mel: &Array) -> Result<AudioOutput, Error> {
        let ref_text = self.reference_text.clone()
            .ok_or_else(|| Error::Message("Reference text not set".to_string()))?;
        let prompt_semantic = self.prompt_semantic.clone()
            .ok_or_else(|| Error::Message("Prompt semantic not set".to_string()))?;

        // Strip leading/trailing punctuation - trailing commas cause audio alignment issues
        let text = text.trim_matches(|c: char| {
            matches!(c, ',' | '.' | '!' | '?' | '，' | '。' | '！' | '？' | '、' | '；' | '：' | ' ')
        });

        // 1. Preprocess reference text
        // Note: preprocess_text produces the same phoneme sequence as Python (no special markers)
        let (ref_phoneme_ids, ref_phonemes, ref_word2ph, ref_text_normalized) = preprocess_text(&ref_text);

        // Trim whitespace from normalized text for BERT alignment
        let ref_text_trimmed = ref_text_normalized.trim();
        let ref_text_chars = ref_text_trimmed.chars().count();
        let ref_word2ph_for_bert = &ref_word2ph[..ref_text_chars.min(ref_word2ph.len())];
        let ref_bert_features = self.extract_bert_features(ref_text_trimmed, ref_word2ph_for_bert, ref_phonemes.len())?;

        // 2. Preprocess target text - use normalized text for BERT
        let (target_phoneme_ids, target_phonemes, target_word2ph, target_text_normalized) = preprocess_text(text);
        let target_ids: Vec<i32> = target_phoneme_ids.as_slice().to_vec();
        eprintln!("DEBUG: target phonemes for '{}': {:?}", text.chars().take(20).collect::<String>(), &target_phonemes[..target_phonemes.len().min(40)]);
        eprintln!("DEBUG: target phoneme IDs: {:?}", &target_ids[..target_ids.len().min(40)]);

        // Trim whitespace from normalized text for BERT alignment
        let target_text_trimmed = target_text_normalized.trim();
        let target_text_chars = target_text_trimmed.chars().count();
        let target_word2ph_for_bert = &target_word2ph[..target_text_chars.min(target_word2ph.len())];
        let mut target_bert_features = self.extract_bert_features(target_text_trimmed, target_word2ph_for_bert, target_phonemes.len())?;

        // Zero out BERT features for punctuation phonemes (comma, period, etc.)
        // This prevents punctuation BERT features from acting as "boundary markers"
        // that cause the T2S model to skip beginning phonemes
        let punct_phonemes = [",", ".", "!", "?", "-", "…"];
        for (i, ph) in target_phonemes.iter().enumerate() {
            if punct_phonemes.contains(&ph.as_str()) {
                // Zero out the BERT feature at this position
                let zeros = Array::zeros::<f32>(&[1, 1, 1024])
                    .map_err(|e| Error::Message(e.to_string()))?;
                // Replace the slice at position i
                let i32_i = i as i32;
                let before = if i > 0 {
                    Some(target_bert_features.index((.., ..i32_i, ..)))
                } else {
                    None
                };
                let after = if i < target_phonemes.len() - 1 {
                    Some(target_bert_features.index((.., (i32_i + 1).., ..)))
                } else {
                    None
                };
                // Rebuild features with zeros at position i
                let parts: Vec<&Array> = [before.as_ref(), Some(&zeros), after.as_ref()]
                    .into_iter()
                    .flatten()
                    .collect();
                target_bert_features = mlx_rs::ops::concatenate_axis(&parts, 1)
                    .map_err(|e| Error::Message(format!("Failed to zero BERT at punct: {}", e)))?;
                eval([&target_bert_features]).map_err(|e| Error::Message(e.to_string()))?;
                eprintln!("DEBUG: Zeroed BERT feature at position {} for '{}'", i, ph);
            }
        }

        // 3. Combine: all_phones = ref_phones + target_phones (Python: prompt_data["phones"] + item["phones"])
        let combined_phoneme_ids = mlx_rs::ops::concatenate_axis(&[&ref_phoneme_ids, &target_phoneme_ids], 1)
            .map_err(|e| Error::Message(format!("Failed to concat phonemes: {}", e)))?;
        eval([&combined_phoneme_ids]).map_err(|e| Error::Message(e.to_string()))?;

        eprintln!("DEBUG: ref_phonemes ({} items): {:?}", ref_phonemes.len(), &ref_phonemes[..ref_phonemes.len().min(30)]);
        let combined_ids: Vec<i32> = combined_phoneme_ids.as_slice().to_vec();
        eprintln!("DEBUG: combined_phoneme_ids ({} items): last 10 = {:?}", combined_ids.len(), &combined_ids[combined_ids.len().saturating_sub(10)..]);

        // 4. Combine: all_bert = ref_bert + target_bert (Python: torch.cat([prompt_data["bert_features"], item["bert_features"]], 1))
        let combined_bert_features = mlx_rs::ops::concatenate_axis(&[&ref_bert_features, &target_bert_features], 1)
            .map_err(|e| Error::Message(format!("Failed to concat BERT features: {}", e)))?;
        eval([&combined_bert_features]).map_err(|e| Error::Message(e.to_string()))?;

        // 5. Generate semantic tokens
        // Use TARGET phoneme count for bounds - prompt_semantic covers ref portion,
        // we only generate new tokens for target text
        let (all_tokens, generated_count) = self.generate_semantic_tokens(
            &combined_phoneme_ids,
            &combined_bert_features,
            target_phonemes.len(),  // Bounds based on target only
            Some(&prompt_semantic),
        )?;

        // 6. Extract only newly generated tokens for VITS (like Python: item[-idx:])
        // Python uses item[-idx:] where idx is the exact count of newly generated tokens
        // This matches exactly - take the LAST generated_count tokens
        let prompt_len = prompt_semantic.shape()[2] as usize;
        let new_tokens = &all_tokens[all_tokens.len().saturating_sub(generated_count)..];
        eprintln!("DEBUG: Extracting last {} new tokens for VITS (prompt_len={}, all_tokens={})",
            generated_count, prompt_len, all_tokens.len());
        eprintln!("DEBUG: First 20 NEW tokens (for VITS): {:?}", &new_tokens[..new_tokens.len().min(20)]);
        eprintln!("DEBUG: target_phoneme_ids count: {}", target_phoneme_ids.shape()[1]);

        // 7. VITS vocoding with target phonemes only (matching Python)
        let audio = self.vocode(new_tokens, &target_phoneme_ids, ref_mel)?;

        // 8. Convert to output
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: new_tokens.len(),
        })
    }

    /// Extract BERT features with proper alignment
    ///
    /// For mixed Chinese/English text:
    /// - Uses zero features (Chinese BERT can't process English)
    /// - For pure Chinese text, extracts actual BERT features
    fn extract_bert_features(&mut self, text: &str, word2ph: &[i32], phoneme_count: usize) -> Result<Array, Error> {
        use crate::text::{is_chinese_char, detect_language, Language};

        let language = detect_language(text);

        // For mixed or English text, use zeros since Chinese BERT can't process English
        if matches!(language, Language::Mixed | Language::English) {
            let bert_features = Array::zeros::<f32>(&[1, phoneme_count as i32, 1024])
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&bert_features]).map_err(|e| Error::Message(e.to_string()))?;
            return Ok(bert_features);
        }

        // Pure Chinese: extract actual BERT features
        let bert_features_raw = self.bert.extract_features(text, word2ph)?;
        eval([&bert_features_raw]).map_err(|e| Error::Message(e.to_string()))?;

        let bert_seq_len = bert_features_raw.shape()[1] as i32;
        let phoneme_count = phoneme_count as i32;

        let bert_features = if bert_seq_len < phoneme_count {
            let pad_len = phoneme_count - bert_seq_len;
            let padding = Array::zeros::<f32>(&[1, pad_len, 1024])
                .map_err(|e| Error::Message(e.to_string()))?;
            mlx_rs::ops::concatenate_axis(&[&bert_features_raw, &padding], 1)
                .map_err(|e| Error::Message(e.to_string()))?
        } else if bert_seq_len > phoneme_count {
            bert_features_raw.index((.., ..phoneme_count, ..))
        } else {
            bert_features_raw
        };

        eval([&bert_features]).map_err(|e| Error::Message(e.to_string()))?;
        Ok(bert_features)
    }

    /// Generate semantic tokens from phonemes and BERT features
    ///
    /// # Arguments
    /// * `phoneme_ids` - Phoneme token IDs
    /// * `bert_features` - BERT features
    /// * `phoneme_count` - Number of phonemes (for generation bounds)
    /// * `prompt_semantic` - Optional prompt semantic codes for few-shot mode
    ///
    /// # Returns
    /// Tuple of (all_tokens, generated_count) like Python's (y, idx)
    /// - all_tokens: prompt + newly generated tokens
    /// - generated_count: number of NEW tokens (use `all_tokens[all_tokens.len()-generated_count..]`)
    fn generate_semantic_tokens(
        &mut self,
        phoneme_ids: &Array,
        bert_features: &Array,
        phoneme_count: usize,
        prompt_semantic: Option<&Array>,
    ) -> Result<(Vec<i32>, usize), Error> {
        // Set seed like Python (seed=233333)
        random::seed(233333).map_err(|e| Error::Message(e.to_string()))?;

        let batch_size = 1;
        let num_layers = self.t2s_config.num_layers as usize;

        // Debug: print prompt length
        if let Some(prompt) = prompt_semantic {
            eprintln!("DEBUG: prompt_semantic shape: {:?}", prompt.shape());
        } else {
            eprintln!("DEBUG: no prompt_semantic (zero-shot mode)");
        }
        let mut caches: Vec<Option<ConcatKeyValueCache>> = (0..num_layers).map(|_| None).collect();

        // Extract prompt tokens for repetition penalty (like Python's y = prompts)
        // Python applies repetition penalty to ALL previous tokens including prompt
        let prompt_tokens: Vec<i32> = if let Some(prompt) = prompt_semantic {
            let prompt_squeezed = prompt.squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&prompt_squeezed]).map_err(|e| Error::Message(e.to_string()))?;
            prompt_squeezed.as_slice().to_vec()
        } else {
            vec![]
        };

        // For few-shot mode, use prompt_semantic as initial semantic_ids
        // For zero-shot mode, start with zeros
        let mut semantic_ids = if let Some(prompt) = prompt_semantic {
            // prompt is [batch, 1, seq], we need [batch, seq]
            let prompt_squeezed = prompt.squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;
            // If it's 1D, add batch dimension
            if prompt_squeezed.ndim() == 1 {
                let seq_len = prompt_squeezed.shape()[0] as i32;
                prompt_squeezed.reshape(&[1, seq_len])
                    .map_err(|e| Error::Message(e.to_string()))?
            } else {
                prompt_squeezed
            }
        } else {
            Array::zeros::<i32>(&[batch_size, 1])
                .map_err(|e| Error::Message(e.to_string()))?
        };

        // Prefill
        let input = T2SInput {
            phoneme_ids,
            semantic_ids: &semantic_ids,
            bert_features,
            cache: &mut caches,
        };
        let logits = self.t2s.forward(input)
            .map_err(|e| Error::Message(e.to_string()))?;
        eval([&logits]).map_err(|e| Error::Message(e.to_string()))?;

        // First token - include prompt_tokens in repetition penalty (Python behavior)
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..)).squeeze()
            .map_err(|e| Error::Message(e.to_string()))?;
        let eos_token = 1024;

        // Debug: print top-k logits for first token
        {
            eval([&last_logits]).map_err(|e| Error::Message(e.to_string()))?;
            let logits_vec: Vec<f32> = last_logits.as_slice().to_vec();
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("DEBUG first_token: top10 = {:?}", &indexed[..10]);
            eprintln!("DEBUG first_token: logit[824] = {:.4}, logit[28] = {:.4}", logits_vec[824], logits_vec[28]);
        }

        // Python masks EOS during first 11 tokens (idx < 11), so we mask here (idx=0)
        let mut token_id = sample_top_k_with_penalty(
            &last_logits,
            &prompt_tokens,  // Include prompt for repetition penalty
            self.config.top_k,
            self.config.temperature,
            self.config.repetition_penalty,
            true,  // mask_eos for first token
        )?;
        semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        // all_tokens contains prompt + generated (Python behavior: returns prompts + new tokens to VITS)
        // This matches Python's infer_panel which returns pred_semantic that includes prompt semantic
        let prompt_len = prompt_tokens.len();
        let mut all_tokens: Vec<i32> = prompt_tokens.clone();
        all_tokens.push(token_id);
        // all_tokens_for_penalty includes the first token for repetition penalty
        // This prevents immediate duplicate tokens (e.g., [47, 47, ...])
        let mut all_tokens_for_penalty: Vec<i32> = all_tokens.clone();
        // Track number of newly generated tokens (excluding prompt)
        let mut generated_count: usize = 1;

        // Generation bounds - adjusted to match Python's token generation rate
        // Python generates ~2.0-2.5 tokens per phoneme on average
        let target_tokens = (phoneme_count as f32 * 2.5) as usize;
        let max_tokens = (phoneme_count * 3).max(100);
        // min_tokens should prevent very early EOS but allow natural stopping
        // Lower threshold (1.8) to trust model's natural EOS detection
        // Forcing continuation past natural EOS causes repetition/bleeding from ref text
        let min_tokens = (phoneme_count as f32 * 1.8) as usize;

        // Autoregressive generation
        for step in 1..max_tokens {
            let input = T2SInput {
                phoneme_ids,
                semantic_ids: &semantic_ids,
                bert_features,
                cache: &mut caches,
            };

            let logits = self.t2s.forward(input)
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&logits]).map_err(|e| Error::Message(e.to_string()))?;

            let seq_len = logits.shape()[1];
            let last_logits = logits.index((.., seq_len - 1, ..)).squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;


            // Sample with repetition penalty applied to ALL previous tokens (including prompt)
            // This matches Python's behavior: sample(logits, y, ...) where y includes prompts
            // Python masks EOS during first 11 tokens: if(idx<11): logits = logits[:, :-1]
            // step=1 corresponds to Python idx=1, so mask_eos when step < 11
            let mask_eos = step < 11;
            token_id = sample_top_k_with_penalty(
                &last_logits,
                &all_tokens_for_penalty,
                self.config.top_k,
                self.config.temperature,
                self.config.repetition_penalty,
                mask_eos,
            )?;

            // Compute argmax token for dual EOS detection (like Python)
            let argmax_token = {
                let logits_vec: Vec<f32> = last_logits.flatten(None, None)
                    .map_err(|e| Error::Message(e.to_string()))?
                    .as_slice()
                    .to_vec();

                // Debug: check logits shape and EOS value
                if step == 1 {
                    eprintln!("DEBUG: logits_vec len={}, EOS[1024]={:.4}",
                        logits_vec.len(),
                        logits_vec.get(1024).copied().unwrap_or(f32::NEG_INFINITY));
                }

                logits_vec.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as i32)
                    .unwrap_or(0)
            };

            // EOS detection: Python checks BOTH sampled AND argmax tokens
            // if (self.EOS in samples[:, 0]) or (self.EOS in tokens):
            let eos_detected = token_id == eos_token || argmax_token == eos_token;

            // Debug: check argmax value periodically
            if step >= 70 && step <= 75 {
                let logits_vec: Vec<f32> = last_logits.flatten(None, None)
                    .map_err(|e| Error::Message(e.to_string()))?
                    .as_slice()
                    .to_vec();
                let eos_logit = logits_vec.get(1024).copied().unwrap_or(f32::NEG_INFINITY);
                let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // Find top 5 logits
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
                eprintln!("DEBUG step {}: top5={:?}, eos_logit={:.4}",
                    step, top5, eos_logit);
            }

            if eos_detected {
                if generated_count >= min_tokens {
                    eprintln!("DEBUG: EOS detected at step {}, breaking with {} new tokens ({} total)",
                        step, generated_count, all_tokens.len());
                    break;
                } else {
                    eprintln!("DEBUG: EOS detected but ignored at step {} (generated={} < min_tokens={})",
                        step, generated_count, min_tokens);
                }
            }

            // Target overflow check - use generated_count (not including prompt)
            // Increased to 1.5 to ensure complete pronunciation of all phonemes
            if generated_count > (target_tokens as f32 * 1.5) as usize {
                eprintln!("DEBUG: Target overflow at step {}, breaking with {} new tokens ({} total)",
                    step, generated_count, all_tokens.len());
                break;
            }

            // EOS retry if too early (only if sampled token is EOS)
            if token_id == eos_token {
                // Retry with EOS masked to force a non-EOS token
                token_id = sample_top_k_with_penalty(
                    &last_logits,
                    &all_tokens_for_penalty,
                    self.config.top_k * 2,
                    self.config.temperature * 1.5,
                    self.config.repetition_penalty,
                    true,  // mask_eos to force non-EOS token
                )?;
            }

            all_tokens.push(token_id);
            all_tokens_for_penalty.push(token_id);
            generated_count += 1;

            // Repetition detection for longer patterns (check only generated portion)
            if generated_count > min_tokens && detect_repetition(&all_tokens[prompt_len..], 3, 8) {
                while generated_count > min_tokens && detect_repetition(&all_tokens[prompt_len..], 3, 5) {
                    all_tokens.pop();
                    generated_count -= 1;
                }
                break;
            }

            semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        }

        // Debug: print token stats
        eprintln!("DEBUG: phoneme_count={}, target_tokens={}, max_tokens={}, min_tokens={}",
            phoneme_count, target_tokens, max_tokens, min_tokens);
        eprintln!("DEBUG: Generated {} new tokens + {} prompt = {} total",
            generated_count, prompt_len, all_tokens.len());
        if !all_tokens.is_empty() {
            eprintln!("DEBUG: First 20 tokens: {:?}", &all_tokens[..20.min(all_tokens.len())]);
            eprintln!("DEBUG: Last 10 tokens: {:?}", &all_tokens[all_tokens.len().saturating_sub(10)..]);
            let unique: std::collections::HashSet<_> = all_tokens.iter().collect();
            eprintln!("DEBUG: Unique tokens: {}", unique.len());
        }

        // Return (all_tokens, generated_count) like Python's (y, idx)
        // This allows caller to extract exactly the last `generated_count` tokens
        Ok((all_tokens, generated_count))
    }

    /// Vocode semantic tokens to audio
    fn vocode(&mut self, tokens: &[i32], phoneme_ids: &Array, ref_mel: &Array) -> Result<Array, Error> {
        let codes = Array::from_slice(tokens, &[1, 1, tokens.len() as i32]);

        let text_ids = phoneme_ids.squeeze()
            .map_err(|e| Error::Message(e.to_string()))?;
        let text_for_vits = text_ids.index(mlx_rs::ops::indexing::NewAxis);

        let audio = self.vits.decode(&codes, &text_for_vits, Some(ref_mel), self.config.noise_scale, self.config.speed)
            .map_err(|e| Error::Message(e.to_string()))?;

        eval([&audio]).map_err(|e| Error::Message(e.to_string()))?;
        Ok(audio)
    }

    /// Save audio to WAV file
    pub fn save_wav(&self, audio: &AudioOutput, path: impl AsRef<Path>) -> Result<(), Error> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let path = path.as_ref();
        let samples = audio.to_i16_samples();

        let file = File::create(path)
            .map_err(|e| Error::Message(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        // RIFF header
        writer.write_all(b"RIFF").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&file_size.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(b"WAVE").map_err(|e| Error::Message(e.to_string()))?;

        // fmt chunk
        writer.write_all(b"fmt ").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&16u32.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&1u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&1u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&audio.sample_rate.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&(audio.sample_rate * 2).to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&2u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&16u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;

        // data chunk
        writer.write_all(b"data").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&data_size.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;

        for sample in samples {
            writer.write_all(&sample.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        }

        Ok(())
    }

    /// Play audio using system player (macOS: afplay)
    #[cfg(target_os = "macos")]
    pub fn play(&self, audio: &AudioOutput) -> Result<(), Error> {
        // Save to temp file
        let temp_path = "/tmp/voice_clone_playback.wav";
        self.save_wav(audio, temp_path)?;

        // Play with afplay (non-blocking)
        Command::new("afplay")
            .arg(temp_path)
            .spawn()
            .map_err(|e| Error::Message(format!("Failed to play audio: {}", e)))?;

        Ok(())
    }

    /// Play audio and wait for completion
    #[cfg(target_os = "macos")]
    pub fn play_blocking(&self, audio: &AudioOutput) -> Result<(), Error> {
        let temp_path = "/tmp/voice_clone_playback.wav";
        self.save_wav(audio, temp_path)?;

        Command::new("afplay")
            .arg(temp_path)
            .status()
            .map_err(|e| Error::Message(format!("Failed to play audio: {}", e)))?;

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn play(&self, _audio: &AudioOutput) -> Result<(), Error> {
        Err(Error::Message("Audio playback not implemented for this platform".to_string()))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn play_blocking(&self, _audio: &AudioOutput) -> Result<(), Error> {
        Err(Error::Message("Audio playback not implemented for this platform".to_string()))
    }
}

/// Compute word2ph (phonemes per character) for text
fn compute_word2ph(text: &str) -> Vec<i32> {
    let mut word2ph = Vec::new();
    for c in text.chars() {
        if c == '，' || c == '。' || c == '！' || c == '？' || c == '；' || c == '：'
            || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':'
        {
            word2ph.push(1);
        } else if c.is_whitespace() {
            word2ph.push(1);
        } else {
            word2ph.push(2); // Most Chinese chars have 2 phonemes (initial + final)
        }
    }
    word2ph
}

/// Sample from logits using top-k sampling with optional repetition penalty
///
/// When `mask_eos` is true, the EOS token (1024) is masked out from sampling.
/// Python does this during the first 11 tokens of generation to prevent early stopping.
fn sample_top_k_with_penalty(
    logits: &Array,
    previous_tokens: &[i32],
    top_k: i32,
    temperature: f32,
    repetition_penalty: f32,
    mask_eos: bool,
) -> Result<i32, Error> {
    // Apply repetition penalty to previously used tokens
    let mut logits_vec: Vec<f32> = logits.flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?
        .as_slice()
        .to_vec();

    // Mask EOS token during early generation (Python: if idx < 11: logits = logits[:, :-1])
    // This prevents early stopping and forces generation of at least 10 tokens (~0.4s audio)
    if mask_eos && logits_vec.len() > 1024 {
        logits_vec[1024] = f32::NEG_INFINITY;
    }

    if repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        use std::collections::HashSet;
        let used_tokens: HashSet<i32> = previous_tokens.iter().cloned().collect();

        // Apply standard repetition penalty to all used tokens
        for &token in &used_tokens {
            if token >= 0 && (token as usize) < logits_vec.len() {
                let score = logits_vec[token as usize];
                // Penalize: if score < 0, multiply by penalty; if score > 0, divide by penalty
                logits_vec[token as usize] = if score < 0.0 {
                    score * repetition_penalty
                } else {
                    score / repetition_penalty
                };
            }
        }

        // Extra penalty for immediate repetition (prevent [47, 47, ...])
        // Set very negative logit for the last token to force diversity
        if let Some(&last_token) = previous_tokens.last() {
            if last_token >= 0 && (last_token as usize) < logits_vec.len() {
                logits_vec[last_token as usize] = f32::NEG_INFINITY;
            }
        }
    }

    let penalized_logits = Array::from_slice(&logits_vec, &[logits_vec.len() as i32]);

    // Apply temperature
    let scaled = if temperature != 1.0 {
        penalized_logits.divide(mlx_rs::array!(temperature))
            .map_err(|e| Error::Message(e.to_string()))?
    } else {
        penalized_logits
    };
    eval([&scaled]).map_err(|e| Error::Message(e.to_string()))?;

    let flat_logits = scaled.flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&flat_logits]).map_err(|e| Error::Message(e.to_string()))?;

    let probs = mlx_rs::ops::softmax_axis(&flat_logits, -1, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&probs]).map_err(|e| Error::Message(e.to_string()))?;

    let prob_vec: Vec<f32> = probs.as_slice().to_vec();

    let mut indexed: Vec<(usize, f32)> = prob_vec.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(top_k as usize).collect();

    let total: f32 = top_k_items.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = top_k_items.iter().map(|(_, p)| p / total).collect();

    let rand_arr = random::uniform::<f32, f32>(0.0, 1.0, &[], None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&rand_arr]).map_err(|e| Error::Message(e.to_string()))?;
    let r: f32 = rand_arr.item();

    let mut cumsum = 0.0f32;
    for (i, p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(top_k_items[i].0 as i32);
        }
    }

    Ok(top_k_items[0].0 as i32)
}

/// Sample from logits using top-k sampling (no repetition penalty)
fn sample_top_k(logits: &Array, top_k: i32, temperature: f32) -> Result<i32, Error> {
    let scaled = if temperature != 1.0 {
        logits.divide(mlx_rs::array!(temperature))
            .map_err(|e| Error::Message(e.to_string()))?
    } else {
        logits.clone()
    };
    eval([&scaled]).map_err(|e| Error::Message(e.to_string()))?;

    let flat_logits = scaled.flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&flat_logits]).map_err(|e| Error::Message(e.to_string()))?;

    let probs = mlx_rs::ops::softmax_axis(&flat_logits, -1, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&probs]).map_err(|e| Error::Message(e.to_string()))?;

    let prob_vec: Vec<f32> = probs.as_slice().to_vec();

    let mut indexed: Vec<(usize, f32)> = prob_vec.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(top_k as usize).collect();

    let total: f32 = top_k_items.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = top_k_items.iter().map(|(_, p)| p / total).collect();

    let rand_arr = random::uniform::<f32, f32>(0.0, 1.0, &[], None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&rand_arr]).map_err(|e| Error::Message(e.to_string()))?;
    let r: f32 = rand_arr.item();

    let mut cumsum = 0.0f32;
    for (i, p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(top_k_items[i].0 as i32);
        }
    }

    Ok(top_k_items[0].0 as i32)
}

/// Detect n-gram repetition
fn detect_repetition(tokens: &[i32], n: usize, min_count: usize) -> bool {
    if tokens.len() < n * 2 {
        return false;
    }
    let last_n: Vec<i32> = tokens[tokens.len() - n..].to_vec();
    tokens.windows(n).filter(|w| *w == last_n.as_slice()).count() >= min_count
}

/// Global language detector (lazy initialized)
/// Uses lingua for ML-based language detection like Python's LangSegment
static LANG_DETECTOR: OnceLock<LanguageDetector> = OnceLock::new();

fn get_lang_detector() -> &'static LanguageDetector {
    LANG_DETECTOR.get_or_init(|| {
        LanguageDetectorBuilder::from_languages(&[
            Language::Chinese,
            Language::English,
            Language::Japanese,
            Language::Korean,
        ])
        .with_preloaded_language_models()
        .build()
    })
}

/// Check if a character is CJK (Chinese, Japanese, or Korean)
fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
        '\u{3040}'..='\u{309F}' |  // Hiragana
        '\u{30A0}'..='\u{30FF}' |  // Katakana
        '\u{AC00}'..='\u{D7AF}' |  // Korean Hangul
        '\u{1100}'..='\u{11FF}'    // Korean Jamo
    )
}

/// Language-aware text segmentation (like Python's LangSegment)
///
/// Uses a hybrid approach:
/// 1. First splits by character class (ASCII letters = English, CJK = Chinese/Japanese/Korean)
/// 2. Uses lingua ML model for ambiguous cases
///
/// This matches Python's LangSegment which uses regex for obvious patterns
/// and py3langid for edge cases.
fn split_text_by_language(text: &str) -> Vec<String> {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Lang { English, Cjk, Other }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // Helper: check if a character is a digit or decimal point
    let is_digit_or_dot = |c: char| c.is_ascii_digit() || c == '.';

    // Step 1: Split by character class (like Python's regex patterns)
    // Special handling: numbers followed directly by CJK go with CJK (e.g., "126.4亿斤")
    let mut char_segments: Vec<(Lang, String)> = Vec::new();
    let mut current = String::new();
    let mut current_lang = Lang::Other;

    let mut i = 0;
    while i < len {
        let ch = chars[i];
        let char_lang = if ch.is_ascii_alphabetic() {
            Lang::English
        } else if is_cjk_char(ch) {
            Lang::Cjk
        } else {
            Lang::Other  // punctuation, numbers, spaces, quotes
        };

        match char_lang {
            Lang::English => {
                if current_lang == Lang::Cjk && !current.is_empty() {
                    char_segments.push((Lang::Cjk, std::mem::take(&mut current)));
                }
                current.push(ch);
                current_lang = Lang::English;
            }
            Lang::Cjk => {
                if current_lang == Lang::English && !current.is_empty() {
                    char_segments.push((Lang::English, std::mem::take(&mut current)));
                }
                current.push(ch);
                current_lang = Lang::Cjk;
            }
            Lang::Other => {
                // Check if this is a number followed directly by CJK
                if is_digit_or_dot(ch) {
                    // Look ahead to see what follows the number
                    let mut j = i;
                    while j < len && is_digit_or_dot(chars[j]) {
                        j += 1;
                    }
                    // If number is followed directly by CJK (no space), treat as CJK
                    if j < len && is_cjk_char(chars[j]) {
                        // Push current English segment if any
                        if current_lang == Lang::English && !current.is_empty() {
                            char_segments.push((Lang::English, std::mem::take(&mut current)));
                        }
                        // Add all the digits to CJK segment
                        while i < j {
                            current.push(chars[i]);
                            i += 1;
                        }
                        current_lang = Lang::Cjk;
                        continue;  // Don't increment i again
                    }
                }
                // Otherwise, punctuation/numbers/quotes attach to current segment
                current.push(ch);
            }
        }
        i += 1;
    }
    if !current.is_empty() {
        char_segments.push((current_lang, current));
    }

    // Step 2: For CJK segments, use lingua to detect if it's Japanese/Korean vs Chinese
    // (This matters for phoneme processing)
    let detector = get_lang_detector();
    let mut lang_segments: Vec<(Language, String)> = Vec::new();

    for (lang, text) in char_segments {
        if text.trim().is_empty() {
            continue;
        }
        match lang {
            Lang::English => {
                lang_segments.push((Language::English, text));
            }
            Lang::Cjk | Lang::Other => {
                // Use lingua to detect Chinese vs Japanese vs Korean
                if let Some(detected) = detector.detect_language_of(&text) {
                    lang_segments.push((detected, text));
                } else {
                    // Default to Chinese
                    lang_segments.push((Language::Chinese, text));
                }
            }
        }
    }

    // Step 3: Split CJK segments at sentence-ending punctuation
    let cjk_sentence_end: std::collections::HashSet<char> =
        ['。', '？', '！', '．'].into_iter().collect();

    let mut result = Vec::new();
    for (lang, text) in lang_segments {
        if matches!(lang, Language::Chinese | Language::Japanese | Language::Korean) {
            // Split CJK at sentence-ending punctuation
            let mut sub = String::new();
            for ch in text.chars() {
                sub.push(ch);
                if cjk_sentence_end.contains(&ch) {
                    if !sub.trim().is_empty() {
                        result.push(sub.clone());
                    }
                    sub.clear();
                }
            }
            if !sub.trim().is_empty() {
                result.push(sub);
            }
        } else {
            // Keep English segments whole
            if !text.trim().is_empty() {
                result.push(text);
            }
        }
    }

    // Step 4: Filter out segments that are too short (only punctuation, less than 2 actual characters)
    let result: Vec<String> = result.into_iter().filter(|s| {
        let content_chars = s.chars().filter(|c| {
            !matches!(*c, ',' | '.' | ';' | '?' | '!' | '、' | '，' | '。' | '？' | '！' | '；' | '：' | '…' | '"' | '"' | '\'' | '（' | '）' | '(' | ')' | '《' | '》' | '【' | '】')
        }).count();
        content_chars >= 2
    }).collect();

    // If filtering removed all segments, return original text as single segment
    if result.is_empty() {
        return vec![text.to_string()];
    }

    result
}

/// Split text at punctuation marks (like Python's cut5 method)
///
/// This splits text at: , . ; ? ! 、，。？！；：…
/// Numbers with decimal points (e.g., "3.14") are kept together.
#[allow(dead_code)]
fn split_text_at_punctuation(text: &str) -> Vec<String> {
    let punctuation: std::collections::HashSet<char> = [
        ',', '.', ';', '?', '!',  // English
        '、', '，', '。', '？', '！', '；', '：', '…',  // Chinese
    ].into_iter().collect();

    let chars: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        if punctuation.contains(&ch) {
            // Check if it's a decimal point (digit.digit)
            if ch == '.' && i > 0 && i < chars.len() - 1 {
                if chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit() {
                    current.push(ch);
                    continue;
                }
            }
            // Add punctuation to current segment
            current.push(ch);
            // Save segment if it has content
            let trimmed = current.trim();
            if !trimmed.is_empty() && !trimmed.chars().all(|c| punctuation.contains(&c)) {
                segments.push(current.clone());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }

    // Add remaining text
    if !current.trim().is_empty() {
        segments.push(current);
    }

    // If no segments were created, return original text
    if segments.is_empty() {
        vec![text.to_string()]
    } else {
        segments
    }
}

/// Convert audio array to f32 samples
fn array_to_f32_samples(audio: &Array) -> Result<Vec<f32>, Error> {
    eval([audio]).map_err(|e| Error::Message(e.to_string()))?;

    let flat = audio.flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&flat]).map_err(|e| Error::Message(e.to_string()))?;

    Ok(flat.as_slice().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_word2ph() {
        let word2ph = compute_word2ph("你好，世界！");
        assert_eq!(word2ph, vec![2, 2, 1, 2, 2, 1]); // 你(2) 好(2) ，(1) 世(2) 界(2) ！(1)
    }

    #[test]
    fn test_detect_repetition() {
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        assert!(detect_repetition(&tokens, 3, 3));
        assert!(!detect_repetition(&tokens, 3, 4));
    }
}
