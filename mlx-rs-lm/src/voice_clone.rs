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
    /// Noise scale for VITS (0.0 = deterministic)
    pub noise_scale: f32,
    /// Speed factor (1.0 = normal)
    pub speed: f32,
}

impl Default for VoiceClonerConfig {
    fn default() -> Self {
        Self {
            t2s_weights: "/tmp/gpt-sovits-mlx/doubao_gpt.safetensors".to_string(),
            bert_weights: "/tmp/gpt-sovits-mlx/bert.safetensors".to_string(),
            bert_tokenizer: "/tmp/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json".to_string(),
            vits_weights: "/tmp/gpt-sovits-mlx/doubao_sovits.safetensors".to_string(),
            hubert_weights: "/tmp/gpt-sovits-mlx/hubert.safetensors".to_string(),
            sample_rate: 32000,
            top_k: 5,
            temperature: 0.8,
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

        // Load pre-computed codes from binary file
        let codes_data = std::fs::read(codes_path)
            .map_err(|e| Error::Message(format!("Failed to read codes file: {}", e)))?;
        let codes: Vec<i32> = codes_data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

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

    /// Synthesize speech from text
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, Error> {
        // Clone reference mel to avoid borrow issues
        let ref_mel = self.reference_mel.clone()
            .ok_or_else(|| Error::Message("No reference audio set. Call set_reference_audio() first.".to_string()))?;

        // Check if we're in few-shot mode
        let mut output = if self.is_few_shot_mode() {
            self.synthesize_few_shot(text, &ref_mel)?
        } else {
            self.synthesize_zero_shot(text, &ref_mel)?
        };

        // Apply fade-in to reduce initial noise artifacts (30ms)
        output.apply_fade_in(30.0);

        Ok(output)
    }

    /// Zero-shot synthesis (no reference text, only reference audio for style)
    fn synthesize_zero_shot(&mut self, text: &str, ref_mel: &Array) -> Result<AudioOutput, Error> {
        // 1. Text preprocessing (word2ph comes from preprocessor for correct handling of mixed text)
        let (phoneme_ids, phonemes, word2ph) = preprocess_text(text);

        // 2. BERT encoding
        // word2ph includes trailing "!" but text doesn't, so slice it for BERT
        let text_chars = text.chars().count();
        let word2ph_for_bert = &word2ph[..text_chars.min(word2ph.len())];
        let bert_features = self.extract_bert_features(text, word2ph_for_bert, phonemes.len())?;

        // 3. Generate semantic tokens
        let tokens = self.generate_semantic_tokens(&phoneme_ids, &bert_features, phonemes.len(), None)?;

        // 4. VITS vocoding
        let audio = self.vocode(&tokens, &phoneme_ids, ref_mel)?;

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


        // 1. Preprocess reference text
        let (ref_phoneme_ids_raw, ref_phonemes_raw, ref_word2ph) = preprocess_text(&ref_text);

        // Strip trailing "!" from REF - Python: ref has NO marker, target HAS marker
        // Combined should have marker only at END (from target)
        let ref_phoneme_count = ref_phonemes_raw.len() - 1;  // Exclude trailing "!"
        let ref_phoneme_ids = ref_phoneme_ids_raw.index((.., ..ref_phoneme_count as i32));
        let ref_phonemes: Vec<String> = ref_phonemes_raw[..ref_phoneme_count].to_vec();

        let ref_text_chars = ref_text.chars().count();
        let ref_word2ph_for_bert = &ref_word2ph[..ref_text_chars.min(ref_word2ph.len())];
        let ref_bert_features = self.extract_bert_features(&ref_text, ref_word2ph_for_bert, ref_phonemes.len())?;

        // 2. Preprocess target text - KEEP the "!" marker
        let (target_phoneme_ids, target_phonemes, target_word2ph) = preprocess_text(text);

        let target_text_chars = text.chars().count();
        let target_word2ph_for_bert = &target_word2ph[..target_text_chars.min(target_word2ph.len())];
        let target_bert_features = self.extract_bert_features(text, target_word2ph_for_bert, target_phonemes.len())?;

        // 3. Combine: all_phones = ref_phones + target_phones (Python: prompt_data["phones"] + item["phones"])
        let combined_phoneme_ids = mlx_rs::ops::concatenate_axis(&[&ref_phoneme_ids, &target_phoneme_ids], 1)
            .map_err(|e| Error::Message(format!("Failed to concat phonemes: {}", e)))?;
        eval([&combined_phoneme_ids]).map_err(|e| Error::Message(e.to_string()))?;

        // 4. Combine: all_bert = ref_bert + target_bert (Python: torch.cat([prompt_data["bert_features"], item["bert_features"]], 1))
        let combined_bert_features = mlx_rs::ops::concatenate_axis(&[&ref_bert_features, &target_bert_features], 1)
            .map_err(|e| Error::Message(format!("Failed to concat BERT features: {}", e)))?;
        eval([&combined_bert_features]).map_err(|e| Error::Message(e.to_string()))?;

        // 5. Generate semantic tokens
        // Use TARGET phoneme count for bounds - prompt_semantic covers ref portion,
        // we only generate new tokens for target text
        let tokens = self.generate_semantic_tokens(
            &combined_phoneme_ids,
            &combined_bert_features,
            target_phonemes.len(),  // Bounds based on target only
            Some(&prompt_semantic),
        )?;

        // 6. VITS vocoding with target phonemes only
        let audio = self.vocode(&tokens, &target_phoneme_ids, ref_mel)?;

        // 7. Convert to output
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: tokens.len(),
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
    fn generate_semantic_tokens(
        &mut self,
        phoneme_ids: &Array,
        bert_features: &Array,
        phoneme_count: usize,
        prompt_semantic: Option<&Array>,
    ) -> Result<Vec<i32>, Error> {
        let batch_size = 1;
        let num_layers = self.t2s_config.num_layers as usize;
        let mut caches: Vec<Option<ConcatKeyValueCache>> = (0..num_layers).map(|_| None).collect();

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

        // First token
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..)).squeeze()
            .map_err(|e| Error::Message(e.to_string()))?;
        let mut token_id = sample_top_k(&last_logits, self.config.top_k, self.config.temperature)?;
        semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        let mut all_tokens = vec![token_id];

        // Generation bounds
        let target_tokens = (phoneme_count as f32 * 2.6) as usize;
        let max_tokens = (phoneme_count * 4).max(100);
        let min_tokens = (phoneme_count * 2).max(15);
        let eos_token = 1024;

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

            token_id = sample_top_k(&last_logits, self.config.top_k, self.config.temperature)?;

            // EOS detection
            if token_id == eos_token && all_tokens.len() >= min_tokens {
                break;
            }

            // Target overflow
            if all_tokens.len() > (target_tokens as f32 * 1.2) as usize {
                break;
            }

            // EOS retry if too early
            if token_id == eos_token {
                token_id = sample_top_k(&last_logits, self.config.top_k * 2, self.config.temperature * 1.5)?;
                if token_id == eos_token {
                    token_id = ((step * 37 + 127) % 1000) as i32;
                }
            }

            all_tokens.push(token_id);

            // Repetition detection
            if all_tokens.len() > min_tokens && detect_repetition(&all_tokens, 3, 8) {
                while all_tokens.len() > min_tokens && detect_repetition(&all_tokens, 3, 5) {
                    all_tokens.pop();
                }
                break;
            }

            semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        }

        // Debug: print token stats
        eprintln!("DEBUG: phoneme_count={}, target_tokens={}, max_tokens={}, min_tokens={}",
            phoneme_count, target_tokens, max_tokens, min_tokens);
        eprintln!("DEBUG: Generated {} tokens (target_overflow at {})",
            all_tokens.len(), (target_tokens as f32 * 1.2) as usize);
        if !all_tokens.is_empty() {
            eprintln!("DEBUG: First 20 tokens: {:?}", &all_tokens[..20.min(all_tokens.len())]);
            eprintln!("DEBUG: Last 10 tokens: {:?}", &all_tokens[all_tokens.len().saturating_sub(10)..]);
            let unique: std::collections::HashSet<_> = all_tokens.iter().collect();
            eprintln!("DEBUG: Unique tokens: {}", unique.len());
        }

        Ok(all_tokens)
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

/// Sample from logits using top-k sampling
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
