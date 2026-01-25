//! BERT Feature Extraction for GPT-SoVITS TTS
//!
//! This module provides the correct BERT feature extraction pipeline:
//! 1. Tokenize text with BERT tokenizer (not phoneme IDs)
//! 2. Run BERT model and extract 3rd-from-last hidden layer
//! 3. Remove CLS/SEP tokens
//! 4. Expand features according to word2ph to align with phonemes
//!
//! This matches the Python GPT-SoVITS implementation exactly.

use std::path::Path;

use mlx_rs::{Array, transforms::eval};
use tokenizers::Tokenizer;

use crate::error::Error;
use crate::models::bert::{BertModel, BertModelInput, load_bert_model};

/// BERT Feature Extractor for TTS
///
/// Combines BERT tokenizer and model to extract features aligned with phonemes.
pub struct BertFeatureExtractor {
    /// BERT tokenizer (HuggingFace tokenizers)
    tokenizer: Tokenizer,
    /// BERT model
    model: BertModel,
    /// Which hidden layer to use (-3 = 3rd from last)
    layer_idx: i32,
}

impl BertFeatureExtractor {
    /// Create a new BERT feature extractor
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to tokenizer.json (HuggingFace format)
    /// * `model_path` - Path to BERT weights (safetensors)
    /// * `layer_idx` - Which layer to use for features (-3 = 3rd from last)
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        tokenizer_path: P1,
        model_path: P2,
        layer_idx: i32,
    ) -> Result<Self, Error> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Message(format!("Failed to load tokenizer: {}", e)))?;

        let model = load_bert_model(model_path)?;

        Ok(Self {
            tokenizer,
            model,
            layer_idx,
        })
    }

    /// Load from default paths
    ///
    /// Expects tokenizer.json and bert.safetensors in the model directory.
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self, Error> {
        let dir = model_dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        let model_path = dir.join("bert.safetensors");

        Self::new(tokenizer_path, model_path, -3)
    }

    /// Tokenize text and return token IDs
    ///
    /// # Arguments
    /// * `text` - Input text
    ///
    /// # Returns
    /// Vector of token IDs (including CLS and SEP)
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>, Error> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| Error::Message(format!("Tokenization failed: {}", e)))?;

        let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        Ok(ids)
    }

    /// Extract BERT features for TTS
    ///
    /// This method:
    /// 1. Tokenizes the text with BERT tokenizer
    /// 2. Runs BERT and gets hidden states from specified layer
    /// 3. Removes CLS and SEP tokens
    /// 4. Expands features according to word2ph to align with phonemes
    ///
    /// # Arguments
    /// * `text` - Input text (will be tokenized)
    /// * `word2ph` - Number of phonemes per character (len must match text length)
    ///
    /// # Returns
    /// Features [1, total_phonemes, hidden_dim]
    pub fn extract_features(
        &mut self,
        text: &str,
        word2ph: &[i32],
    ) -> Result<Array, Error> {
        // Verify word2ph length matches text length
        let text_chars: Vec<char> = text.chars().collect();
        if word2ph.len() != text_chars.len() {
            return Err(Error::Message(format!(
                "word2ph length ({}) doesn't match text character count ({})",
                word2ph.len(), text_chars.len()
            )));
        }

        // Tokenize
        let token_ids = self.tokenize(text)?;

        // Create input array
        let input_ids = Array::from_slice(&token_ids, &[1, token_ids.len() as i32]);

        // Extract features with word2ph alignment
        let features = self.model.extract_features_for_tts(
            &input_ids,
            word2ph,
            self.layer_idx,
        )?;

        eval([&features])?;

        Ok(features)
    }

    /// Extract raw BERT hidden states (without word2ph expansion)
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `remove_cls_sep` - Whether to remove CLS and SEP tokens
    ///
    /// # Returns
    /// Features [1, seq_len, hidden_dim]
    pub fn extract_raw_features(
        &mut self,
        text: &str,
        remove_cls_sep: bool,
    ) -> Result<Array, Error> {
        use mlx_rs::ops::indexing::IndexOp;

        // Tokenize
        let token_ids = self.tokenize(text)?;

        // Create input array
        let input_ids = Array::from_slice(&token_ids, &[1, token_ids.len() as i32]);

        // Get hidden states
        let output = self.model.forward_with_hidden_states(BertModelInput {
            input_ids: &input_ids,
            token_type_ids: None,
            attention_mask: None,
        })?;

        // Get specified layer
        let num_layers = output.hidden_states.len() as i32;
        let actual_idx = if self.layer_idx < 0 {
            (num_layers + self.layer_idx) as usize
        } else {
            self.layer_idx as usize
        };

        let hidden = output.hidden_states.get(actual_idx)
            .ok_or_else(|| Error::Message(format!(
                "Layer index {} out of range (have {} layers)",
                self.layer_idx, num_layers
            )))?;

        let result = if remove_cls_sep {
            // Remove CLS (first) and SEP (last) tokens using index
            let seq_len = hidden.shape()[1] as i32;
            hidden.index((.., 1..(seq_len - 1), ..))
        } else {
            hidden.clone()
        };

        eval([&result])?;

        Ok(result)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// Convenience function to extract BERT features for TTS
///
/// # Arguments
/// * `text` - Input text
/// * `word2ph` - Phoneme counts per character
/// * `tokenizer_path` - Path to tokenizer.json
/// * `model_path` - Path to BERT weights
///
/// # Returns
/// Features [1, total_phonemes, hidden_dim]
pub fn extract_bert_features<P1: AsRef<Path>, P2: AsRef<Path>>(
    text: &str,
    word2ph: &[i32],
    tokenizer_path: P1,
    model_path: P2,
) -> Result<Array, Error> {
    let mut extractor = BertFeatureExtractor::new(tokenizer_path, model_path, -3)?;
    extractor.extract_features(text, word2ph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_chinese() {
        // This test requires the tokenizer file
        let tokenizer_path = "/tmp/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json";
        if !Path::new(tokenizer_path).exists() {
            println!("Skipping test: tokenizer not found at {}", tokenizer_path);
            return;
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let text = "你好";
        let encoding = tokenizer.encode(text, true).unwrap();
        let ids = encoding.get_ids();

        // Should have: [CLS] 你 好 [SEP]
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], 101);  // [CLS]
        assert_eq!(ids[ids.len() - 1], 102);  // [SEP]
    }

    #[test]
    fn test_word2ph_validation() {
        let tokenizer_path = "/tmp/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json";
        let model_path = "/tmp/gpt-sovits-mlx/bert.safetensors";

        if !Path::new(tokenizer_path).exists() || !Path::new(model_path).exists() {
            println!("Skipping test: required files not found");
            return;
        }

        let mut extractor = BertFeatureExtractor::new(tokenizer_path, model_path, -3).unwrap();

        // Test with mismatched word2ph
        let text = "你好";  // 2 characters
        let word2ph = vec![2, 2, 2];  // 3 entries - wrong!

        let result = extractor.extract_features(text, &word2ph);
        assert!(result.is_err());
    }
}
