//! Text processing for GPT-SoVITS
//!
//! This module provides text-to-phoneme conversion for TTS:
//! - Phoneme vocabulary and symbol mappings
//! - Text normalization (Chinese/English)
//! - Grapheme-to-phoneme conversion
//! - Language detection
//! - BERT feature extraction for TTS
//! - G2PW polyphonic character disambiguation

pub mod bert_features;
pub mod cmudict;
pub mod g2pw;
pub mod preprocessor;
pub mod symbols;

pub use bert_features::{BertFeatureExtractor, extract_bert_features};

pub use preprocessor::{
    Language, PreprocessorConfig, PreprocessorOutput, TextPreprocessor,
    detect_language, is_chinese_char, normalize_chinese, normalize_english,
    preprocess_text,
};

pub use symbols::{
    bos_id, eos_id, pad_id, sp_id, unk_id,
    has_symbol, id_to_symbol, ids_to_symbols, symbol_to_id, symbols_to_ids,
    vocab_size, all_symbols,
    PAD, UNK, BOS, EOS, SP,
};
