//! G2PW - Grapheme-to-Phoneme for Chinese Polyphonic Characters
//!
//! Uses ONNX Runtime with CoreML (GPU/ANE) to run the G2PW model for disambiguating
//! polyphonic Chinese characters.
//! Based on: https://github.com/GitYCC/g2pW

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::{ep, inputs, session::Session, value::Tensor};
use tokenizers::Tokenizer;

/// Global G2PW instance (lazy initialized, wrapped in Mutex for thread-safe mutable access)
static G2PW: OnceLock<Mutex<Option<G2PWConverter>>> = OnceLock::new();

/// Get pinyin for a sentence using the global G2PW converter
/// Returns a vector of Option<String> for each character (None for non-Chinese or unknown chars)
pub fn get_pinyin_with_g2pw(sentence: &str) -> Vec<Option<String>> {
    let mutex = G2PW.get_or_init(|| {
        // Try to load G2PW model from common locations
        let model_paths = [
            "/Users/yuechen/home/mcp/dora/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts/text/G2PWModel",
            "/Users/yuechen/.dora/models/primespeech/moyoyo/G2PWModel",
        ];

        for path in model_paths {
            if Path::new(path).exists() {
                match G2PWConverter::new(path) {
                    Ok(converter) => {
                        eprintln!("G2PW: Loaded from {}", path);
                        return Mutex::new(Some(converter));
                    }
                    Err(e) => {
                        eprintln!("G2PW: Failed to load from {}: {}", path, e);
                    }
                }
            }
        }
        eprintln!("G2PW: Model not found, polyphonic disambiguation disabled");
        Mutex::new(None)
    });

    if let Ok(mut guard) = mutex.lock() {
        if let Some(ref mut converter) = *guard {
            return converter.get_pinyin(sentence);
        }
    }

    // Fallback: return None for all characters
    vec![None; sentence.chars().count()]
}

/// G2PW Converter for polyphonic character disambiguation
pub struct G2PWConverter {
    session: Session,
    tokenizer: Tokenizer,
    /// Polyphonic characters that need ML inference
    polyphonic_chars: HashSet<char>,
    /// Monophonic characters with fixed pronunciation
    monophonic_chars: HashMap<char, String>,
    /// Bopomofo to pinyin conversion
    bopomofo_to_pinyin: HashMap<String, String>,
    /// Labels (phoneme predictions)
    labels: Vec<String>,
    /// Character to valid phoneme indices
    char2phonemes: HashMap<char, Vec<usize>>,
    /// Sorted list of polyphonic characters
    chars: Vec<char>,
}

impl G2PWConverter {
    /// Create a new G2PW converter
    pub fn new(model_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = Path::new(model_dir).join("g2pW.onnx");
        let polyphonic_path = Path::new(model_dir).join("POLYPHONIC_CHARS.txt");
        let monophonic_path = Path::new(model_dir).join("MONOPHONIC_CHARS.txt");
        let bopomofo_path = Path::new(model_dir).join("bopomofo_to_pinyin_wo_tune_dict.json");

        // Load ONNX session with CoreML execution provider for GPU/ANE acceleration
        // Falls back to CPU if CoreML is not available
        let cache_dir = Path::new(model_dir).join("coreml_cache");
        std::fs::create_dir_all(&cache_dir).ok();

        let coreml_ep = ep::CoreML::default()
            .with_compute_units(ep::coreml::ComputeUnits::All)  // Use GPU + ANE + CPU
            .with_model_format(ep::coreml::ModelFormat::NeuralNetwork)  // Better compatibility
            .with_model_cache_dir(cache_dir.to_string_lossy().to_string())  // Cache compiled model
            .build();

        eprintln!("G2PW: Using CoreML execution provider (GPU/ANE accelerated)");

        let session = Session::builder()?
            .with_execution_providers([coreml_ep])?
            .with_intra_threads(2)?
            .commit_from_file(&model_path)?;

        // Load tokenizer (bert-base-chinese)
        let tokenizer = Tokenizer::from_pretrained("bert-base-chinese", None)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Load polyphonic characters: "char\tbopomofo"
        let polyphonic_content = std::fs::read_to_string(&polyphonic_path)?;
        let polyphonic_pairs: Vec<(char, String)> = polyphonic_content
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() == 2 {
                    parts[0].chars().next().map(|c| (c, parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // Build labels (unique phonemes only, NOT char+phoneme)
        // The model was trained with use_char_phoneme=False
        let mut label_set: HashSet<String> = HashSet::new();
        for (_char, phoneme) in &polyphonic_pairs {
            label_set.insert(phoneme.clone());
        }
        let mut labels: Vec<String> = label_set.into_iter().collect();
        labels.sort();

        // Build char2phonemes mapping (char -> valid phoneme indices)
        let mut char2phonemes: HashMap<char, Vec<usize>> = HashMap::new();
        for (char, phoneme) in &polyphonic_pairs {
            if let Some(idx) = labels.iter().position(|l| l == phoneme) {
                char2phonemes.entry(*char).or_default().push(idx);
            }
        }
        // Deduplicate phoneme indices for each char
        for indices in char2phonemes.values_mut() {
            indices.sort();
            indices.dedup();
        }

        let mut chars: Vec<char> = char2phonemes.keys().copied().collect();
        chars.sort();

        // Characters to exclude from polyphonic processing
        let non_polyphonic: HashSet<char> = "一不和咋嗲剖差攢倒難奔勁拗肖瘙誒泊听噢"
            .chars().collect();

        let polyphonic_chars: HashSet<char> = chars.iter()
            .filter(|c| !non_polyphonic.contains(c))
            .copied()
            .collect();

        // Load monophonic characters
        let monophonic_content = std::fs::read_to_string(&monophonic_path)?;
        let non_monophonic: HashSet<char> = "似攢".chars().collect();
        let monophonic_chars: HashMap<char, String> = monophonic_content
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() == 2 {
                    parts[0].chars().next().map(|c| (c, parts[1].to_string()))
                } else {
                    None
                }
            })
            .filter(|(c, _)| !non_monophonic.contains(c))
            .collect();

        // Load bopomofo to pinyin mapping
        let bopomofo_content = std::fs::read_to_string(&bopomofo_path)?;
        let bopomofo_to_pinyin: HashMap<String, String> = serde_json::from_str(&bopomofo_content)?;

        Ok(Self {
            session,
            tokenizer,
            polyphonic_chars,
            monophonic_chars,
            bopomofo_to_pinyin,
            labels,
            char2phonemes,
            chars,
        })
    }

    /// Convert bopomofo to pinyin with tone
    fn bopomofo_to_pinyin(&self, bopomofo: &str) -> Option<String> {
        if bopomofo.is_empty() {
            return None;
        }
        let tone = bopomofo.chars().last()?;
        if !"12345".contains(tone) {
            return None;
        }
        let component = &bopomofo[..bopomofo.len() - tone.len_utf8()];
        self.bopomofo_to_pinyin.get(component).map(|p| format!("{}{}", p, tone))
    }

    /// Check if a character is polyphonic
    pub fn is_polyphonic(&self, c: char) -> bool {
        self.polyphonic_chars.contains(&c)
    }

    /// Get pinyin for a sentence, disambiguating polyphonic characters
    /// Returns a vector of Option<String> for each character
    pub fn get_pinyin(&mut self, sentence: &str) -> Vec<Option<String>> {
        let chars: Vec<char> = sentence.chars().collect();
        let mut results: Vec<Option<String>> = vec![None; chars.len()];

        // Collect polyphonic character positions
        let mut texts: Vec<String> = Vec::new();
        let mut query_ids: Vec<usize> = Vec::new();

        for (i, &c) in chars.iter().enumerate() {
            if self.polyphonic_chars.contains(&c) {
                texts.push(sentence.to_string());
                query_ids.push(i);
            } else if let Some(bopomofo) = self.monophonic_chars.get(&c) {
                results[i] = self.bopomofo_to_pinyin(bopomofo);
            }
            // Other characters left as None (will use pypinyin fallback)
        }

        if texts.is_empty() {
            return results;
        }

        // Prepare ONNX input and run inference
        if let Ok(predictions) = self.predict(&texts, &query_ids) {
            for (query_id, pred) in query_ids.iter().zip(predictions.iter()) {
                if let Some(pinyin) = self.bopomofo_to_pinyin(pred) {
                    results[*query_id] = Some(pinyin);
                }
            }
        }

        results
    }

    /// Run ONNX inference for polyphonic characters
    fn predict(&mut self, texts: &[String], query_ids: &[usize]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let mut all_input_ids: Vec<Vec<i64>> = Vec::new();
        let mut all_token_type_ids: Vec<Vec<i64>> = Vec::new();
        let mut all_attention_masks: Vec<Vec<i64>> = Vec::new();
        let mut all_phoneme_masks: Vec<Vec<f32>> = Vec::new();
        let mut all_char_ids: Vec<i64> = Vec::new();
        let mut all_position_ids: Vec<i64> = Vec::new();

        let num_labels = self.labels.len();

        for (text, &query_id) in texts.iter().zip(query_ids.iter()) {
            let text_lower = text.to_lowercase();
            let chars: Vec<char> = text_lower.chars().collect();

            // Tokenize
            let encoding = self.tokenizer.encode(text_lower.clone(), true)
                .map_err(|e| format!("Tokenization failed: {}", e))?;

            let tokens = encoding.get_ids();
            let input_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let token_type_ids: Vec<i64> = vec![0; input_ids.len()];
            let attention_mask: Vec<i64> = vec![1; input_ids.len()];

            // Get query character and build phoneme mask
            let query_char = chars.get(query_id).copied().unwrap_or(' ');
            let phoneme_mask: Vec<f32> = if let Some(valid_phonemes) = self.char2phonemes.get(&query_char) {
                (0..num_labels).map(|i| if valid_phonemes.contains(&i) { 1.0 } else { 0.0 }).collect()
            } else {
                vec![1.0; num_labels]
            };

            // Get char_id
            let char_id = self.chars.iter().position(|&c| c == query_char).unwrap_or(0) as i64;

            // Get position_id (token position for query character)
            // This is approximate - we use the character offset
            let position_id = (query_id + 1) as i64; // +1 for [CLS] token

            all_input_ids.push(input_ids);
            all_token_type_ids.push(token_type_ids);
            all_attention_masks.push(attention_mask);
            all_phoneme_masks.push(phoneme_mask);
            all_char_ids.push(char_id);
            all_position_ids.push(position_id);
        }

        // Pad sequences to same length
        let max_len = all_input_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        for i in 0..batch_size {
            let pad_len = max_len - all_input_ids[i].len();
            all_input_ids[i].extend(vec![0i64; pad_len]);
            all_token_type_ids[i].extend(vec![0i64; pad_len]);
            all_attention_masks[i].extend(vec![0i64; pad_len]);
        }

        // Flatten for ONNX
        let input_ids_flat: Vec<i64> = all_input_ids.into_iter().flatten().collect();
        let token_type_ids_flat: Vec<i64> = all_token_type_ids.into_iter().flatten().collect();
        let attention_masks_flat: Vec<i64> = all_attention_masks.into_iter().flatten().collect();
        let phoneme_masks_flat: Vec<f32> = all_phoneme_masks.into_iter().flatten().collect();

        // Create ONNX tensors
        let input_ids = Tensor::from_array(([batch_size, max_len], input_ids_flat.into_boxed_slice()))?;
        let token_type_ids = Tensor::from_array(([batch_size, max_len], token_type_ids_flat.into_boxed_slice()))?;
        let attention_mask = Tensor::from_array(([batch_size, max_len], attention_masks_flat.into_boxed_slice()))?;
        let phoneme_mask = Tensor::from_array(([batch_size, num_labels], phoneme_masks_flat.into_boxed_slice()))?;
        let char_ids = Tensor::from_array(([batch_size], all_char_ids.into_boxed_slice()))?;
        let position_ids = Tensor::from_array(([batch_size], all_position_ids.into_boxed_slice()))?;

        // Run inference
        let outputs = self.session.run(inputs![
            "input_ids" => input_ids,
            "token_type_ids" => token_type_ids,
            "attention_mask" => attention_mask,
            "phoneme_mask" => phoneme_mask,
            "char_ids" => char_ids,
            "position_ids" => position_ids,
        ])?;

        // Get predictions - outputs["probs"] returns (&Shape, &[f32])
        let probs_value = &outputs["probs"];
        let (_shape, probs_data) = probs_value.try_extract_tensor::<f32>()?;

        let mut predictions = Vec::new();
        for i in 0..batch_size {
            let row_start = i * num_labels;
            let row: Vec<f32> = (0..num_labels).map(|j| probs_data[row_start + j]).collect();

            // Find argmax
            let pred_idx = row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Get label (it's just the phoneme, since use_char_phoneme=False)
            let phoneme = &self.labels[pred_idx];
            predictions.push(phoneme.to_string());
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2pw_loading() {
        if let Some(g2pw) = get_g2pw() {
            assert!(g2pw.is_polyphonic('行'));
            assert!(g2pw.is_polyphonic('了'));
            assert!(!g2pw.is_polyphonic('我'));
        }
    }
}
