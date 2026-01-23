//! Text preprocessor for GPT-SoVITS
//!
//! Converts text to phoneme sequences for Chinese and English.
//!
//! Pipeline:
//! 1. Text normalization
//! 2. Language detection
//! 3. Grapheme-to-phoneme conversion
//! 4. Phoneme ID conversion

use std::collections::HashMap;

use pinyin::ToPinyin;

use super::symbols::{self, bos_id, eos_id, has_symbol, symbol_to_id};

/// Detected language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Mixed,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Chinese => "zh",
            Language::English => "en",
            Language::Mixed => "mixed",
        }
    }
}

/// Output from text preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessorOutput {
    /// Phoneme IDs
    pub phoneme_ids: Vec<i32>,
    /// Phoneme strings
    pub phonemes: Vec<String>,
    /// Number of phonemes per word/character
    pub word2ph: Vec<i32>,
    /// Normalized text
    pub text_normalized: String,
    /// Detected/specified language
    pub language: Language,
}

/// Pinyin initials (consonants)
const PINYIN_INITIALS: &[&str] = &[
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh",
];

/// Multi-character initials (check these first)
const MULTI_CHAR_INITIALS: &[&str] = &["zh", "ch", "sh"];

/// Zero-initial vowel mapping
fn zero_initial_map() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut map = HashMap::new();
    map.insert("a", ("AA", "a"));
    map.insert("ai", ("AA", "ai"));
    map.insert("an", ("AA", "an"));
    map.insert("ang", ("AA", "ang"));
    map.insert("ao", ("AA", "ao"));
    map.insert("e", ("EE", "e"));
    map.insert("ei", ("EE", "ei"));
    map.insert("en", ("EE", "en"));
    map.insert("eng", ("EE", "eng"));
    map.insert("er", ("EE", "er"));
    map.insert("o", ("OO", "o"));
    map.insert("ou", ("OO", "ou"));
    map
}

/// Full-width to half-width punctuation mapping
fn fullwidth_to_halfwidth() -> HashMap<char, char> {
    let mut map = HashMap::new();
    map.insert('，', ',');
    map.insert('。', '.');
    map.insert('！', '!');
    map.insert('？', '?');
    map.insert('；', ';');
    map.insert('：', ':');
    map.insert('、', ',');
    map.insert('"', '"');
    map.insert('"', '"');
    map.insert('\u{2018}', '\'');  // Left single quote
    map.insert('\u{2019}', '\'');  // Right single quote
    map.insert('（', '(');
    map.insert('）', ')');
    map.insert('【', '[');
    map.insert('】', ']');
    map.insert('《', '"');
    map.insert('》', '"');
    map.insert('～', '~');
    map
}

/// Check if character is Chinese
pub fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)      // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&code)   // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&code) // CJK Extension B
        || (0xF900..=0xFAFF).contains(&code)   // CJK Compatibility Ideographs
}

/// Detect primary language of text
///
/// Returns `Mixed` if both Chinese and English characters are present,
/// regardless of which has more. This ensures proper phoneme conversion
/// for code-switching text like "Hello世界".
pub fn detect_language(text: &str) -> Language {
    let chinese_count = text.chars().filter(|&c| is_chinese_char(c)).count();
    let english_count = text.chars().filter(|&c| c.is_ascii_alphabetic()).count();

    // If both Chinese and English are present, treat as mixed
    if chinese_count > 0 && english_count > 0 {
        Language::Mixed
    } else if chinese_count > 0 {
        Language::Chinese
    } else if english_count > 0 {
        Language::English
    } else {
        // No letters found, default to Chinese (handles punctuation-only)
        Language::Chinese
    }
}

/// Normalize Chinese text (full-width to half-width punctuation)
pub fn normalize_chinese(text: &str) -> String {
    let map = fullwidth_to_halfwidth();
    text.chars()
        .map(|c| *map.get(&c).unwrap_or(&c))
        .collect()
}

/// Normalize Chinese text for BERT (removes English characters, keeps Chinese and punctuation)
/// This matches Python's replace_punctuation behavior
pub fn normalize_chinese_for_bert(text: &str) -> String {
    let map = fullwidth_to_halfwidth();
    text.chars()
        .filter_map(|c| {
            // Convert full-width punctuation first
            let c = *map.get(&c).unwrap_or(&c);
            // Keep only Chinese characters and basic punctuation
            if is_chinese_char(c) || is_punctuation(c) {
                Some(c)
            } else {
                None
            }
        })
        .collect()
}

/// Check if character is punctuation (matching Python's punctuation set)
fn is_punctuation(c: char) -> bool {
    matches!(c,
        '!' | '"' | '#' | '$' | '%' | '&' | '\'' | '(' | ')' | '*' |
        '+' | ',' | '-' | '.' | '/' | ':' | ';' | '<' | '=' | '>' |
        '?' | '@' | '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' |
        '}' | '~' | ' '
    )
}

/// Normalize English text
pub fn normalize_english(text: &str) -> String {
    // Remove extra whitespace
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split pinyin into initial (consonant) and final (vowel with tone)
///
/// # Arguments
/// * `pinyin` - Pinyin syllable with tone number (e.g., "ni3", "hao3")
///
/// # Returns
/// Tuple of (initial, final) where final includes tone number
///
/// # Apical Vowel Handling
///
/// In Mandarin Chinese, the "i" vowel has two distinct pronunciations:
///
/// 1. **Normal "i"** (as in English "bee"): Used in syllables like xi, bi, pi, mi, di, ti, ni, li
///    - Encoded as `i1`, `i2`, `i3`, `i4`, `i5` (with tone number)
///
/// 2. **Apical vowel "i"** (a buzzing sound, no English equivalent): Used after z, c, s, zh, ch, sh, r
///    - zi (资), ci (次), si (四), zhi (知), chi (吃), shi (是), ri (日)
///    - Encoded as `i01`, `i02`, `i03`, `i04`, `i05` (with tone number)
///    - This is phonetically written as [ɿ] (after z/c/s) or [ʅ] (after zh/ch/sh/r) in IPA
///
/// This distinction is critical for correct TTS pronunciation:
/// - 司 (sī) uses apical vowel → phonemes: `s` + `i01`
/// - 西 (xī) uses normal vowel → phonemes: `x` + `i1`
///
/// Without this distinction, words like 司/西, 次/戏, 四/细 would sound identical.
pub fn get_initial_final(pinyin: &str) -> (Option<&'static str>, String) {
    // Extract tone number if present
    let (pinyin_base, tone) = if pinyin.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        let tone = pinyin.chars().last().unwrap();
        (&pinyin[..pinyin.len()-1], tone)
    } else {
        (pinyin, '5') // Neutral tone
    };

    // Check for multi-character initials first (zh, ch, sh)
    for &initial in MULTI_CHAR_INITIALS {
        if pinyin_base.starts_with(initial) {
            let final_part = &pinyin_base[initial.len()..];
            // Special case: apical vowel "i" after zh/ch/sh/r becomes "i0"
            // This is the buzzing vowel in zhi/chi/shi/ri, different from normal "i"
            let final_str = if final_part == "i" && (initial == "zh" || initial == "ch" || initial == "sh") {
                format!("i0{}", tone)
            } else {
                format!("{}{}", final_part, tone)
            };
            return (Some(initial), final_str);
        }
    }

    // Single character initials
    for &initial in PINYIN_INITIALS {
        if initial.len() == 1 && pinyin_base.starts_with(initial) {
            let final_part = &pinyin_base[1..];
            // Special case: apical vowel "i" after z/c/s/r becomes "i0"
            // This is the buzzing vowel in zi/ci/si/ri, different from normal "i" in xi/bi/pi
            let final_str = if final_part == "i" && (initial == "z" || initial == "c" || initial == "s" || initial == "r") {
                format!("i0{}", tone)
            } else {
                format!("{}{}", final_part, tone)
            };
            return (Some(initial), final_str);
        }
    }

    // Zero initial - check mapping
    let zero_map = zero_initial_map();
    if let Some(&(init, vowel)) = zero_map.get(pinyin_base) {
        return (Some(init), format!("{}{}", vowel, tone));
    }

    // Default: treat entire pinyin as final with special initial
    (Some("AA"), format!("{}{}", pinyin_base, tone))
}

/// Convert Chinese character to pinyin using the pinyin crate
fn get_pinyin_for_char(c: char) -> Option<String> {
    // Use the pinyin crate for full Chinese character coverage
    // ToPinyin trait works on &str slices
    let char_str = c.to_string();
    let char_slice: &str = &char_str;
    for pinyin_result in char_slice.to_pinyin() {
        if let Some(pinyin) = pinyin_result {
            // Use with_tone_num_end() for format like "ni3"
            let mut result = pinyin.with_tone_num_end().to_string();

            // Convert 'ü' to 'v' for GPT-SoVITS symbol table compatibility
            result = result.replace('ü', "v");

            // Ensure tone number is present (add neutral tone 5 if missing)
            if !result.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                result.push('5');
            }

            return Some(result);
        }
    }
    None
}

/// Convert Chinese character to phonemes
fn char_to_phonemes(c: char) -> Vec<String> {
    if let Some(pinyin) = get_pinyin_for_char(c) {
        let (initial, final_part) = get_initial_final(&pinyin);
        let mut phonemes = Vec::new();
        if let Some(init) = initial {
            if has_symbol(init) {
                phonemes.push(init.to_string());
            }
        }
        if has_symbol(&final_part) {
            phonemes.push(final_part);
        }
        if phonemes.is_empty() {
            // Fallback: return unknown
            phonemes.push(symbols::UNK.to_string());
        }
        phonemes
    } else if is_chinese_char(c) {
        // Unknown Chinese character (shouldn't happen with pinyin crate)
        vec![symbols::UNK.to_string()]
    } else {
        // Non-Chinese character
        vec![]
    }
}

/// Convert Chinese text to phonemes
pub fn chinese_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    let mut phonemes = Vec::new();
    let mut word2ph = Vec::new();

    for c in text.chars() {
        if c.is_whitespace() {
            phonemes.push(symbols::SP.to_string());
            word2ph.push(1);
        } else if c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' {
            phonemes.push(c.to_string());
            word2ph.push(1);
        } else if is_chinese_char(c) {
            let char_phonemes = char_to_phonemes(c);
            let count = char_phonemes.len() as i32;
            phonemes.extend(char_phonemes);
            word2ph.push(count);
        } else if c.is_ascii_alphabetic() {
            // English letter in Chinese text
            phonemes.push(c.to_ascii_uppercase().to_string());
            word2ph.push(1);
        }
    }

    (phonemes, word2ph)
}

/// Convert English text to phonemes using CMU dictionary
pub fn english_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    use super::cmudict;

    let mut phonemes = Vec::new();
    let mut word2ph = Vec::new();

    // Split text into words, preserving punctuation
    let mut current_word = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_alphabetic() || c == '\'' {
            current_word.push(c);
        } else {
            // Process accumulated word
            if !current_word.is_empty() {
                let word_phonemes = cmudict::word_to_phonemes(&current_word);
                let count = word_phonemes.len() as i32;
                phonemes.extend(word_phonemes);
                word2ph.push(count);
                current_word.clear();
            }

            // Handle punctuation and spaces
            if c.is_whitespace() {
                // Skip multiple spaces
                while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
                    chars.next();
                }
            } else if has_symbol(&c.to_string()) {
                phonemes.push(c.to_string());
                word2ph.push(1);
            }
        }
    }

    // Process final word if any
    if !current_word.is_empty() {
        let word_phonemes = cmudict::word_to_phonemes(&current_word);
        let count = word_phonemes.len() as i32;
        phonemes.extend(word_phonemes);
        word2ph.push(count);
    }

    (phonemes, word2ph)
}

/// Language segment for mixed text processing
#[derive(Debug, Clone)]
struct LangSegment {
    text: String,
    is_english: bool,
}

/// Segment text into Chinese and English chunks
fn segment_by_language(text: &str) -> Vec<LangSegment> {
    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_is_english: Option<bool> = None;

    for c in text.chars() {
        let is_en = c.is_ascii_alphabetic();
        let is_zh = is_chinese_char(c);
        let is_punct = is_punctuation(c) || c.is_whitespace();

        if is_en {
            // English character
            if current_is_english == Some(false) && !current_text.is_empty() {
                segments.push(LangSegment { text: current_text.clone(), is_english: false });
                current_text.clear();
            }
            current_text.push(c);
            current_is_english = Some(true);
        } else if is_zh {
            // Chinese character
            if current_is_english == Some(true) && !current_text.is_empty() {
                segments.push(LangSegment { text: current_text.clone(), is_english: true });
                current_text.clear();
            }
            current_text.push(c);
            current_is_english = Some(false);
        } else if is_punct {
            // Punctuation belongs to current segment
            current_text.push(c);
        }
        // Skip other characters
    }

    // Add final segment
    if !current_text.is_empty() {
        segments.push(LangSegment {
            text: current_text,
            is_english: current_is_english.unwrap_or(false)
        });
    }

    segments
}

/// Convert mixed Chinese/English text to phonemes
pub fn mixed_g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    let segments = segment_by_language(text);
    let mut all_phonemes = Vec::new();
    let mut all_word2ph = Vec::new();

    for segment in segments {
        let (phonemes, word2ph) = if segment.is_english {
            english_g2p(&segment.text)
        } else {
            chinese_g2p(&segment.text)
        };
        all_phonemes.extend(phonemes);
        all_word2ph.extend(word2ph);
    }

    (all_phonemes, all_word2ph)
}

/// Text preprocessor configuration
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    /// Default language if not detected
    pub default_language: Language,
    /// Whether to add BOS token
    pub add_bos: bool,
    /// Whether to add EOS token
    pub add_eos: bool,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            default_language: Language::Chinese,
            add_bos: true,
            add_eos: true,
        }
    }
}

/// Text preprocessor
pub struct TextPreprocessor {
    config: PreprocessorConfig,
}

impl TextPreprocessor {
    /// Create new preprocessor with config
    pub fn new(config: PreprocessorConfig) -> Self {
        Self { config }
    }

    /// Preprocess text to phonemes
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `language` - Optional language override (None for auto-detect)
    pub fn preprocess(&self, text: &str, language: Option<Language>) -> PreprocessorOutput {
        if text.trim().is_empty() {
            return PreprocessorOutput {
                phoneme_ids: if self.config.add_bos {
                    vec![bos_id(), eos_id()]
                } else {
                    vec![eos_id()]
                },
                phonemes: if self.config.add_bos {
                    vec!["BOS".to_string(), "EOS".to_string()]
                } else {
                    vec!["EOS".to_string()]
                },
                word2ph: if self.config.add_bos { vec![1, 1] } else { vec![1] },
                text_normalized: String::new(),
                language: language.unwrap_or(self.config.default_language),
            };
        }

        // Detect language if not specified
        let language = language.unwrap_or_else(|| detect_language(text));

        // Normalize text
        let text_normalized = match language {
            Language::Chinese => normalize_chinese(text),
            Language::English => normalize_english(text),
            Language::Mixed => normalize_chinese(text),
        };

        // Convert to phonemes
        let (mut phonemes, mut word2ph) = match language {
            Language::Chinese => chinese_g2p(&text_normalized),
            Language::English => english_g2p(&text_normalized),
            Language::Mixed => {
                // For mixed, segment by language and process each segment
                mixed_g2p(&text_normalized)
            }
        };

        // Add BOS/EOS tokens
        if self.config.add_bos {
            phonemes.insert(0, symbols::BOS.to_string());
            word2ph.insert(0, 1);
        }
        if self.config.add_eos {
            phonemes.push(symbols::EOS.to_string());
            word2ph.push(1);
        }

        // Convert to IDs
        let phoneme_ids: Vec<i32> = phonemes
            .iter()
            .map(|s| symbol_to_id(s))
            .collect();

        PreprocessorOutput {
            phoneme_ids,
            phonemes,
            word2ph,
            text_normalized,
            language,
        }
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new(PreprocessorConfig::default())
    }
}

/// Convenience function to preprocess text
pub fn preprocess_text(text: &str, language: Option<Language>) -> PreprocessorOutput {
    TextPreprocessor::default().preprocess(text, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_chinese_char() {
        assert!(is_chinese_char('你'));
        assert!(is_chinese_char('好'));
        assert!(is_chinese_char('世'));
        assert!(!is_chinese_char('a'));
        assert!(!is_chinese_char('1'));
        assert!(!is_chinese_char(' '));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("你好世界"), Language::Chinese);
        assert_eq!(detect_language("hello world"), Language::English);
        // "你好 world" has both Chinese and English -> Mixed
        assert_eq!(detect_language("你好 world"), Language::Mixed);
        // Any mix of Chinese and English is Mixed
        assert_eq!(detect_language("你好wo"), Language::Mixed);
        assert_eq!(detect_language("Hello世界"), Language::Mixed);
    }

    #[test]
    fn test_normalize_chinese() {
        assert_eq!(normalize_chinese("你好，世界！"), "你好,世界!");
        assert_eq!(normalize_chinese("（测试）"), "(测试)");
    }

    #[test]
    fn test_get_initial_final() {
        let (init, final_) = get_initial_final("ni3");
        assert_eq!(init, Some("n"));
        assert_eq!(final_, "i3");

        let (init, final_) = get_initial_final("hao3");
        assert_eq!(init, Some("h"));
        assert_eq!(final_, "ao3");

        let (init, final_) = get_initial_final("shi4");
        assert_eq!(init, Some("sh"));
        assert_eq!(final_, "i4");

        let (init, final_) = get_initial_final("zhi1");
        assert_eq!(init, Some("zh"));
        assert_eq!(final_, "i1");
    }

    #[test]
    fn test_chinese_g2p() {
        let (phonemes, word2ph) = chinese_g2p("你好");
        // "你" -> "n" + "i3" (2 phonemes)
        // "好" -> "h" + "ao3" (2 phonemes)
        assert!(!phonemes.is_empty());
        assert_eq!(phonemes.len(), word2ph.iter().sum::<i32>() as usize);
    }

    #[test]
    fn test_english_g2p() {
        let (phonemes, word2ph) = english_g2p("hello world");
        assert!(!phonemes.is_empty());
        // Each letter becomes a phoneme
        assert!(phonemes.contains(&"H".to_string()));
        assert!(phonemes.contains(&"E".to_string()));
    }

    #[test]
    fn test_preprocessor() {
        let preprocessor = TextPreprocessor::default();

        let output = preprocessor.preprocess("你好", Some(Language::Chinese));
        assert!(!output.phoneme_ids.is_empty());
        assert!(output.phonemes.contains(&"BOS".to_string()));
        assert!(output.phonemes.contains(&"EOS".to_string()));
    }

    #[test]
    fn test_empty_text() {
        let preprocessor = TextPreprocessor::default();
        let output = preprocessor.preprocess("", None);
        assert_eq!(output.phonemes, vec!["BOS", "EOS"]);
    }

    #[test]
    fn test_preprocess_text_convenience() {
        let output = preprocess_text("你好", Some(Language::Chinese));
        assert!(!output.phoneme_ids.is_empty());
        assert_eq!(output.language, Language::Chinese);
    }
}
