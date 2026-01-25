# GPT-SoVITS Rust Implementation Architecture

## Overview

This document describes the architecture of the pure Rust GPT-SoVITS voice cloning implementation in `mlx-rs-lm`. The implementation supports zero-shot and few-shot voice cloning without any Python dependencies.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VoiceCloner API                                 │
│                         (src/voice_clone.rs)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
         ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
         │  Text Processing │ │    Models    │ │  Audio Processing│
         │   (src/text/)    │ │(src/models/) │ │  (src/audio.rs)  │
         └──────────────────┘ └──────────────┘ └──────────────────┘
```

## Pipeline Comparison: Zero-Shot vs Few-Shot

### Zero-Shot Mode

Uses only reference audio for voice style (mel spectrogram). Simpler but less accurate voice matching.

```
Input Text ──► Text Preprocessing ──► Phoneme IDs
                                           │
                                           ▼
Reference Audio ──► Mel Spectrogram    BERT Features (zeros for mixed/English)
       │                 │                 │
       │                 │                 ▼
       │                 │           T2S Model ──► Semantic Tokens
       │                 │                              │
       │                 ▼                              ▼
       │              VITS Vocoder ◄─────────── Phoneme IDs
       │                   │
       ▼                   ▼
   Ref Style ──────► Audio Output
```

**Code Path:**
```
VoiceCloner::synthesize()
  └── synthesize_zero_shot()
        ├── preprocess_text()           # Text → Phonemes
        ├── extract_bert_features()     # BERT encoding (zeros for non-Chinese)
        ├── generate_semantic_tokens()  # T2S: Phonemes + BERT → Semantic tokens
        └── vocode()                    # VITS: Semantic + Phonemes + Mel → Audio
```

### Few-Shot Mode

Uses reference audio + reference text + HuBERT semantic codes. Better voice matching and prosody.

```
Reference Audio ──► HuBERT ──► Prompt Semantic Codes
       │                              │
       ▼                              │
  Mel Spectrogram                     │
                                      │
Reference Text ──► Preprocessing ──► Ref Phonemes + Ref BERT
                                           │
Input Text ────► Preprocessing ────► Target Phonemes + Target BERT
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │   Concatenate:        │
                               │   - Ref + Target Phonemes │
                               │   - Ref + Target BERT │
                               │   - Prompt Semantic   │
                               └───────────────────────┘
                                           │
                                           ▼
                                      T2S Model ──► New Semantic Tokens
                                                         │
                                                         ▼
                                    VITS Vocoder ◄─── Target Phonemes
                                          │
                                          ▼
                                     Audio Output
```

**Code Path:**
```
VoiceCloner::synthesize()
  └── synthesize_few_shot()
        ├── preprocess_text(ref_text)      # Reference text → phonemes
        ├── extract_bert_features(ref)     # Reference BERT features
        ├── preprocess_text(target_text)   # Target text → phonemes
        ├── extract_bert_features(target)  # Target BERT features
        ├── concatenate(ref + target)      # Combine all inputs
        ├── generate_semantic_tokens()     # T2S with prompt_semantic prefix
        └── vocode()                       # VITS: uses target phonemes only
```

## Component Details

### 1. Text Processing (`src/text/`)

```
src/text/
├── mod.rs              # Module exports
├── preprocessor.rs     # Language detection, G2P conversion
├── symbols.rs          # Phoneme vocabulary (322 symbols)
├── cmudict.rs          # CMU dictionary for English
└── bert_features.rs    # BERT feature extraction
```

#### Text Pipeline

```
Input Text
    │
    ▼
Language Detection ──► Chinese / English / Mixed
    │
    ├──► Chinese: pypinyin → Initial + Final (with tone)
    │              Example: "你" → ["n", "i3"]
    │
    ├──► English: CMU Dictionary → ARPAbet phonemes
    │              Example: "movie" → ["M", "UW1", "V", "IY0"]
    │
    └──► Mixed: Segment by language, process each
               Example: "这部movie" → ["zh", "e4", "b", "u4", "M", "UW1", "V", "IY0"]
    │
    ▼
Phoneme IDs (symbol_to_id lookup)
```

#### Phoneme Symbol Table

The symbol table contains 322 phonemes:

| Category | Examples | Description |
|----------|----------|-------------|
| Punctuation | `!` `,` `.` `?` | Preserved in output |
| Chinese Initials | `b` `p` `m` `f` `zh` `ch` `sh` | Consonants |
| Chinese Finals | `a1`-`a5` `ai1`-`ai5` `i01`-`i05` | Vowels with tones |
| English ARPAbet | `AA0` `AE1` `IY0` `UW1` | CMU phonemes |
| Special | `SP` `UNK` `_` (BOS) `!` (EOS) | Control tokens |

### 2. Models (`src/models/`)

```
src/models/
├── bert.rs      # Chinese BERT for text encoding
├── t2s.rs       # Text-to-Semantic (GPT-style decoder)
├── vits.rs      # VITS vocoder
└── hubert.rs    # HuBERT for audio → semantic codes
```

#### Model Flow

```
                    ┌─────────────┐
   Phoneme IDs ────►│             │
                    │   T2S       │──► Semantic Tokens (1024 vocab)
BERT Features ─────►│  (GPT-2)    │
                    │             │
Prompt Semantic ───►│             │
 (few-shot only)    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
Semantic Tokens ───►│             │
                    │   VITS      │──► Audio Waveform
  Phoneme IDs ─────►│  Vocoder    │
                    │             │
Reference Mel ─────►│             │
                    └─────────────┘
```

### 3. Audio Processing (`src/audio.rs`)

- **Input**: WAV files (any sample rate)
- **Mel Spectrogram**: 100-dim, hop=256, win=1024
- **Output**: 32kHz WAV

## How to Fix Wrong Pronunciations

### Problem: Character pronounced incorrectly

#### For Chinese Characters

1. **Check pinyin output**:
```rust
// In preprocessor.rs, get_pinyin_for_char()
let pinyin = get_pinyin_for_char('熵');  // Should be "shang1"
```

2. **Add override in preprocessor** (if pypinyin is wrong):
```rust
// Add to char_to_phonemes() in preprocessor.rs
fn char_to_phonemes(c: char) -> Vec<String> {
    // Add manual overrides for problematic characters
    match c {
        '熵' => return vec!["sh".to_string(), "ang1".to_string()],
        // ... other overrides
        _ => {}
    }
    // ... rest of function
}
```

3. **For polyphones (characters with multiple readings)**:
```rust
// Create a context-aware lookup
// Example: 了 can be "le5" (particle) or "liao3" (understand)
fn get_pinyin_with_context(text: &str, pos: usize) -> String {
    let c = text.chars().nth(pos).unwrap();
    match c {
        '了' => {
            // Check context
            if is_sentence_final(text, pos) {
                "le5".to_string()  // Particle
            } else {
                "liao3".to_string()  // Verb
            }
        }
        _ => get_pinyin_for_char(c).unwrap_or_default()
    }
}
```

#### For English Words

1. **Add to CMU dictionary** (`src/text/cmudict.rs`):
```rust
// In CMU_DICT LazyLock
m.insert("restaurant", &["R", "EH1", "S", "T", "ER0", "AA2", "N", "T"][..]);
m.insert("genre", &["ZH", "AA1", "N", "R", "AH0"][..]);
```

2. **For brand names or neologisms**:
```rust
m.insert("chatgpt", &["CH", "AE1", "T", "JH", "IY1", "P", "IY1", "T", "IY1"][..]);
m.insert("openai", &["OW1", "P", "AH0", "N", "EY1", "AY1"][..]);
```

### Debugging Pronunciation Issues

```rust
// Add debug output in preprocessor.rs
let output = preprocess_text("问题文本", None);
for (i, ph) in output.phonemes.iter().enumerate() {
    println!("{}: {} -> ID {}", i, ph, output.phoneme_ids[i]);
}
```

## How to Add 方言 (Dialect) Support

### Architecture for Dialects

```
src/text/
├── preprocessor.rs
├── dialects/
│   ├── mod.rs           # Dialect trait + registry
│   ├── cantonese.rs     # 粤语
│   ├── hokkien.rs       # 闽南语
│   ├── shanghainese.rs  # 上海话
│   └── sichuanese.rs    # 四川话
```

### Step 1: Define Dialect Trait

```rust
// src/text/dialects/mod.rs

pub trait Dialect {
    /// Dialect identifier
    fn id(&self) -> &'static str;

    /// Convert character to dialect phonemes
    fn char_to_phonemes(&self, c: char) -> Option<Vec<String>>;

    /// Get additional symbols needed for this dialect
    fn additional_symbols(&self) -> &[&str];

    /// Tone sandhi rules (optional)
    fn apply_tone_sandhi(&self, phonemes: &mut Vec<String>) {}
}
```

### Step 2: Implement Cantonese Example

```rust
// src/text/dialects/cantonese.rs

use super::Dialect;

pub struct Cantonese;

impl Dialect for Cantonese {
    fn id(&self) -> &'static str { "yue" }

    fn char_to_phonemes(&self, c: char) -> Option<Vec<String>> {
        // Cantonese has 6 tones + different initials/finals
        // Use jyutping romanization
        CANTONESE_DICT.get(&c).map(|p| p.to_vec())
    }

    fn additional_symbols(&self) -> &[&str] {
        // Cantonese-specific phonemes not in Mandarin
        &[
            "aa1", "aa2", "aa3", "aa4", "aa5", "aa6",  // Long 'a' with 6 tones
            "eo1", "eo2", "eo3", "eo4", "eo5", "eo6",  // Schwa vowel
            "ng",   // Syllabic ng (五, 吳)
            "gw",   // Labialized velar (廣, 國)
            "kw",   // Labialized velar aspirated
            // ... more Cantonese-specific phonemes
        ]
    }
}

lazy_static! {
    static ref CANTONESE_DICT: HashMap<char, Vec<String>> = {
        let mut m = HashMap::new();
        // 你 in Cantonese is "nei5" (not "ni3")
        m.insert('你', vec!["n".into(), "ei5".into()]);
        // 好 in Cantonese is "hou2" (not "hao3")
        m.insert('好', vec!["h".into(), "ou2".into()]);
        // ... load from jyutping dictionary
        m
    };
}
```

### Step 3: Update Symbol Table

```rust
// src/text/symbols.rs

// Add dialect symbols dynamically
pub fn get_symbols_for_dialect(dialect_id: &str) -> Vec<&'static str> {
    let mut symbols = GPT_SOVITS_SYMBOLS.to_vec();

    match dialect_id {
        "yue" => {
            // Add Cantonese symbols
            symbols.extend(CANTONESE_SYMBOLS.iter());
        }
        "nan" => {
            // Add Hokkien symbols
            symbols.extend(HOKKIEN_SYMBOLS.iter());
        }
        _ => {}
    }

    symbols
}
```

### Step 4: Training Considerations

1. **Model weights**: Need dialect-specific T2S and VITS weights trained on dialect data
2. **BERT**: May need dialect-specific BERT or use multilingual BERT
3. **Data**: Need dialect speech corpus with transcriptions

### Step 5: Integration

```rust
// src/text/preprocessor.rs

pub fn preprocess_text_dialect(
    text: &str,
    dialect: Option<&dyn Dialect>
) -> PreprocessorOutput {
    let dialect = dialect.unwrap_or(&Mandarin);

    // Use dialect-specific G2P
    let (phonemes, word2ph) = dialect_g2p(text, dialect);

    // ... rest of preprocessing
}
```

## How to Add Foreign Languages (French, etc.)

### Architecture for Multilingual Support

```
src/text/
├── preprocessor.rs
├── languages/
│   ├── mod.rs           # Language trait + registry
│   ├── chinese.rs       # Mandarin (current)
│   ├── english.rs       # English with CMU dict
│   ├── french.rs        # French
│   ├── japanese.rs      # Japanese
│   └── korean.rs        # Korean
```

### Step 1: Define Language Trait

```rust
// src/text/languages/mod.rs

pub trait Language {
    /// ISO 639-1 code
    fn code(&self) -> &'static str;

    /// Convert text to phonemes
    fn text_to_phonemes(&self, text: &str) -> (Vec<String>, Vec<i32>);

    /// Get language-specific symbols
    fn symbols(&self) -> &[&str];

    /// Normalize text (remove accents, etc.)
    fn normalize(&self, text: &str) -> String;

    /// Check if character belongs to this language
    fn is_char(&self, c: char) -> bool;
}
```

### Step 2: Implement French

```rust
// src/text/languages/french.rs

use super::Language;

pub struct French;

/// French phoneme inventory (IPA-based)
const FRENCH_PHONEMES: &[&str] = &[
    // Oral vowels
    "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə",
    // Nasal vowels
    "ɛ̃", "ɑ̃", "ɔ̃", "œ̃",
    // Consonants
    "p", "b", "t", "d", "k", "g",
    "f", "v", "s", "z", "ʃ", "ʒ",
    "m", "n", "ɲ", "ŋ",
    "l", "ʁ",  // French 'r' is uvular
    "w", "j", "ɥ",
];

impl Language for French {
    fn code(&self) -> &'static str { "fr" }

    fn text_to_phonemes(&self, text: &str) -> (Vec<String>, Vec<i32>) {
        let mut phonemes = Vec::new();
        let mut word2ph = Vec::new();

        for word in text.split_whitespace() {
            let word_phonemes = french_g2p(word);
            word2ph.push(word_phonemes.len() as i32);
            phonemes.extend(word_phonemes);
        }

        (phonemes, word2ph)
    }

    fn symbols(&self) -> &[&str] {
        FRENCH_PHONEMES
    }

    fn normalize(&self, text: &str) -> String {
        // Keep accents - they affect pronunciation
        // é, è, ê, ë all sound different
        text.to_lowercase()
    }

    fn is_char(&self, c: char) -> bool {
        c.is_ascii_alphabetic() ||
        matches!(c, 'é' | 'è' | 'ê' | 'ë' | 'à' | 'â' | 'ù' | 'û' | 'ô' | 'î' | 'ï' | 'ç' | 'œ' | 'æ')
    }
}

/// French G2P rules
fn french_g2p(word: &str) -> Vec<String> {
    // French has complex orthography-to-phoneme rules
    // Examples:
    // - "eau" → /o/
    // - "oi" → /wa/
    // - "ch" → /ʃ/
    // - "gn" → /ɲ/
    // - silent final consonants except C, R, F, L ("careful")

    let mut phonemes = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check multi-character patterns first
        let remaining = &word[i..];

        if remaining.starts_with("eau") {
            phonemes.push("o".into());
            i += 3;
        } else if remaining.starts_with("ai") || remaining.starts_with("ei") {
            phonemes.push("ɛ".into());
            i += 2;
        } else if remaining.starts_with("oi") {
            phonemes.push("w".into());
            phonemes.push("a".into());
            i += 2;
        } else if remaining.starts_with("ou") {
            phonemes.push("u".into());
            i += 2;
        } else if remaining.starts_with("ch") {
            phonemes.push("ʃ".into());
            i += 2;
        } else if remaining.starts_with("gn") {
            phonemes.push("ɲ".into());
            i += 2;
        } else if remaining.starts_with("qu") {
            phonemes.push("k".into());
            i += 2;
        } else {
            // Single character
            let ph = match chars[i] {
                'a' | 'à' | 'â' => "a",
                'e' => "ə",
                'é' => "e",
                'è' | 'ê' | 'ë' => "ɛ",
                'i' | 'î' | 'ï' => "i",
                'o' | 'ô' => "o",
                'u' | 'û' => "y",
                'ù' => "u",
                'c' => if i + 1 < chars.len() && matches!(chars[i+1], 'e' | 'i') { "s" } else { "k" },
                'ç' => "s",
                'g' => if i + 1 < chars.len() && matches!(chars[i+1], 'e' | 'i') { "ʒ" } else { "g" },
                'j' => "ʒ",
                'r' => "ʁ",
                'y' => "i",
                c if c.is_ascii_alphabetic() => {
                    // Return character as-is for standard consonants
                    phonemes.push(c.to_lowercase().to_string());
                    i += 1;
                    continue;
                }
                _ => {
                    i += 1;
                    continue;
                }
            };
            phonemes.push(ph.into());
            i += 1;
        }
    }

    phonemes
}
```

### Step 3: Create French Dictionary (Optional)

For better accuracy, use a pronunciation dictionary:

```rust
// src/text/languages/french_dict.rs

use std::collections::HashMap;
use std::sync::LazyLock;

static FRENCH_DICT: LazyLock<HashMap<&'static str, &'static [&'static str]>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    // Common words with irregular pronunciations
    m.insert("monsieur", &["m", "ə", "s", "j", "ø"][..]);
    m.insert("femme", &["f", "a", "m"][..]);  // Not "fɛm"
    m.insert("oignon", &["ɔ", "ɲ", "ɔ̃"][..]);
    m.insert("fils", &["f", "i", "s"][..]);  // Final 's' pronounced
    m.insert("sept", &["s", "ɛ", "t"][..]);
    m.insert("dix", &["d", "i", "s"][..]);
    // ... more exceptions

    m
});

pub fn lookup(word: &str) -> Option<Vec<String>> {
    FRENCH_DICT.get(word.to_lowercase().as_str())
        .map(|ph| ph.iter().map(|s| s.to_string()).collect())
}
```

### Step 4: Multilingual Preprocessor

```rust
// src/text/preprocessor.rs

pub fn detect_language_multilingual(text: &str) -> Vec<LangSegment> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut current_lang: Option<&str> = None;

    for c in text.chars() {
        let lang = if is_chinese_char(c) {
            "zh"
        } else if is_french_char(c) {
            "fr"
        } else if is_japanese_char(c) {
            "ja"
        } else if c.is_ascii_alphabetic() {
            "en"  // Default to English for ASCII
        } else {
            current_lang.unwrap_or("en")  // Keep current for punctuation
        };

        if Some(lang) != current_lang && !current.is_empty() {
            segments.push(LangSegment {
                text: current.clone(),
                lang: current_lang.unwrap().into()
            });
            current.clear();
        }
        current.push(c);
        current_lang = Some(lang);
    }

    if !current.is_empty() {
        segments.push(LangSegment {
            text: current,
            lang: current_lang.unwrap().into()
        });
    }

    segments
}
```

### Step 5: Model Requirements

To support a new language:

1. **Phoneme symbols**: Add language-specific phonemes to symbol table
2. **T2S model**: Train on target language data (or multilingual)
3. **VITS vocoder**: Train on target language audio
4. **BERT**: Use multilingual BERT (XLM-R) or language-specific BERT

### Example: Adding Japanese

```rust
// src/text/languages/japanese.rs

use super::Language;

pub struct Japanese;

// Japanese phonemes (mora-based)
const JAPANESE_PHONEMES: &[&str] = &[
    // Vowels
    "a", "i", "u", "e", "o",
    // Consonants + vowel combinations are usually treated as units
    "ka", "ki", "ku", "ke", "ko",
    "sa", "si", "su", "se", "so",
    // ... all kana
    // Special
    "N",  // Syllabic n (ん)
    "Q",  // Gemination (っ)
    "pau",  // Pause
];

impl Language for Japanese {
    fn code(&self) -> &'static str { "ja" }

    fn text_to_phonemes(&self, text: &str) -> (Vec<String>, Vec<i32>) {
        // Japanese uses mora-based phonemes
        // Convert kanji → hiragana → romaji → phonemes

        // For kanji: use MeCab or similar morphological analyzer
        // For this example, assume pre-converted hiragana

        let mut phonemes = Vec::new();
        let mut word2ph = Vec::new();

        for c in text.chars() {
            if let Some(mora) = hiragana_to_mora(c) {
                phonemes.push(mora);
                word2ph.push(1);
            }
        }

        (phonemes, word2ph)
    }

    fn is_char(&self, c: char) -> bool {
        // Hiragana: U+3040-U+309F
        // Katakana: U+30A0-U+30FF
        // Kanji: Same as Chinese
        let code = c as u32;
        (0x3040..=0x309F).contains(&code) ||  // Hiragana
        (0x30A0..=0x30FF).contains(&code) ||  // Katakana
        (0x4E00..=0x9FFF).contains(&code)     // Kanji
    }

    // ...
}

fn hiragana_to_mora(c: char) -> Option<String> {
    match c {
        'あ' => Some("a".into()),
        'い' => Some("i".into()),
        'う' => Some("u".into()),
        'え' => Some("e".into()),
        'お' => Some("o".into()),
        'か' => Some("ka".into()),
        'き' => Some("ki".into()),
        // ... all hiragana
        _ => None
    }
}
```

## File Reference

| File | Purpose |
|------|---------|
| `src/voice_clone.rs` | Main API: VoiceCloner, zero-shot/few-shot synthesis |
| `src/text/preprocessor.rs` | Text → phonemes, language detection, G2P |
| `src/text/symbols.rs` | Phoneme vocabulary (322 symbols) |
| `src/text/cmudict.rs` | English CMU dictionary |
| `src/text/bert_features.rs` | BERT feature extraction |
| `src/models/bert.rs` | Chinese BERT model |
| `src/models/t2s.rs` | Text-to-Semantic GPT model |
| `src/models/vits.rs` | VITS vocoder |
| `src/models/hubert.rs` | HuBERT for semantic extraction |
| `src/audio.rs` | Mel spectrogram, WAV I/O |

## How to Implement SSML with 语气 (Emotion) Annotations

> **STATUS: PLANNED** - This section describes the proposed architecture for SSML support.
> Not yet implemented. See implementation plan below.

SSML (Speech Synthesis Markup Language) provides fine-grained control over speech synthesis. This section describes how to implement SSML support with Chinese 语气 (tone/emotion) annotations.

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| SSML Parser | ❌ Planned | Parse XML tags |
| Prosody (rate) | ⚠️ Partial | Use VITS `speed` parameter |
| Prosody (pitch) | ❌ Planned | Requires DSP post-processing |
| Prosody (volume) | ❌ Planned | Simple amplitude scaling |
| Emotion mapping | ❌ Planned | Map 语气 to noise_scale/speed |
| Phoneme override | ❌ Planned | Bypass G2P for marked text |
| Break/pause | ❌ Planned | Insert silence samples |

### Implementation Approach

**Phase 1 (Quick Win)**: Use existing VITS parameters
- Map emotions to `noise_scale` (0.0-1.0) and `speed` (0.5-2.0)
- No model changes required

**Phase 2 (DSP)**: Add post-processing
- Time stretching (WSOLA algorithm) for rate
- Pitch shifting via resample + time stretch
- Volume scaling with soft clipping

**Phase 3 (Model-level)**: Train with conditioning
- Add emotion embeddings to T2S model
- Requires training data with emotion labels

### SSML Overview

```xml
<speak>
  你好，<prosody rate="slow" pitch="+10%">欢迎来到</prosody>我们的系统。
  <break time="500ms"/>
  <emotion type="happy">今天天气真不错！</emotion>
  <phoneme ph="sh ang1">熵</phoneme>是一个物理概念。
</speak>
```

### Architecture for SSML Processing

```
src/text/
├── ssml/
│   ├── mod.rs           # SSML parser and types
│   ├── parser.rs        # XML parsing
│   ├── prosody.rs       # Prosody control (rate, pitch, volume)
│   ├── emotion.rs       # 语气/emotion handling
│   └── phoneme.rs       # Phoneme overrides
```

### Step 1: Define SSML Types

```rust
// src/text/ssml/mod.rs

pub mod parser;
pub mod prosody;
pub mod emotion;

/// SSML document containing speech segments
#[derive(Debug, Clone)]
pub struct SsmlDocument {
    pub segments: Vec<SsmlSegment>,
}

/// A segment of SSML-annotated text
#[derive(Debug, Clone)]
pub struct SsmlSegment {
    pub text: String,
    pub prosody: Option<Prosody>,
    pub emotion: Option<Emotion>,
    pub phoneme_override: Option<Vec<String>>,
    pub break_ms: Option<u32>,
}

/// Prosody control parameters
#[derive(Debug, Clone, Default)]
pub struct Prosody {
    /// Speaking rate: 0.5 = half speed, 2.0 = double speed
    pub rate: Option<f32>,
    /// Pitch shift in semitones: -12 to +12
    pub pitch: Option<f32>,
    /// Volume: 0.0 to 2.0 (1.0 = normal)
    pub volume: Option<f32>,
}

/// 语气 (Emotion/Tone) types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    // Basic emotions
    Neutral,        // 平静
    Happy,          // 开心
    Sad,            // 悲伤
    Angry,          // 愤怒
    Fearful,        // 恐惧
    Surprised,      // 惊讶
    Disgusted,      // 厌恶

    // Chinese-specific 语气
    Excited,        // 兴奋
    Gentle,         // 温柔
    Serious,        // 严肃
    Playful,        // 俏皮
    Encouraging,    // 鼓励
    Sympathetic,    // 同情
    Sarcastic,      // 讽刺
    Whisper,        // 耳语
    Shouting,       // 喊叫

    // Business/Professional tones
    Professional,   // 专业
    Friendly,       // 友好
    Apologetic,     // 抱歉
    Thankful,       // 感谢
}

impl Emotion {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            // English
            "neutral" => Some(Self::Neutral),
            "happy" | "joy" => Some(Self::Happy),
            "sad" | "sadness" => Some(Self::Sad),
            "angry" | "anger" => Some(Self::Angry),
            "fear" | "fearful" => Some(Self::Fearful),
            "surprise" | "surprised" => Some(Self::Surprised),
            "disgust" | "disgusted" => Some(Self::Disgusted),
            "excited" => Some(Self::Excited),
            "gentle" | "soft" => Some(Self::Gentle),
            "serious" => Some(Self::Serious),
            "playful" => Some(Self::Playful),
            "encouraging" => Some(Self::Encouraging),
            "sympathetic" => Some(Self::Sympathetic),
            "sarcastic" => Some(Self::Sarcastic),
            "whisper" => Some(Self::Whisper),
            "shout" | "shouting" => Some(Self::Shouting),
            "professional" => Some(Self::Professional),
            "friendly" => Some(Self::Friendly),
            "apologetic" | "sorry" => Some(Self::Apologetic),
            "thankful" | "grateful" => Some(Self::Thankful),

            // Chinese 语气
            "平静" => Some(Self::Neutral),
            "开心" | "高兴" | "快乐" => Some(Self::Happy),
            "悲伤" | "难过" | "伤心" => Some(Self::Sad),
            "愤怒" | "生气" => Some(Self::Angry),
            "恐惧" | "害怕" => Some(Self::Fearful),
            "惊讶" | "吃惊" => Some(Self::Surprised),
            "厌恶" | "恶心" => Some(Self::Disgusted),
            "兴奋" | "激动" => Some(Self::Excited),
            "温柔" | "柔和" => Some(Self::Gentle),
            "严肃" | "认真" => Some(Self::Serious),
            "俏皮" | "调皮" => Some(Self::Playful),
            "鼓励" | "激励" => Some(Self::Encouraging),
            "同情" | "怜悯" => Some(Self::Sympathetic),
            "讽刺" | "嘲讽" => Some(Self::Sarcastic),
            "耳语" | "低语" => Some(Self::Whisper),
            "喊叫" | "大喊" => Some(Self::Shouting),
            "专业" => Some(Self::Professional),
            "友好" | "亲切" => Some(Self::Friendly),
            "抱歉" | "道歉" => Some(Self::Apologetic),
            "感谢" | "感激" => Some(Self::Thankful),

            _ => None,
        }
    }

    /// Get prosody modifiers for this emotion
    pub fn to_prosody(&self) -> Prosody {
        match self {
            Self::Neutral => Prosody::default(),
            Self::Happy => Prosody {
                rate: Some(1.1),
                pitch: Some(2.0),
                volume: Some(1.1),
            },
            Self::Sad => Prosody {
                rate: Some(0.85),
                pitch: Some(-2.0),
                volume: Some(0.9),
            },
            Self::Angry => Prosody {
                rate: Some(1.2),
                pitch: Some(3.0),
                volume: Some(1.3),
            },
            Self::Fearful => Prosody {
                rate: Some(1.15),
                pitch: Some(4.0),
                volume: Some(0.85),
            },
            Self::Surprised => Prosody {
                rate: Some(1.1),
                pitch: Some(5.0),
                volume: Some(1.2),
            },
            Self::Excited => Prosody {
                rate: Some(1.25),
                pitch: Some(4.0),
                volume: Some(1.25),
            },
            Self::Gentle => Prosody {
                rate: Some(0.9),
                pitch: Some(-1.0),
                volume: Some(0.8),
            },
            Self::Serious => Prosody {
                rate: Some(0.95),
                pitch: Some(-1.5),
                volume: Some(1.0),
            },
            Self::Whisper => Prosody {
                rate: Some(0.85),
                pitch: Some(-3.0),
                volume: Some(0.5),
            },
            Self::Shouting => Prosody {
                rate: Some(1.1),
                pitch: Some(2.0),
                volume: Some(1.5),
            },
            _ => Prosody::default(),
        }
    }
}
```

### Step 2: Implement SSML Parser

```rust
// src/text/ssml/parser.rs

use super::{SsmlDocument, SsmlSegment, Prosody, Emotion};

/// Parse SSML markup into structured document
pub fn parse_ssml(input: &str) -> Result<SsmlDocument, SsmlError> {
    // Check if input is SSML (starts with <speak>)
    let input = input.trim();
    if !input.starts_with("<speak>") {
        // Plain text - wrap in neutral segment
        return Ok(SsmlDocument {
            segments: vec![SsmlSegment {
                text: input.to_string(),
                prosody: None,
                emotion: None,
                phoneme_override: None,
                break_ms: None,
            }],
        });
    }

    let mut segments = Vec::new();
    let mut parser = SsmlParser::new(input);

    while let Some(segment) = parser.next_segment()? {
        segments.push(segment);
    }

    Ok(SsmlDocument { segments })
}

struct SsmlParser<'a> {
    input: &'a str,
    pos: usize,
    current_prosody: Option<Prosody>,
    current_emotion: Option<Emotion>,
}

impl<'a> SsmlParser<'a> {
    fn new(input: &'a str) -> Self {
        // Skip opening <speak> tag
        let start = input.find('>').map(|i| i + 1).unwrap_or(0);
        Self {
            input,
            pos: start,
            current_prosody: None,
            current_emotion: None,
        }
    }

    fn next_segment(&mut self) -> Result<Option<SsmlSegment>, SsmlError> {
        self.skip_whitespace();

        if self.pos >= self.input.len() || self.remaining().starts_with("</speak>") {
            return Ok(None);
        }

        // Check for tags
        if self.remaining().starts_with('<') {
            self.parse_tag()
        } else {
            // Plain text until next tag
            self.parse_text()
        }
    }

    fn parse_tag(&mut self) -> Result<Option<SsmlSegment>, SsmlError> {
        let remaining = self.remaining();

        // Break tag: <break time="500ms"/>
        if remaining.starts_with("<break") {
            let time_ms = self.parse_break_tag()?;
            return Ok(Some(SsmlSegment {
                text: String::new(),
                prosody: None,
                emotion: None,
                phoneme_override: None,
                break_ms: Some(time_ms),
            }));
        }

        // Prosody tag: <prosody rate="slow" pitch="+10%">
        if remaining.starts_with("<prosody") {
            self.current_prosody = Some(self.parse_prosody_tag()?);
            return self.next_segment();
        }
        if remaining.starts_with("</prosody>") {
            self.pos += "</prosody>".len();
            self.current_prosody = None;
            return self.next_segment();
        }

        // Emotion tag: <emotion type="happy"> or <语气 type="开心">
        if remaining.starts_with("<emotion") || remaining.starts_with("<语气") {
            self.current_emotion = Some(self.parse_emotion_tag()?);
            return self.next_segment();
        }
        if remaining.starts_with("</emotion>") || remaining.starts_with("</语气>") {
            let end_tag = if remaining.starts_with("</emotion>") {
                "</emotion>"
            } else {
                "</语气>"
            };
            self.pos += end_tag.len();
            self.current_emotion = None;
            return self.next_segment();
        }

        // Phoneme tag: <phoneme ph="sh ang1">熵</phoneme>
        if remaining.starts_with("<phoneme") {
            return self.parse_phoneme_tag();
        }

        // Unknown tag - skip
        if let Some(end) = remaining.find('>') {
            self.pos += end + 1;
        }
        self.next_segment()
    }

    fn parse_prosody_tag(&mut self) -> Result<Prosody, SsmlError> {
        let tag_end = self.remaining().find('>').ok_or(SsmlError::UnclosedTag)?;
        let tag_content = &self.remaining()[..tag_end];

        let mut prosody = Prosody::default();

        // Parse rate attribute
        if let Some(rate) = extract_attribute(tag_content, "rate") {
            prosody.rate = Some(parse_rate(&rate)?);
        }

        // Parse pitch attribute
        if let Some(pitch) = extract_attribute(tag_content, "pitch") {
            prosody.pitch = Some(parse_pitch(&pitch)?);
        }

        // Parse volume attribute
        if let Some(volume) = extract_attribute(tag_content, "volume") {
            prosody.volume = Some(parse_volume(&volume)?);
        }

        self.pos += tag_end + 1;
        Ok(prosody)
    }

    fn parse_emotion_tag(&mut self) -> Result<Emotion, SsmlError> {
        let tag_end = self.remaining().find('>').ok_or(SsmlError::UnclosedTag)?;
        let tag_content = &self.remaining()[..tag_end];

        let emotion_type = extract_attribute(tag_content, "type")
            .ok_or(SsmlError::MissingAttribute("type"))?;

        let emotion = Emotion::from_str(&emotion_type)
            .ok_or(SsmlError::UnknownEmotion(emotion_type))?;

        self.pos += tag_end + 1;
        Ok(emotion)
    }

    fn parse_phoneme_tag(&mut self) -> Result<Option<SsmlSegment>, SsmlError> {
        let tag_end = self.remaining().find('>').ok_or(SsmlError::UnclosedTag)?;
        let tag_content = &self.remaining()[..tag_end];

        // Extract phoneme override
        let phonemes = extract_attribute(tag_content, "ph")
            .ok_or(SsmlError::MissingAttribute("ph"))?;

        self.pos += tag_end + 1;

        // Get text content until </phoneme>
        let content_end = self.remaining().find("</phoneme>")
            .ok_or(SsmlError::UnclosedTag)?;
        let text = self.remaining()[..content_end].to_string();

        self.pos += content_end + "</phoneme>".len();

        Ok(Some(SsmlSegment {
            text,
            prosody: self.current_prosody.clone(),
            emotion: self.current_emotion,
            phoneme_override: Some(phonemes.split_whitespace().map(String::from).collect()),
            break_ms: None,
        }))
    }

    fn parse_break_tag(&mut self) -> Result<u32, SsmlError> {
        let tag_end = self.remaining().find("/>")
            .or_else(|| self.remaining().find('>'))
            .ok_or(SsmlError::UnclosedTag)?;
        let tag_content = &self.remaining()[..tag_end];

        let time_str = extract_attribute(tag_content, "time")
            .unwrap_or_else(|| "250ms".to_string());

        let time_ms = parse_time(&time_str)?;

        self.pos += tag_end + if self.remaining()[tag_end..].starts_with("/>") { 2 } else { 1 };
        Ok(time_ms)
    }

    fn parse_text(&mut self) -> Result<Option<SsmlSegment>, SsmlError> {
        let text_end = self.remaining().find('<').unwrap_or(self.remaining().len());
        let text = self.remaining()[..text_end].to_string();
        self.pos += text_end;

        if text.trim().is_empty() {
            return self.next_segment();
        }

        Ok(Some(SsmlSegment {
            text,
            prosody: self.current_prosody.clone(),
            emotion: self.current_emotion,
            phoneme_override: None,
            break_ms: None,
        }))
    }

    fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() &&
              self.input[self.pos..].starts_with(char::is_whitespace) {
            self.pos += 1;
        }
    }
}

// Helper functions
fn extract_attribute(tag: &str, name: &str) -> Option<String> {
    let pattern = format!("{}=\"", name);
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn parse_rate(s: &str) -> Result<f32, SsmlError> {
    match s {
        "x-slow" => Ok(0.5),
        "slow" => Ok(0.75),
        "medium" => Ok(1.0),
        "fast" => Ok(1.25),
        "x-fast" => Ok(1.5),
        _ => {
            // Parse percentage or decimal
            if s.ends_with('%') {
                let pct: f32 = s.trim_end_matches('%').parse()
                    .map_err(|_| SsmlError::InvalidValue)?;
                Ok(pct / 100.0)
            } else {
                s.parse().map_err(|_| SsmlError::InvalidValue)
            }
        }
    }
}

fn parse_pitch(s: &str) -> Result<f32, SsmlError> {
    match s {
        "x-low" => Ok(-6.0),
        "low" => Ok(-3.0),
        "medium" => Ok(0.0),
        "high" => Ok(3.0),
        "x-high" => Ok(6.0),
        _ => {
            // Parse semitones or percentage
            if s.ends_with('%') {
                let pct: f32 = s.trim_start_matches('+').trim_end_matches('%').parse()
                    .map_err(|_| SsmlError::InvalidValue)?;
                Ok(pct / 10.0)  // 10% ≈ 1 semitone
            } else if s.ends_with("st") {
                s.trim_end_matches("st").trim_start_matches('+').parse()
                    .map_err(|_| SsmlError::InvalidValue)
            } else {
                s.trim_start_matches('+').parse()
                    .map_err(|_| SsmlError::InvalidValue)
            }
        }
    }
}

fn parse_volume(s: &str) -> Result<f32, SsmlError> {
    match s {
        "silent" => Ok(0.0),
        "x-soft" => Ok(0.25),
        "soft" => Ok(0.5),
        "medium" => Ok(1.0),
        "loud" => Ok(1.5),
        "x-loud" => Ok(2.0),
        _ => {
            if s.ends_with("dB") {
                let db: f32 = s.trim_end_matches("dB").trim_start_matches('+').parse()
                    .map_err(|_| SsmlError::InvalidValue)?;
                Ok(10_f32.powf(db / 20.0))  // dB to linear
            } else {
                s.parse().map_err(|_| SsmlError::InvalidValue)
            }
        }
    }
}

fn parse_time(s: &str) -> Result<u32, SsmlError> {
    if s.ends_with("ms") {
        s.trim_end_matches("ms").parse().map_err(|_| SsmlError::InvalidValue)
    } else if s.ends_with('s') {
        let secs: f32 = s.trim_end_matches('s').parse()
            .map_err(|_| SsmlError::InvalidValue)?;
        Ok((secs * 1000.0) as u32)
    } else {
        s.parse().map_err(|_| SsmlError::InvalidValue)
    }
}

#[derive(Debug)]
pub enum SsmlError {
    UnclosedTag,
    MissingAttribute(&'static str),
    UnknownEmotion(String),
    InvalidValue,
}
```

### Step 3: Integrate SSML with Synthesis

```rust
// src/voice_clone.rs

use crate::text::ssml::{parse_ssml, SsmlDocument, SsmlSegment, Prosody};

impl VoiceCloner {
    /// Synthesize speech from SSML markup
    pub fn synthesize_ssml(&mut self, ssml: &str) -> Result<AudioOutput, Error> {
        let doc = parse_ssml(ssml)
            .map_err(|e| Error::Message(format!("SSML parse error: {:?}", e)))?;

        let mut all_samples = Vec::new();
        let mut total_tokens = 0;

        for segment in doc.segments {
            if let Some(break_ms) = segment.break_ms {
                // Insert silence
                let silence_samples = (self.config.sample_rate as f32 * break_ms as f32 / 1000.0) as usize;
                all_samples.extend(vec![0.0f32; silence_samples]);
                continue;
            }

            if segment.text.trim().is_empty() {
                continue;
            }

            // Synthesize segment
            let mut audio = self.synthesize_segment(&segment)?;
            total_tokens += audio.num_tokens;

            // Apply prosody modifications
            if let Some(ref prosody) = segment.prosody {
                apply_prosody(&mut audio.samples, prosody, self.config.sample_rate);
            }

            // Apply emotion-based prosody
            if let Some(emotion) = segment.emotion {
                let emotion_prosody = emotion.to_prosody();
                apply_prosody(&mut audio.samples, &emotion_prosody, self.config.sample_rate);
            }

            all_samples.extend(audio.samples);
        }

        let duration = all_samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples: all_samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: total_tokens,
        })
    }

    fn synthesize_segment(&mut self, segment: &SsmlSegment) -> Result<AudioOutput, Error> {
        // If phoneme override is specified, use it directly
        if let Some(ref phonemes) = segment.phoneme_override {
            return self.synthesize_with_phonemes(&segment.text, phonemes);
        }

        // Regular synthesis
        self.synthesize(&segment.text)
    }

    fn synthesize_with_phonemes(&mut self, text: &str, phonemes: &[String]) -> Result<AudioOutput, Error> {
        // Convert phonemes to IDs
        let phoneme_ids: Vec<i32> = phonemes
            .iter()
            .map(|p| crate::text::symbol_to_id(p))
            .collect();

        // Create word2ph (1 phoneme per "word" for manual override)
        let word2ph: Vec<i32> = vec![phonemes.len() as i32];

        // Extract BERT features (use zeros for manual phonemes)
        let bert_features = Array::zeros::<f32>(&[1, phonemes.len() as i32, 1024])
            .map_err(|e| Error::Message(e.to_string()))?;

        // ... rest of synthesis pipeline
        todo!("Complete synthesis with manual phonemes")
    }
}

/// Apply prosody modifications to audio samples
fn apply_prosody(samples: &mut Vec<f32>, prosody: &Prosody, sample_rate: u32) {
    // Apply volume
    if let Some(volume) = prosody.volume {
        for sample in samples.iter_mut() {
            *sample *= volume;
        }
    }

    // Apply rate (time stretching)
    if let Some(rate) = prosody.rate {
        if (rate - 1.0).abs() > 0.01 {
            *samples = time_stretch(samples, rate, sample_rate);
        }
    }

    // Apply pitch shift
    if let Some(pitch) = prosody.pitch {
        if pitch.abs() > 0.1 {
            *samples = pitch_shift(samples, pitch, sample_rate);
        }
    }
}

/// Simple time stretching using linear interpolation
fn time_stretch(samples: &[f32], rate: f32, _sample_rate: u32) -> Vec<f32> {
    let new_len = (samples.len() as f32 / rate) as usize;
    let mut result = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f32 * rate;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f32;

        if src_idx + 1 < samples.len() {
            let sample = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
            result.push(sample);
        } else if src_idx < samples.len() {
            result.push(samples[src_idx]);
        }
    }

    result
}

/// Simple pitch shifting using resampling
fn pitch_shift(samples: &[f32], semitones: f32, sample_rate: u32) -> Vec<f32> {
    // Pitch shift ratio: 2^(semitones/12)
    let ratio = 2_f32.powf(semitones / 12.0);

    // Resample to change pitch, then time-stretch to restore duration
    let resampled = resample(samples, ratio);
    time_stretch(&resampled, ratio, sample_rate)
}

fn resample(samples: &[f32], ratio: f32) -> Vec<f32> {
    let new_len = (samples.len() as f32 * ratio) as usize;
    let mut result = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f32 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f32;

        if src_idx + 1 < samples.len() {
            let sample = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
            result.push(sample);
        } else if src_idx < samples.len() {
            result.push(samples[src_idx]);
        }
    }

    result
}
```

### Step 4: Usage Examples

```rust
use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cloner = VoiceCloner::new(VoiceClonerConfig::default())?;
    cloner.set_reference_audio("ref.wav")?;

    // Example 1: Basic prosody control
    let ssml = r#"
        <speak>
            你好，<prosody rate="slow">欢迎来到</prosody>我们的系统。
        </speak>
    "#;
    let audio = cloner.synthesize_ssml(ssml)?;

    // Example 2: Emotion/语气 control
    let ssml = r#"
        <speak>
            <emotion type="happy">今天天气真不错！</emotion>
            <break time="500ms"/>
            <emotion type="sad">但是明天要下雨。</emotion>
        </speak>
    "#;
    let audio = cloner.synthesize_ssml(ssml)?;

    // Example 3: Chinese 语气 tags
    let ssml = r#"
        <speak>
            <语气 type="兴奋">我们赢了！</语气>
            <break time="300ms"/>
            <语气 type="温柔">别担心，一切都会好的。</语气>
        </speak>
    "#;
    let audio = cloner.synthesize_ssml(ssml)?;

    // Example 4: Phoneme override for correct pronunciation
    let ssml = r#"
        <speak>
            <phoneme ph="sh ang1">熵</phoneme>是热力学中的重要概念。
        </speak>
    "#;
    let audio = cloner.synthesize_ssml(ssml)?;

    // Example 5: Combined controls
    let ssml = r#"
        <speak>
            <prosody rate="1.1" pitch="+2st">
                <emotion type="excited">
                    这个消息太棒了！
                </emotion>
            </prosody>
            <break time="500ms"/>
            <prosody rate="0.9" volume="soft">
                <emotion type="gentle">
                    让我慢慢告诉你details。
                </emotion>
            </prosody>
        </speak>
    "#;
    let audio = cloner.synthesize_ssml(ssml)?;

    cloner.save_wav(&audio, "output.wav")?;
    Ok(())
}
```

### Step 5: Advanced Emotion Implementation with Model Support

For better emotion rendering, train emotion-specific model components:

```rust
// src/models/emotion_embeddings.rs

/// Emotion embedding layer for conditioning T2S model
pub struct EmotionEmbedding {
    embeddings: Array,  // [num_emotions, hidden_dim]
}

impl EmotionEmbedding {
    pub fn new(num_emotions: usize, hidden_dim: usize) -> Self {
        // Initialize from trained weights
        todo!()
    }

    pub fn get_embedding(&self, emotion: Emotion) -> Array {
        let idx = emotion as i32;
        self.embeddings.index((idx, ..))
    }
}

// Modify T2S model to accept emotion conditioning
pub struct T2SInput<'a> {
    pub phoneme_ids: &'a Array,
    pub semantic_ids: &'a Array,
    pub bert_features: &'a Array,
    pub emotion_embedding: Option<&'a Array>,  // NEW
    pub cache: &'a mut Vec<Option<ConcatKeyValueCache>>,
}
```

### Supported SSML Tags Summary

| Tag | Attributes | Example |
|-----|------------|---------|
| `<speak>` | - | Root element |
| `<prosody>` | `rate`, `pitch`, `volume` | `<prosody rate="slow">` |
| `<break>` | `time` | `<break time="500ms"/>` |
| `<emotion>` | `type` | `<emotion type="happy">` |
| `<语气>` | `type` | `<语气 type="开心">` |
| `<phoneme>` | `ph` | `<phoneme ph="sh ang1">` |

### Supported 语气 Types

| English | 中文 | Prosody Effect |
|---------|------|----------------|
| happy | 开心/高兴 | +rate, +pitch, +volume |
| sad | 悲伤/难过 | -rate, -pitch, -volume |
| angry | 愤怒/生气 | +rate, +pitch, ++volume |
| excited | 兴奋/激动 | ++rate, +pitch, +volume |
| gentle | 温柔/柔和 | -rate, -pitch, -volume |
| serious | 严肃/认真 | -rate, -pitch |
| whisper | 耳语/低语 | -rate, -pitch, --volume |
| shouting | 喊叫/大喊 | +rate, +pitch, ++volume |

## API Quick Reference

```rust
use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};

// Zero-shot
let mut cloner = VoiceCloner::new(VoiceClonerConfig::default())?;
cloner.set_reference_audio("ref.wav")?;
let audio = cloner.synthesize("你好世界")?;
cloner.save_wav(&audio, "output.wav")?;

// Few-shot (better quality)
cloner.set_reference_with_precomputed_codes(
    "ref.wav",
    "参考文本",
    "prompt_semantic.bin"
)?;
let audio = cloner.synthesize("你好世界")?;

// SSML with emotion
let ssml = r#"<speak><emotion type="happy">你好世界！</emotion></speak>"#;
let audio = cloner.synthesize_ssml(ssml)?;
```
