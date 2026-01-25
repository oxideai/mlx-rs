---
name: mlx-tts
description: GPT-SoVITS TTS pipeline implementation details. Use when working on text-to-speech, audio generation, or vocoding.
allowed-tools: Read, Grep, Glob, Bash(cargo:*, afplay:*)
---

# GPT-SoVITS TTS Pipeline

## Quick Start - Voice Clone API

```rust
use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};

// Create voice cloner
let mut cloner = VoiceCloner::with_defaults()?;

// Set reference voice
cloner.set_reference_audio("/path/to/reference.wav")?;

// Synthesize speech
let audio = cloner.synthesize("你好，世界！")?;

// Save or play
cloner.save_wav(&audio, "/tmp/output.wav")?;
cloner.play(&audio)?;  // macOS only
```

## CLI Usage

```bash
# Basic synthesis with default voice (doubao)
cargo run --example voice_clone --release -- "你好，世界！"

# Use specific voice (doubao, luoxiang)
cargo run --example voice_clone --release -- --voice doubao "你好，世界！"
cargo run --example voice_clone --release -- --voice luoxiang "你好，世界！"

# Custom reference audio
cargo run --example voice_clone --release -- "你好" --ref /path/to/voice.wav

# Save to file
cargo run --example voice_clone --release -- "你好" --output /tmp/output.wav

# Interactive mode
cargo run --example voice_clone --release -- --interactive
```

### Available Voices

| Voice | Reference Audio | Reference Text |
|-------|-----------------|----------------|
| `doubao` | `doubao_ref_mix_new.wav` | "这家resturant的steak很有名，但是vegetable salad的price有点贵" |
| `luoxiang` | `luoxiang_ref.wav` | "复杂的问题背后也许没有统一的答案，选择站在正方还是反方，其实取决于你对一系列价值判断的回答。" |

## Pipeline Overview

```
Text → Preprocessing → Phonemes (194 for 101 chars)
         ↓
Text → BERT → Features [1, seq, 1024]
         ↓
     T2S Model → Semantic Tokens (~550 at 25Hz)
         ↓
Reference → MelStyleEncoder → Style [1, 512, 1]
         ↓
     VITS Vocoder → Audio [1, 1, samples]
```

## Text Preprocessing

**Location**: `src/text/preprocessor.rs`

```rust
use mlx_rs_lm::inference::preprocess_text;

let (phoneme_ids, phonemes) = preprocess_text("你好");
// phonemes: ["n", "i3", "h", "ao3", "!"]
// IDs use 322-symbol vocabulary from symbols.rs
```

**Chinese G2P**:
- Character → Pinyin with tone (pinyin crate)
- Split initial + final: "ni3" → "n" + "i3"
- Zero-initial vowels use AA/EE/OO markers

## BERT Feature Extraction

**Location**: `src/text/bert_features.rs`

```rust
let mut bert = BertFeatureExtractor::new(tokenizer_path, model_path, -3)?;

// word2ph: phonemes per character (2 for Chinese, 1 for punctuation)
let word2ph = vec![2, 2, 1];  // "你好，"
let features = bert.extract_features(text, &word2ph)?;
// Output: [1, sum(word2ph), 1024]
```

## T2S Generation

**Location**: `src/models/t2s.rs`, `examples/tts_vits.rs`

### Generation Parameters
```rust
let top_k = 5;
let temperature = 0.8;
let target_tokens = (phoneme_count as f32 * 2.6) as usize;  // ~2.6 tok/phone
let max_tokens = phoneme_count * 4;
let min_tokens = phoneme_count * 2;
let eos_token = 1024;
```

### Top-k Sampling
```rust
fn sample_top_k(logits: &Array, top_k: i32, temperature: f32) -> Result<i32, _> {
    let scaled = logits.divide(array!(temperature))?;
    let probs = softmax_axis(&scaled, -1, None)?;

    // Get top-k, renormalize, sample
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Cumulative sampling from top-k
    let r: f32 = random::uniform(0.0, 1.0)?;
    // ...
}
```

### Repetition Detection
```rust
fn detect_repetition(tokens: &[i32], n: usize, min_count: usize) -> bool {
    if tokens.len() < n * 2 { return false; }
    let last_n = &tokens[tokens.len() - n..];
    tokens.windows(n).filter(|w| *w == last_n).count() >= min_count
}

// Stop if 3-gram repeats 8+ times
if detect_repetition(&all_tokens, 3, 8) {
    break;
}
```

## VITS Vocoding

**Location**: `src/models/vits.rs`

```rust
let audio = vits.decode(
    &codes,           // [1, 1, seq] semantic codes
    &phoneme_ids,     // [1, phone_seq] phoneme IDs
    Some(&ref_mel),   // [1, 704, time] reference mel
    0.5,              // noise_scale
    1.0,              // speed
)?;
```

### Key Steps
1. **Style extraction**: `ref_enc.forward(&ref_mel)` → [1, 512, 1]
2. **Quantizer decode**: codes → [1, 768, seq]
3. **Upsample 25Hz→50Hz**: repeat each position 2x
4. **TextEncoder**: combine SSL + text features
5. **Flow reverse**: z_p → z
6. **HiFiGAN decode**: z → audio

## Audio Output

**Location**: `src/audio.rs`

```rust
// Save audio to WAV
let samples: Vec<f32> = audio.as_slice().to_vec();
// Write 32kHz mono WAV file
save_wav(&samples, 32000, output_path)?;
```

## Running TTS

```bash
# Basic usage
cargo run --example tts_vits --release -- "你的文本" /tmp/output.wav

# Play result
afplay /tmp/output.wav

# Debug EOS detection
cargo run --example debug_tts_eos --release -- "你的文本"
```

## Expected Performance

| Text Length | Phonemes | Tokens | Audio Duration | Generation Time |
|-------------|----------|--------|----------------|-----------------|
| 23 chars    | 45       | ~138   | ~5.5s          | ~500ms          |
| 91 chars    | ~180     | ~493   | ~19.7s         | ~4900ms         |
| 101 chars   | 194      | ~550   | ~22s           | ~5500ms         |

## Performance Comparison: Rust vs Python (Jan 2025)

Benchmark: 91 Chinese characters with decimals and percentages, doubao voice, few-shot mode.

**Test text**: `从季节上看，主要是增在秋粮，2025年秋粮增产163.6亿斤，占全年粮食增量九成多。从区域上看，主要增在东北三省、内蒙古和新疆，这5个省粮食增产114.7亿斤，占全国增量接近70%。`

### Benchmark Results (5 runs each)

| Run | Rust MLX (ms) | Python MPS (ms) |
|-----|---------------|-----------------|
| 1   | 4,936         | 17,009 (warmup) |
| 2   | 4,886         | 9,537           |
| 3   | 4,889         | 9,430           |
| 4   | 4,953         | 9,608           |
| 5   | 4,861         | 9,665           |
| **Avg** | **4,905** | **9,560** (excl. warmup) |

### Performance Summary

| Metric | Rust (MLX) | Python (MPS) | Speedup |
|--------|------------|--------------|---------|
| Model load | 49ms | 3,895ms | **79x** |
| Synthesis (avg) | 4,905ms | 9,560ms | **1.95x** |
| Audio duration | 19.72s | 22.27s | similar |
| RTF (realtime factor) | **4.02x** | 2.33x | 1.7x |
| Consistency | ±1.9% | ±2.5% | similar |

**Key findings:**
- Rust is **~2x faster** than Python for synthesis
- Rust generates 19.72s audio in 4.9s (**4x realtime**)
- Python generates 22.27s audio in 9.6s (**2.3x realtime**)
- Model loading is **79x faster** in Rust (49ms vs 3.9s)

### Benchmark Commands

```bash
# Rust benchmark
TEXT='从季节上看，主要是增在秋粮，2025年秋粮增产163.6亿斤，占全年粮食增量九成多。从区域上看，主要增在东北三省、内蒙古和新疆，这5个省粮食增产114.7亿斤，占全国增量接近70%。'
for i in 1 2 3 4 5; do
  cargo run --release --example voice_clone -- --text "$TEXT" --voice doubao 2>&1 | grep "Generated.*tokens in"
done

# Python benchmark (use validation script)
cd ~/home/mofa-studio/models/setup-local-models/primespeech-validation
python test_tts_direct.py --voice doubao --device mps
```

## Python Reference Testing

For validating Rust TTS against Python dora-primespeech implementation:

```bash
# Use the reference test script
cd ~/home/mofa-studio/models/setup-local-models/primespeech-validation
python test_tts_direct.py --voice doubao --device mps
```

**Key reference data for `doubao` voice:**
- **Ref audio**: `~/.dora/models/primespeech/moyoyo/ref_audios/doubao_ref_mix_new.wav`
- **Ref text**: `这家resturant的steak很有名，但是vegetable salad的price有点贵`

## Few-Shot Mode with Python Codes (Best Quality)

**IMPORTANT**: Rust HuBERT extraction produces different codes than Python HuBERT.
For best few-shot quality, use Python pre-computed prompt_semantic codes:

```bash
# Step 1: Extract codes with Python (one-time per reference audio)
python3 -c "
import numpy as np
# Load from existing file or extract with Python HuBERT
codes = np.load('/tmp/gpt-sovits-mlx/doubao_mixed_prompt_semantic.npy').astype(np.int32)
codes.tofile('/tmp/python_prompt_semantic.bin')
"

# Step 2: Use codes in Rust
cargo run --release --example voice_clone -- \
  --text "你的文本" \
  --ref ~/.dora/models/primespeech/moyoyo/ref_audios/doubao_ref_mix_new.wav \
  --ref-text "这家resturant的steak很有名，但是vegetable salad的price有点贵" \
  --codes /tmp/python_prompt_semantic.bin
```

**Known Issues**:
1. Rust HuBERT extracts 137 tokens vs Python's 145 tokens for the same audio
2. Rust T2S generates different first tokens (28/47) vs Python (824) due to logit differences
3. This causes audio to have extra sounds or missing content

**Solution**: Use Python-extracted semantic tokens for best quality:

```bash
# Extract tokens with dora-primespeech Python, then use in Rust:
cargo run --release --example voice_clone -- \
  --tokens /tmp/python_new_tokens.bin \
  --text "你的文本"
```

**Token ratio**: ~2.5-2.9 tokens per phoneme is normal.

## Troubleshooting

### Audio too long / repetitive
- Check EOS detection (should be ~2.6 tok/phone)
- Verify top-k sampling is working
- Check repetition detection threshold

### Strange sounds at end
- Token count may exceed target
- Add early stopping at target * 1.2

### No audio output
- Verify weight files exist
- Check VITS decode shapes
- Ensure reference mel is [1, 704, time]

---

## Critical Fixes (Jan 2026)

### 1. Text Segmentation (Long Text Support)

**Problem**: Long text (>50 chars) causes beginning to be skipped and garbage at end.

**Root cause**: T2S attention cannot handle >100 phonemes. Python splits at ~50 chars.

**Fix in `voice_clone.rs`**:
```rust
// Split text at punctuation with max 50 chars per segment
let segments = split_text_at_punctuation_max_len(text, 50);
for segment in segments {
    // Process each segment independently
    let audio = self.synthesize_segment(&segment, ...)?;
    all_audio.extend(audio);
}
```

### 2. BERT Punctuation Feature Zeroing

**Problem**: Commas cause beginning of sentences to be skipped (e.g., "从季节上看，..." → "主要是增在秋粮").

**Root cause**: Comma's BERT features act as "attention anchors" that pull attention away from the beginning.

**Fix in `voice_clone.rs`**:
```rust
// Zero out BERT features for punctuation positions
let punct_phonemes = [",", ".", "!", "?", "-", "…"];
for (i, ph) in target_phonemes.iter().enumerate() {
    if punct_phonemes.contains(&ph.as_str()) {
        // Zero out BERT feature at position i
        let zeros = Array::zeros::<f32>(&[1, 1, 1024])?;
        // ... concatenate to replace position i
    }
}
```

### 3. EOS Detection Threshold

**Problem**: Model generates garbage at end because min_tokens forces it past natural EOS.

**Root cause**: min_tokens = 2.8 × phonemes was too high. Model wants to stop earlier.

**Fix in `voice_clone.rs`**:
```rust
// Lower threshold to allow natural EOS detection
let min_tokens = (phoneme_count as f32 * 2.3) as usize;  // was 2.8
```

### 4. Quote/Bracket Normalization

**Problem**: Chinese quotes `"..."` and brackets `(...)` cause word2ph mismatch errors.

**Root cause**: G2P doesn't handle quotes, but they exist in text passed to BERT.

**Fix in `preprocessor.rs`**:
```rust
// Strip quotes and brackets - they don't affect pronunciation
let text: String = text.chars()
    .filter(|&c| !matches!(c, '"' | '\'' | '(' | ')' | '[' | ']' | ':' | ';'))
    .collect();
```

**Fix in `inference.rs`** - return normalized text:
```rust
pub fn preprocess_text(text: &str) -> (Array, Vec<String>, Vec<i32>, String) {
    // ... preprocessing ...
    (phoneme_ids, phonemes, word2ph, text_normalized)  // Added text_normalized
}
```

**Fix in `voice_clone.rs`** - use normalized text for BERT:
```rust
let (phoneme_ids, phonemes, word2ph, text_normalized) = preprocess_text(text);
let bert_features = self.extract_bert_features(&text_normalized, ...)?;
```

### 5. Number and Percentage Normalization

**Problem**: Numbers like "163.6" and "70%" not pronounced correctly.

**Fix in `preprocessor.rs`**:
```rust
// "163.6" → "一百六十三点六"
fn number_to_chinese_with_decimal(num_str: &str) -> String { ... }

// "70%" → "百分之七十"
fn replace_percentage(text: &str) -> String {
    let re = Regex::new(r"(-?)(\d+(?:\.\d+)?)%").unwrap();
    re.replace_all(text, |caps| {
        let prefix = if &caps[1] == "-" { "负" } else { "" };
        format!("{}百分之{}", prefix, number_to_chinese_with_decimal(&caps[2]))
    })
}
```

### 6. Boundary Punctuation Stripping

**Problem**: Trailing commas cause BERT feature misalignment, skipping beginning of audio (e.g., "从区域上看，" → "从区域上" missed).

**Root cause**: BERT zeroing at trailing comma position causes attention to skip initial content.

**Fix in `voice_clone.rs`**:
```rust
// Strip leading/trailing punctuation before processing
let text = text.trim_matches(|c: char| {
    matches!(c, ',' | '.' | '!' | '?' | '，' | '。' | '！' | '？' | '、' | '；' | '：' | ' ')
});
```

### 7. Context-Aware Number Segmentation

**Problem**: "126.4亿斤" was split between English ("126.4") and Chinese ("亿斤") segments.

**Root cause**: Digits were always attached to English context.

**Fix in `preprocessor.rs` - `segment_by_language()`**:
```rust
// Look ahead: if digits followed by CJK, treat as Chinese
let mut j = i + 1;
while j < len && (chars[j].is_ascii_digit() || chars[j] == '.') {
    j += 1;
}
let followed_by_chinese = j < len && is_chinese_char(chars[j]);
if followed_by_chinese {
    // Keep "126.4亿斤" together as Chinese segment
    current_is_english = Some(false);
}
```

### 8. English Number Pronunciation

**Problem**: English numbers like "2050" pronounced as "twenty fifty" instead of "two thousand fifty".

**Fix**: Use `num2en` crate (Rust equivalent of Python's `inflect`):
```toml
# Cargo.toml
num2en = "1.0.0"
```

```rust
// preprocessor.rs
use num2en;
let words = num2en::u64_to_words(2050);  // "two thousand fifty"
```

## Debugging Checklist

When TTS has issues, check in order:

1. **Segmentation**: Is text being split at ~50 chars? Check debug output for segment count.
2. **Normalization**: Are special chars (quotes, numbers, %) converted? Check "Normalized:" debug line.
3. **BERT alignment**: Does word2ph length match text char count? Error shows mismatch.
4. **EOS detection**: See "EOS detected at step X" vs "EOS detected but ignored". If ignored, min_tokens too high.
5. **Token generation**: Check "Generated X new tokens". Should be ~2.3-2.6 × phoneme count.

## Debug Command

```bash
# Full debug output
cargo run --release --example voice_clone -- \
  --text "从季节上看，主要是增在秋粮。" \
  --voice doubao \
  --play 2>&1 | tee /tmp/voice_log.txt

# Key lines to check:
# DEBUG: Split text into N segments
# DEBUG: Normalized: '...' -> '...'
# DEBUG: Zeroed BERT feature at position X for ','
# DEBUG: EOS detected at step X, breaking with Y new tokens
```
