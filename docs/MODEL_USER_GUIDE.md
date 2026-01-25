# OminiX-MLX Model User Guide

This guide covers all available models in the OminiX-MLX project with usage examples.

## Table of Contents

- [LLM Models](#llm-models)
  - [Mistral-7B](#mistral-7b)
  - [Mixtral-8x7B MoE](#mixtral-8x7b-moe)
  - [Qwen3](#qwen3)
  - [Qwen3 MoE](#qwen3-moe)
  - [GLM-4](#glm-4)
  - [GLM-4.5 MoE](#glm-45-moe)
- [Image Generation](#image-generation)
  - [FLUX.2 Klein](#flux2-klein)
  - [Qwen-Image](#qwen-image)
- [Speech & Audio](#speech--audio)
  - [FunASR Paraformer (ASR)](#funasr-paraformer-asr)
  - [GPT-SoVITS (Voice Cloning)](#gpt-sovits-voice-cloning)
- [Performance Tips](#performance-tips)

---

## LLM Models

### Mistral-7B

**Crate:** `mistral-mlx`

Mistral-7B instruction-following model for text generation.

#### Quick Start

```bash
# Generate text
cargo run --release -p mistral-mlx --example generate_mistral -- \
    --prompt "What is the capital of France?"

# Run benchmark
cargo run --release -p mistral-mlx --example benchmark_mistral
```

#### Recommended Model

```
mlx-community/Mistral-7B-Instruct-v0.2-4bit
```

#### Performance

- ~83 tok/s on Apple Silicon (4-bit quantized)

---

### Mixtral-8x7B MoE

**Crate:** `mixtral-mlx`

Mixtral Mixture-of-Experts model with 8 experts.

#### Quick Start

```bash
# Download model first (will be cached)
cargo run --release -p mixtral-mlx --example generate_mixtral -- \
    /path/to/Mixtral-8x7B-Instruct-v0.1-4bit \
    "Explain quantum computing in simple terms"
```

#### Usage

```rust
use mixtral_mlx::{load_model, load_tokenizer, Generate, KVCache};

let tokenizer = load_tokenizer(model_dir)?;
let mut model = load_model(model_dir)?;

let generator = Generate::<KVCache>::new(&mut model, &mut cache, temperature, &prompt_tokens);
for token in generator.take(max_tokens) {
    // Process tokens...
}
```

#### Recommended Model

```
mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
```

#### Performance

- ~45 tok/s on Apple Silicon (4-bit quantized)

---

### Qwen3

**Crate:** `qwen3-mlx`

Qwen3 models for text generation and chat.

#### Quick Start

```bash
# Text generation
cargo run --release -p qwen3-mlx --example generate_qwen3 -- \
    /path/to/Qwen3-4B-bf16 \
    "Hello, how are you?"

# Interactive chat
cargo run --release -p qwen3-mlx --example chat_qwen3 -- \
    /path/to/Qwen3-4B-bf16
```

#### Chat Format

Qwen3 uses the ChatML format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

#### EOS Tokens

- `151643` - `<|im_end|>`
- `151645` - `<|endoftext|>`

---

### Qwen3 MoE

**Crate:** `mlx-rs-lm`

Qwen3 Mixture-of-Experts models (e.g., Qwen3-30B-A3B with 128 experts, 8 active).

#### Quick Start

```bash
cargo run --release -p mlx-rs-lm --example qwen3_moe -- \
    --prompt "Explain the concept of machine learning" \
    --max-tokens 100
```

#### Full Options

```bash
cargo run --release -p mlx-rs-lm --example qwen3_moe -- \
    --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
    --prompt "Your question here" \
    --max-tokens 100 \
    --temperature 0.7 \
    --system "You are a helpful assistant" \
    --debug
```

#### Recommended Model

```
mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
```

#### Performance

- ~98 tok/s on Apple Silicon (4-bit quantized)
- Memory: ~17 GB

---

### GLM-4

**Crate:** `glm4-mlx`

GLM-4 models from Zhipu AI.

#### Quick Start

```bash
cargo run --release -p glm4-mlx --example generate_glm4 -- \
    /path/to/GLM-4-9B-Chat-4bit \
    "你好，请介绍一下自己。"
```

#### Usage

```rust
use glm4_mlx::{load_model, load_tokenizer, Generate, KVCache};

let tokenizer = load_tokenizer(model_dir)?;
let mut model = load_model(model_dir)?;

let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt_tokens);
```

---

### GLM-4.5 MoE

**Crate:** `glm4-moe-mlx`

GLM-4.5 Mixture-of-Experts model with 60 experts.

#### Quick Start

```bash
cargo run --release -p glm4-moe-mlx --example generate_glm4_moe -- \
    /path/to/GLM-4.5-Air-3bit \
    "请解释一下什么是人工智能"

# Run benchmark
cargo run --release -p glm4-moe-mlx --example benchmark_glm4_moe
```

#### Recommended Model

```
mlx-community/GLM-4.5-Air-3bit
```

#### Performance

- ~45 tok/s on Apple Silicon (3-bit quantized)
- Memory: ~47 GB

---

## Image Generation

### FLUX.2 Klein

**Crate:** `flux-klein-mlx`

FLUX.2-klein-4B image generation model with Qwen3 text encoder.

#### Quick Start

```bash
# Generate image
cargo run --release -p flux-klein-mlx --example generate_klein -- \
    "a beautiful sunset over the ocean"

# With INT8 quantization (lower memory)
cargo run --release -p flux-klein-mlx --example generate_klein -- \
    --quantize "a cat sitting on a windowsill"
```

#### Requirements

- HuggingFace token for model download
- ~13 GB VRAM

#### Model Architecture

- Qwen3-4B text encoder (36 layers)
- 5 double + 20 single transformer blocks
- 4 denoising steps

#### Output

- Saves to `output_klein.ppm` in current directory

---

### Qwen-Image

**Crate:** `qwen-image-mlx`

Qwen-Image model for text-to-image generation.

#### Quick Start

```bash
# First download the model
huggingface-cli download mlx-community/Qwen-Image-2512-4bit \
    --include 'transformer/*.safetensors' \
    --include 'vae/*.safetensors' \
    --include 'text_encoder/*' \
    --include 'tokenizer/*'

# Generate image
cargo run --release -p qwen-image-mlx --example generate_qwen_image -- \
    --prompt "a cat sitting on a couch" \
    --height 512 \
    --width 512 \
    --steps 20
```

#### Full Options

```bash
cargo run --release -p qwen-image-mlx --example generate_qwen_image -- \
    --prompt "your prompt here" \
    --output output.png \
    --height 512 \
    --width 512 \
    --steps 20 \
    --guidance 5.0 \
    --seed 42
```

#### Output

- Saves to `output_qwen.ppm` (RGB image)
- Also saves `output_qwen_latent.pgm` (latent visualization)

---

## Speech & Audio

### FunASR Paraformer (ASR)

**Crate:** `funasr-mlx`

Paraformer speech recognition model for automatic speech transcription.

#### Quick Start

```bash
cargo run --release -p funasr-mlx --example transcribe -- \
    audio.wav \
    /path/to/paraformer-model
```

#### Model Files Required

The model directory should contain:
- `paraformer.safetensors` - Model weights
- `am.mvn` - CMVN normalization
- `tokens.txt` - Vocabulary

#### Audio Requirements

- WAV format
- Will auto-resample to 16kHz if needed
- Mono channel

#### Example Output

```
=== Results ===
Text: 今天天气真好

Performance:
  Audio duration: 3.50s
  Inference time: 245 ms
  RTF: 0.0700x
  Speed: 14.3x real-time
```

---

### GPT-SoVITS (Voice Cloning)

**Crate:** `mlx-rs-lm`

GPT-SoVITS voice cloning and text-to-speech synthesis.

#### Quick Start

```bash
# Basic usage (zero-shot mode)
cargo run --release -p mlx-rs-lm --example voice_clone -- \
    "你好，世界！"

# With custom reference audio
cargo run --release -p mlx-rs-lm --example voice_clone -- \
    "你好，世界！" \
    --ref /path/to/reference.wav

# Save to file
cargo run --release -p mlx-rs-lm --example voice_clone -- \
    "你好，世界！" \
    --output output.wav

# Interactive mode
cargo run --release -p mlx-rs-lm --example voice_clone -- --interactive
```

#### Few-Shot Mode (Better Quality)

Few-shot mode uses reference text to improve voice cloning quality:

```bash
# With reference transcript
cargo run --release -p mlx-rs-lm --example voice_clone -- \
    "测试语音" \
    --ref voice.wav \
    --ref-text "这是参考音频的文本"

# With pre-computed semantic codes (best quality)
python scripts/extract_prompt_semantic.py voice.wav codes.bin
cargo run --release -p mlx-rs-lm --example voice_clone -- \
    "测试语音" \
    --ref voice.wav \
    --ref-text "参考文本" \
    --codes codes.bin
```

#### Interactive Commands

In interactive mode:
- `/ref <path>` - Change reference audio
- `/save <path>` - Save last audio to file
- `/quit` - Exit
- `<text>` - Synthesize and play text

#### API Usage

```rust
use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};

let config = VoiceClonerConfig::default();
let mut cloner = VoiceCloner::new(config)?;

// Set reference audio
cloner.set_reference_audio("/path/to/reference.wav")?;

// Synthesize
let audio = cloner.synthesize("你好，世界！")?;

// Play or save
cloner.play_blocking(&audio)?;
cloner.save_wav(&audio, "output.wav")?;
```

---

## Performance Tips

### 1. Use Pre-Quantized Models

Pre-quantized models from HuggingFace are significantly faster than on-the-fly quantization:

| Approach | Performance |
|----------|-------------|
| On-the-fly quantization | ~52 tok/s |
| Pre-quantized model | ~83 tok/s |

### 2. Async Pipelining

For best performance, use async pipelining in generation loops:

```rust
// Good - enables CPU/GPU overlap
for _ in 0..num_tokens {
    let logits = model.forward(input)?;
    let next_y = sample(&logits)?;
    async_eval([&next_y])?;     // Start GPU work
    let _ = y.item::<u32>();     // Sync previous token
    y = next_y;
}
```

### 3. Wired Memory Limit

For MoE models, set the wired memory limit for optimal GPU performance:

```rust
unsafe {
    let info = mlx_sys::mlx_metal_device_info();
    let max_size = info.max_recommended_working_set_size;
    mlx_sys::mlx_set_wired_limit(&mut old_limit, max_size);
}
```

### 4. Periodic Cache Clearing

For long generations, clear the MLX cache periodically:

```rust
if token_count % 256 == 0 {
    unsafe { mlx_sys::mlx_clear_cache(); }
}
```

---

## Model Performance Summary

| Model | Type | Performance | Memory |
|-------|------|-------------|--------|
| Mistral-7B-4bit | Dense | 83 tok/s | ~4 GB |
| Mixtral-8x7B-4bit | MoE | 45 tok/s | ~27 GB |
| Qwen3-30B-A3B-4bit | MoE | 98 tok/s | ~17 GB |
| GLM-4.5-Air-3bit | MoE | 45 tok/s | ~47 GB |
| FunASR Paraformer | ASR | 14x real-time | ~1 GB |
| GPT-SoVITS | TTS | ~0.3x real-time | ~4 GB |

*Benchmarked on Apple Silicon (M-series)*
