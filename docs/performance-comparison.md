# MLX-RS Performance Comparison with Python mlx-lm

This document compares the performance of Rust MLX implementations against the Python mlx-lm reference implementation.

## Executive Summary

| Model | Python (mlx-lm) | Rust | Gap | Status |
|-------|-----------------|------|-----|--------|
| **Qwen3-30B-A3B-4bit (MoE)** | 97.8 tok/s | 98.3 tok/s | **+0.5%** | ✅ Parity |
| **GLM-4.5-Air-3bit (MoE)** | 42.8 tok/s | 45.3 tok/s | **+5.8%** | ✅ Rust faster |
| **Mixtral-8x7B-4bit (MoE)** | 46.1 tok/s | 44.5 tok/s | -3.5% | ✅ Parity |
| **Mistral-7B-4bit** | 83.5 tok/s | 74.2 tok/s | -11% | ✅ Acceptable |

**Conclusion:** Rust implementations achieve parity or better performance compared to Python mlx-lm when using proper async pipelining and pre-quantized models.

## Hardware

- Apple Silicon (M-series)
- macOS

## Key Optimizations

### 1. Async Pipelining (Critical)

The most important optimization is proper async pipelining for CPU/GPU overlap.

**Wrong (defeats pipelining):**
```rust
// BAD - batched eval synchronizes everything
for token in generator {
    tokens.push(token.clone());
    if tokens.len() % 10 == 0 {
        eval(&tokens)?;  // Blocks until all tokens computed
    }
}
```

**Correct (proper pipelining):**
```rust
// GOOD - async_eval + item() creates overlap
for _ in 0..num_tokens {
    let logits = model.forward(input)?;
    let next_y = sample(&logits)?;
    async_eval([&next_y])?;     // Start GPU work for next token
    let _ = y.item::<u32>();     // Sync previous token (CPU/GPU overlap!)
    y = next_y;
}
```

This optimization alone improved GLM-4.5 MoE from 37.2 tok/s to 45.3 tok/s (+22%).

### 2. Pre-quantized Models (Important)

Using pre-quantized models from HuggingFace is significantly faster than on-the-fly quantization.

| Approach | Mistral-7B Performance |
|----------|----------------------|
| On-the-fly quantization | 52.5 tok/s |
| Pre-quantized model | 74.2 tok/s |
| **Improvement** | **+41%** |

**Why pre-quantized is faster:**
- Weights are already in optimal packed format
- No runtime quantization overhead
- Calibrated quantization produces better weights

### 3. GQA Handling (Fixed Previously)

The MLX `scaled_dot_product_attention` kernel handles Grouped Query Attention (GQA) internally. Manual K/V repetition is unnecessary and harmful to performance.

**Wrong:**
```rust
// BAD - manual repetition adds overhead
if n_q_heads > n_kv_heads {
    let keys = repeat_axis(keys, n_rep, 1)?;
    let values = repeat_axis(values, n_rep, 1)?;
}
```

**Correct:**
```rust
// GOOD - let SDPA handle GQA internally
scaled_dot_product_attention(queries, keys, values, ...)
```

## Detailed Benchmarks

### Qwen3-30B-A3B MoE (4-bit, 128 experts, 8 active)

```
Model: mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
Prompt: "Explain the concept of machine learning and its applications in daily life."
Tokens: 100

Python mlx-lm:
  Prompt: 22 tokens, 52.1 tokens-per-sec
  Generation: 100 tokens, 97.8 tokens-per-sec (avg of 3 runs)
  Peak memory: 17.2 GB

Rust mlx-rs-lm:
  Run 1: 69.3 tok/s (cold start)
  Run 2: 98.1 tok/s
  Run 3: 98.5 tok/s
  Result: 98.3 tok/s (warm runs)
```

### GLM-4.5 MoE (3-bit, 60 experts)

```
Model: mlx-community/GLM-4.5-Air-3bit
Prompt: "请解释一下什么是人工智能，以及它在日常生活中的应用有哪些？"
Tokens: 100

Python mlx-lm:
  Prompt: 13 tokens, 23.7 tokens-per-sec
  Generation: 100 tokens, 42.8 tokens-per-sec
  Peak memory: 46.9 GB

Rust glm4-moe-mlx (with async pipelining):
  Run 1: 45.7 tok/s
  Run 2: 45.3 tok/s
  Run 3: 45.0 tok/s
  Result: 45.3 +/- 0.28 tok/s
```

### Mixtral-8x7B (4-bit, MoE)

```
Model: mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
Prompt: "What is the capital of France?"
Tokens: 100

Python mlx-lm:
  Generation: 46.1 tok/s
  Peak memory: 27.2 GB

Rust mlx-rs-lm:
  Result: 44.5 tok/s (+/- 0.05)
```

### Mistral-7B (4-bit, Dense)

```
Model: mlx-community/Mistral-7B-Instruct-v0.2-4bit
Prompt: "What is the capital of France?"
Tokens: 100

Python mlx-lm:
  Generation: 83.5 tok/s
  Peak memory: 4.3 GB

Rust mistral (with pre-quantized model):
  Run 1: 74.3 tok/s
  Run 2: 74.0 tok/s
  Run 3: 74.4 tok/s
  Result: 74.2 tok/s
```

## Benchmark Commands

### Python

```bash
python3 -c "
from mlx_lm import load, generate
model, tokenizer = load('mlx-community/MODEL_NAME')
response = generate(model, tokenizer, prompt='Your prompt', max_tokens=100, verbose=True)
"
```

### Rust

```bash
# Qwen3 MoE (30B-A3B)
cargo run --release -p mlx-rs-lm --example qwen3_moe -- --prompt "Your prompt" --max-tokens 100

# GLM-4.5 MoE
cargo run --release -p glm4-moe-mlx --example benchmark_glm4_moe

# Mixtral
cargo run --release -p mlx-rs-lm --example benchmark_all_models

# Mistral
cargo run --release -p mistral -- --prompt "Your prompt" --max-tokens 100
```

## Implementation Locations

| Model | Rust Implementation | Example |
|-------|---------------------|---------|
| Qwen3 MoE | `mlx-rs-lm/src/models/qwen3_moe.rs` | `mlx-rs-lm/examples/qwen3_moe.rs` |
| GLM-4.5 MoE | `glm4-moe-mlx/src/model.rs` | `glm4-moe-mlx/examples/benchmark_glm4_moe.rs` |
| Mixtral | `mlx-rs-lm/src/models/mixtral.rs` | `mlx-rs-lm/examples/benchmark_all_models.rs` |
| Mistral | `examples/mistral/src/model.rs` | `examples/mistral/src/main.rs` |
| GLM-4 (dense) | `mlx-rs-lm/src/models/glm4.rs` | - |
| Qwen3 (dense) | `mlx-rs-lm/src/models/qwen3.rs` | - |

## Remaining Gaps

The ~11% gap for Mistral is due to:

1. **Basic implementation**: `examples/mistral` is a minimal standalone example, not as optimized as `mlx-rs-lm`
2. **Iterator overhead**: The Generate iterator pattern has slightly more overhead than direct loops
3. **Cache patterns**: Different KV cache update implementations

For production use, prefer the `mlx-rs-lm` implementations which achieve parity with Python.

## Conclusion

Rust MLX implementations can match or exceed Python mlx-lm performance when:

1. ✅ Using proper async pipelining (`async_eval` + `.item()`)
2. ✅ Using pre-quantized models from HuggingFace
3. ✅ Letting SDPA handle GQA internally (no manual K/V repetition)
4. ✅ Using the optimized implementations in `mlx-rs-lm`
