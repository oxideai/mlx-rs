# RoPE Multi-Head Attention Bug Fix

**Date**: 2025-01-21
**Component**: `mlx-rs/src/nn/positional_encoding.rs`
**Issue**: Mixtral (and other GQA models) producing garbled output during decode phase

## Problem Description

When running Mixtral-8x7B inference with mlx-rs-lm, the generated text was incoherent:

```
Input: "Hello"
Output: "Hello! How can I help you today? How can I assist you with machine learning, I answer? I'm you today? I'm"
```

Expected output (from Python mlx-lm):
```
Input: "Hello"
Output: "Hello! It's nice to meet you. Is there something you would like to ask or discuss..."
```

## Investigation Process

### 1. Initial Hypothesis: GQA SDPA Issue

First suspected that `scaled_dot_product_attention` wasn't handling Grouped Query Attention (GQA) correctly. Mixtral has 32 query heads but only 8 KV heads.

**Finding**: GQA fix was already in place in `src/utils/mod.rs` (K/V head repetition). Isolated testing confirmed SDPA with repeated K/V produces correct results matching Python.

### 2. Token-by-Token Comparison

Compared token generation step-by-step:
- Prefill: Python and Rust produce same token (22557 → "Hello")
- Decode step 1: Both produce same token (28808)
- Decode step 2: **Divergence** - Python picks 661 ("It"), Rust picks 1602 ("How")

### 3. Attention Layer Tracing

Created `examples/trace_attn_step2.rs` to trace attention computation at decode step 2:

| Component | Python vs Rust |
|-----------|----------------|
| Q projection | ✓ Match |
| K projection | ✓ Match |
| V projection | ✓ Match |
| Q after RoPE (head 0) | ✓ Match |
| Q after RoPE (head 1+) | ✗ **MISMATCH** |

**Key observation**: Before RoPE, all heads match. After RoPE, only head 0 matches.

### 4. Isolated RoPE Testing

Created `examples/test_rope_multihead.rs` and `examples/test_rope_multihead.py` to compare RoPE behavior directly.

**Critical finding**: When input shape `[1, 4, 1, 8]` is reshaped to `[4, 1, 8]` before calling `fast::rope`:
- Sequence 0: Correct values
- Sequences 1-3: **All zeros**

This happens in both Python's `mx.fast.rope` and Rust's `mlx_rs::fast::rope` - it's a kernel behavior when batch dimension has sequence length 1.

## Root Cause

**Location**: `mlx-rs/src/nn/positional_encoding.rs`, line 123

```rust
fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
    let RopeInput { x, offset } = input.into();
    let shape = x.shape();
    let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)])?;  // BUG: This reshape breaks multi-head
    let x = crate::fast::rope(...)?;
    x.reshape(shape)
}
```

The Rust implementation was reshaping `[B, n_heads, L, D]` to `[B*n_heads, L, D]` before calling `fast::rope`.

Python's `nn.RoPE` does **NOT** reshape - it passes the input directly:

```python
# Python mlx/nn/layers/positional_encoding.py
def __call__(self, x, offset: int = 0):
    return mx.fast.rope(x, self.dims, ...)  # No reshape!
```

When sequence length L=1 (decode phase), the underlying MLX rope kernel only processes the first "batch" correctly after the reshape, producing zeros for subsequent batches (which are actually the other attention heads).

## The Fix

Remove the unnecessary reshape in Rust's `RotaryPositionalEncoding::forward`:

```rust
fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
    let RopeInput { x, offset } = input.into();
    // Note: Do NOT reshape the input. The underlying fast::rope kernel
    // expects the input shape to be preserved. Reshaping [B, H, L, D] to
    // [B*H, L, D] causes incorrect behavior for multi-head attention.
    crate::fast::rope(
        x,
        self.dimensions,
        self.traditional,
        self.base,
        self.scale,
        offset,
        None,
    )
}
```

## Verification

### RoPE Multi-Head Test

```
After RoPE (offset=10):
  Head 0: [0.188103, -0.396822, 0.228618, ...]  ✓ Matches Python
  Head 1: [-0.188103, 0.396822, -0.228618, ...] ✓ Matches Python
  Head 2: [1.88103, -3.96822, 2.28618, ...]     ✓ Matches Python
  Head 3: [-1.88103, 3.96822, -2.28618, ...]    ✓ Matches Python
```

### Mixtral Generation

**After fix**:
```
Rust:   "Hello! It's nice to meet you. Is there something specific you would like to ask or discuss about artificial intelligence and machine learning?"
Python: "Hello! It's nice to meet you. Is there something you would like to ask or discuss about computer science and programming?"
```

Both produce coherent, similar responses. Minor variations are expected due to sampling randomness.

## Files Changed

- `mlx-rs/src/nn/positional_encoding.rs` - Removed reshape in `RotaryPositionalEncoding::forward`

## Test Files Created

- `examples/test_rope_multihead.rs` - Rust RoPE multi-head test
- `examples/test_rope_multihead.py` - Python RoPE comparison
- `examples/trace_attn_step2.rs` - Attention layer tracing

## Lessons Learned

1. **Don't assume reshape is safe**: The underlying MLX kernels may have specific expectations about input shapes, especially for edge cases like sequence length 1.

2. **Compare with Python implementation**: When behavior diverges, comparing the exact implementation (not just the API) can reveal subtle differences.

3. **Test at multiple granularities**: The bug only manifested during decode (L=1), not prefill (L>1). Testing both phases is important.

4. **Trace layer by layer**: When outputs diverge, binary search through the computation to find the exact point of divergence.
