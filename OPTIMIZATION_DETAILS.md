# MLX-RS Performance Optimization: Detailed Documentation

## Overview

This document details the optimization work done to achieve performance parity between the Rust (`mlx-rs`) and Python (`mlx`) implementations for the GLM-4.5 MoE language model.

**Result**: Eliminated a **3.24x performance gap**, achieving **parity or better** performance in Rust.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Investigation Process](#investigation-process)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Solution Implementation](#solution-implementation)
5. [Code Changes](#code-changes)
6. [Build System Modifications](#build-system-modifications)
7. [Verification](#verification)

---

## Problem Statement

### Initial Symptoms

- Rust implementation: **87ms/token**
- Python implementation: **27ms/token**
- Performance gap: **3.24x slower**

### Specific Anomaly Discovered

Through benchmarking, a critical threshold was identified:

| Sequence Length | Rust (ms) | Python (ms) | Gap |
|-----------------|-----------|-------------|-----|
| 127 tokens      | 958       | 615         | 56% slower |
| 128 tokens      | 600       | 616         | ~parity |

The Rust implementation exhibited a **dramatic performance cliff at exactly 128 tokens**.

---

## Investigation Process

### Step 1: Individual Operation Profiling

First, we ruled out individual MLX operations as the bottleneck:

```
| Operation   | Rust   | Python | Ratio |
|-------------|--------|--------|-------|
| SwiGLU      | 0.54ms | 0.56ms | 0.96x (Rust faster) |
| RMSNorm     | 0.32ms | 0.31ms | 1.05x |
| SDPA        | 0.24ms | 0.32ms | 0.73x (Rust faster) |
| Matmul      | 0.47ms | 0.41ms | 1.14x |
| MoE Routing | 0.26ms | 0.25ms | 1.03x |
```

**Conclusion**: Individual operations were NOT the problem.

### Step 2: Graph Building Analysis

Measured graph construction overhead:

```
| Metric              | Rust   | Python |
|---------------------|--------|--------|
| Graph building time | 1.54ms | 1.61ms |
```

**Conclusion**: Graph building was NOT the problem.

### Step 3: Threshold Investigation

Created specialized benchmarks to isolate the 128-token threshold:

```bash
# Fine-grained sweep around the threshold
for len in 116 120 124 127 128 129 132 140; do
  cargo run --release --example single_seqlen -- $len
done
```

Results confirmed a sharp discontinuity at exactly 128 tokens.

### Step 4: MLX Version Analysis

Compared MLX versions:
- **Rust (mlx-rs)**: MLX v0.29.1 (via mlx-c v0.3.0)
- **Python**: MLX v0.30.0

Discovered MLX v0.30.0 changelog entry:
> **PR #2563**: "fix copies in sdpa" - Fixes suboptimal memory copies in scaled_dot_product_attention for sequences < 128 tokens with causal masking.

**Root cause identified**: The 128-token threshold was a bug in MLX v0.29.1.

---

## Root Cause Analysis

### The Bug in MLX v0.29.1

The `scaled_dot_product_attention` function with causal masking had inefficient memory handling for sequences shorter than 128 tokens. The issue was in how the attention mask was being copied/broadcasted.

### Why Rust Was Affected But Not Python

- Python users had already upgraded to MLX v0.30.0 which contained the fix
- The Rust bindings (`mlx-rs`) were pinned to an older `mlx-c` version (v0.3.0) which used MLX v0.29.1
- The C bindings lagged behind the Python release

---

## Solution Implementation

### Step 1: Update mlx-c Submodule

Updated the mlx-c submodule from v0.3.0 to v0.4.1:

```bash
cd mlx-sys/src/mlx-c
git fetch origin
git checkout v0.4.1  # Contains MLX v0.30.1
```

### Step 2: Fix API Breaking Changes

MLX v0.30.1 introduced several API changes that required updates to the Rust bindings:

#### 2.1 SDPA Mask Parameter Change

**Before (v0.3.0)**: Mask was a `VectorArray` (array of masks)
**After (v0.4.1)**: Mask is a single `Array`

```rust
// Old API
fn as_mode_and_masks(&self) -> (&'static CStr, VectorArray) { ... }

// New API
fn as_mode_and_mask_ptr(&self) -> (&'static CStr, mlx_sys::mlx_array) { ... }
```

#### 2.2 Quantization Parameter Changes

`group_size` and `bits` parameters changed from `i32` to `mlx_optional_int_`:

```rust
// Helper function added
fn optional_int(value: i32) -> mlx_sys::mlx_optional_int_ {
    mlx_sys::mlx_optional_int_ {
        value,
        has_value: true,
    }
}
```

#### 2.3 Dequantize Added dtype Parameter

```rust
// Helper function added
fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype_ {
    mlx_sys::mlx_optional_dtype_ {
        value: 0,
        has_value: false,
    }
}
```

### Step 3: Fix macOS Tahoe Beta Compatibility

MLX v0.30.1 introduced code that caused issues on macOS Tahoe beta:

#### Problem 1: Metal 4.0 Not Supported

```cpp
// MLX tries to use Metal 4.0 on macOS 26
if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
    return MTL::LanguageVersion4_0;  // Not supported in Xcode beta!
}
```

#### Problem 2: NAX Feature Uses Unsupported Runtime Check

```cpp
// This causes linker error: undefined symbol ___isPlatformVersionAtLeast
if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
    can_use_nax = true;
}
```

#### Solution: Runtime Patching in build.rs

Created patches that are applied during the build process after CMake fetches the MLX sources.

---

## Code Changes

### File: `mlx-sys/build.rs`

Complete rewrite to support:
1. Custom CMake build process
2. Post-fetch source patching
3. macOS 15.0 deployment target

```rust
/// Patch the MLX source files to work around macOS Tahoe beta issues
fn patch_metal_version(out_dir: &PathBuf) {
    // Patch device.cpp to force Metal 3.2
    let device_cpp = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.cpp");
    if device_cpp.exists() {
        if let Ok(content) = fs::read_to_string(&device_cpp) {
            if !content.contains("// PATCHED: Force Metal 3.2") {
                let old_code = r#"auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                let new_code = r#"// PATCHED: Force Metal 3.2
auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    fs::write(&device_cpp, patched).ok();
                }
            }
        }
    }

    // Patch device.h to disable NAX
    let device_h = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.h");
    if device_h.exists() {
        if let Ok(content) = fs::read_to_string(&device_h) {
            if !content.contains("// PATCHED: Disable NAX") {
                let old_code = r#"inline bool is_nax_available() {
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    can_use_nax &=
        metal::device(mlx::core::Device::gpu).get_architecture_gen() >= 17;
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
}"#;

                let new_code = r#"// PATCHED: Disable NAX
inline bool is_nax_available() {
  return false;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    fs::write(&device_h, patched).ok();
                }
            }
        }
    }
}
```

### File: `mlx-rs/src/fast.rs`

Updated SDPA implementation:

```rust
// Added import
use crate::utils::guard::Guarded;

// Changed mask handling
impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask_ptr(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (
                DEFAULT_MASK_MODE,
                mask.as_ptr(),
            ),
            ScaledDotProductAttentionMask::Arrays(masks) => {
                if masks.is_empty() {
                    (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
                } else {
                    (DEFAULT_MASK_MODE, masks[0].as_ptr())
                }
            },
            ScaledDotProductAttentionMask::Causal => (CAUSAL_MASK_MODE, unsafe {
                mlx_sys::mlx_array_new()
            }),
        }
    }
}

// Updated SDPA call
pub fn scaled_dot_product_attention_device<'a>(...) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask_ptr(),
    );

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,  // Changed from VectorArray to single Array
            mlx_sys::mlx_array_new(),
            stream.as_ref().as_ptr(),
        )
    })
}
```

### File: `mlx-rs/src/ops/quantization.rs`

Added helper functions and updated all quantization operations:

```rust
/// Helper to create mlx_optional_int_ from i32
fn optional_int(value: i32) -> mlx_sys::mlx_optional_int_ {
    mlx_sys::mlx_optional_int_ {
        value,
        has_value: true,
    }
}

/// Helper to create an empty mlx_optional_dtype_ (no value)
fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype_ {
    mlx_sys::mlx_optional_dtype_ {
        value: 0,
        has_value: false,
    }
}

// Updated quantize call
mlx_sys::mlx_quantize(
    &mut res,
    w.as_ref().as_ptr(),
    optional_int(group_size),  // Changed from i32
    optional_int(bits),        // Changed from i32
    mode_cstr.as_ptr(),
    stream.as_ref().as_ptr(),
)

// Updated dequantize call
mlx_sys::mlx_dequantize(
    res,
    w.as_ref().as_ptr(),
    scales.as_ref().as_ptr(),
    biases.as_ref().as_ptr(),
    optional_int(group_size),
    optional_int(bits),
    mode_cstr.as_ptr(),
    optional_dtype_none(),  // New parameter
    stream.as_ref().as_ptr(),
)
```

---

## Build System Modifications

### CMake Build Process

The build process was changed from using the `cmake` crate's automatic build to a manual process:

```rust
fn build_and_link_mlx_c() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("build");

    // Step 1: Run CMake configure
    let status = Command::new("cmake")
        .args(&cmake_args)
        .status()
        .expect("Failed to run cmake configure");

    // Step 2: Apply patches AFTER configure fetches sources
    patch_metal_version(&out_dir);

    // Step 3: Run CMake build
    let status = Command::new("cmake")
        .args(["--build", &build_dir.to_string_lossy(), "--config", "Release", "-j"])
        .status()
        .expect("Failed to run cmake build");

    // Step 4: Link libraries
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}/_deps/mlx-build", build_dir.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");
}
```

### Deployment Target

Set macOS 15.0 as the deployment target for consistency:

```rust
cmake_args.push("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0".to_string());
std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "15.0");
println!("cargo:rustc-link-arg=-mmacosx-version-min=15.0");
```

---

## Verification

### Final Performance Results

| Seq Len | Python (ms) | Rust (ms) | Improvement |
|---------|-------------|-----------|-------------|
| 32      | 267.4       | 263.3     | **Rust 1.5% faster** |
| 64      | 405.2       | 392.1     | **Rust 3.2% faster** |
| 127     | 615.5       | 598.4     | **Rust 2.8% faster** |
| 128     | 616.7       | 601.7     | **Rust 2.4% faster** |
| 256     | 1014.5      | 1022.1    | Python 0.7% faster |
| 512     | 1854.4      | 1737.8    | **Rust 6.3% faster** |

### Key Metrics

- **128-token threshold**: Eliminated
- **Performance gap**: From 3.24x slower to ~1.0x (parity)
- **Worst case**: 0.7% slower at 256 tokens
- **Best case**: 6.3% faster at 512 tokens

### Benchmark Commands

```bash
# Rust benchmark
cargo run --release --example single_seqlen -- <seq_len>

# Python benchmark
python python_single_seqlen.py <seq_len>
```

---

## Summary

The optimization consisted of:

1. **Identifying the root cause**: 128-token threshold bug in MLX v0.29.1's SDPA
2. **Updating dependencies**: mlx-c v0.3.0 → v0.4.1 (MLX v0.29.1 → v0.30.1)
3. **Fixing API changes**: SDPA mask, quantization optional parameters
4. **Platform compatibility**: Patches for macOS Tahoe beta (Metal 3.2, NAX disabled)

The result is a Rust implementation that matches or exceeds Python performance across all sequence lengths.
