//! # glm4-moe-mlx
//!
//! GLM-4.5 MoE (Mixture of Experts) LLM inference on Apple Silicon with MLX.
//!
//! ## Features
//!
//! - Partial RoPE (rotary position embedding on partial dimensions)
//! - Mixture of Experts with top-k routing (shared + routed experts)
//! - Custom fused SwiGLU Metal kernel (10-12x faster)
//! - 3-bit quantization support
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use glm4_moe_mlx::{load_model, load_tokenizer, Generate, KVCache};
//! use mlx_rs::ops::indexing::NewAxis;
//!
//! let mut model = load_model("path/to/GLM-4-9B-Chat-1M")?;
//! let tokenizer = load_tokenizer("path/to/GLM-4-9B-Chat-1M")?;
//!
//! let encoding = tokenizer.encode("你好", true)?;
//! let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
//! let mut cache = Vec::new();
//!
//! let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);
//!
//! for token in generator.take(50) {
//!     let token = token?;
//!     print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
//! }
//! ```

pub mod model;

// Re-export shared components from mlx-lm-core
pub use mlx_lm_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    fused_swiglu,  // Custom Metal kernel
    utils::{create_attention_mask, scaled_dot_product_attention,
            AttentionMask, SdpaMask},
};

pub use model::{
    load_model, load_tokenizer, get_model_args, init_cache,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    Attention, AttentionInput, MLP, MoE, MoEGate, SwitchGLU, DecoderLayer, LanguageModel,
    sample,
};
