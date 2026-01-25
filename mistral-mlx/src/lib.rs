//! # mistral-mlx
//!
//! Mistral LLM inference on Apple Silicon with MLX.
//!
//! ## Features
//!
//! - Optimized for pre-quantized 4-bit models
//! - Async pipelining for maximum throughput
//! - Grouped Query Attention (GQA) support
//! - ~74 tok/s on Mistral-7B-4bit (M-series Macs)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mistral_mlx::{load_model, load_tokenizer, Generate, KVCache};
//! use mlx_rs::ops::indexing::NewAxis;
//!
//! let mut model = load_model("path/to/Mistral-7B-4bit")?;
//! let tokenizer = load_tokenizer("path/to/Mistral-7B-4bit")?;
//!
//! let encoding = tokenizer.encode("Hello, ", true)?;
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
    utils::{create_attention_mask, scaled_dot_product_attention, AttentionMask, SdpaMask},
    load_tokenizer,
};

pub use model::{
    load_model, get_model_args, init_cache,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    Attention, AttentionInput, FeedForward, TransformerBlock,
    sample,
};
