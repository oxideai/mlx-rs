//! Mistral model implementation
//!
//! This module implements the Mistral architecture with:
//! - Grouped Query Attention (GQA)
//! - RoPE (Rotary Position Embedding)
//! - SwiGLU activation
//! - Support for pre-quantized models

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array, argmax_axis,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, Param},
    nn,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use crate::{Error, KeyValueCache, scaled_dot_product_attention, SdpaMask};

/// Quantization configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

/// Model configuration (HuggingFace format)
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_head_dim() -> i32 { 128 }
fn default_rms_norm_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f32 { 10000.0 }

impl ModelArgs {
    pub fn head_dim(&self) -> i32 {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }
}

/// Mistral Attention with GQA and RoPE
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub rope: nn::Rope,
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE with cache offset
        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(cache.offset())
            .build()?;
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(cache.offset())
            .build()?;
        keys = self.rope.forward(k_input)?;

        // Update cache and get all K/V
        (keys, values) = cache.update_and_fetch(keys, values)?;

        // Determine mask mode
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention(
            queries, keys, values, Some(cache), self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Feed-forward network with SwiGLU activation
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct FeedForward {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // SwiGLU: silu(gate) * up
        let activated = nn::silu(gate)?.multiply(up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// Transformer block
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: FeedForward,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        // Self attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let attn_out = self.self_attn.forward(attn_input)?;
        let h = x.add(&attn_out)?;

        // MLP
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        h.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

/// Mistral language model
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MistralModel {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

/// Full Mistral model with LM head
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: MistralModel,
    #[quantizable]
    #[param]
    pub lm_head: MaybeQuantized<nn::Linear>,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<C>,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;

        let mut h = self.model.embed_tokens.forward(inputs)?;

        // Pre-allocate cache if needed
        if cache.is_empty() {
            *cache = init_cache(self.model.layers.len());
        }

        let mask = mask.cloned();

        for (layer, c) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c,
            };
            h = layer.forward(layer_input)?;
        }

        let h = self.model.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.embed_tokens.training_mode(mode);
        for layer in &mut self.model.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.model.norm.training_mode(mode);
        self.lm_head.training_mode(mode);
    }
}

// ============================================================================
// Loading functions
// ============================================================================

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&config_path)?;
    let args: ModelArgs = serde_json::from_reader(file)?;
    Ok(args)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;
    let linear_bias = get_weight_optional(weights, &format!("{}.bias", prefix));

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(linear_bias),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);

    Ok(ql)
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    // Try sharded weights first
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)?;
        let index: serde_json::Value = serde_json::from_str(&json)?;

        let weight_map = index["weight_map"].as_object()
            .ok_or_else(|| Error::InvalidConfig("Invalid weight index".to_string()))?;

        let weight_files: HashSet<&str> = weight_map.values()
            .filter_map(|v| v.as_str())
            .collect();

        let mut all_weights: HashMap<String, Array> = HashMap::new();

        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            let loaded = Array::load_safetensors(&weights_filename)?;
            all_weights.extend(loaded);
        }

        return Ok(all_weights);
    }

    // Try single file
    let weights_file = model_dir.join("model.safetensors");
    if weights_file.exists() {
        let loaded = Array::load_safetensors(&weights_file)?;
        return Ok(loaded);
    }

    Err(Error::InvalidConfig("No weights file found".to_string()))
}

/// Load a pre-quantized Mistral model
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::InvalidConfig("Model must be quantized".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    eprintln!("Loading {}-bit quantized Mistral model...", bits);
    let weights = load_all_weights(model_dir)?;

    let head_dim = args.head_dim();
    let n_heads = args.num_attention_heads;
    let n_kv_heads = args.num_key_value_heads;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let layer_prefix = format!("model.layers.{}", i);

        let attention = Attention {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.q_proj", layer_prefix), group_size, bits
            )?),
            k_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.k_proj", layer_prefix), group_size, bits
            )?),
            v_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.v_proj", layer_prefix), group_size, bits
            )?),
            o_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", layer_prefix), group_size, bits
            )?),
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(args.rope_theta)
                .build()
                .unwrap(),
        };

        let mlp = FeedForward {
            gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.gate_proj", layer_prefix), group_size, bits
            )?),
            up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.up_proj", layer_prefix), group_size, bits
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.down_proj", layer_prefix), group_size, bits
            )?),
        };

        let block = TransformerBlock {
            self_attn: attention,
            mlp,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
        };

        layers.push(block);
    }

    // Embedding may or may not be quantized
    let embed_tokens = if weights.contains_key("model.embed_tokens.scales") {
        let weight = get_weight(&weights, "model.embed_tokens.weight")?;
        let scales = get_weight(&weights, "model.embed_tokens.scales")?;
        let biases = get_weight(&weights, "model.embed_tokens.biases")?;

        let inner = nn::Embedding { weight: Param::new(weight) };
        let mut qe = nn::QuantizedEmbedding {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        };
        qe.freeze_parameters(true);
        MaybeQuantized::Quantized(qe)
    } else {
        let weight = get_weight(&weights, "model.embed_tokens.weight")?;
        MaybeQuantized::Original(nn::Embedding { weight: Param::new(weight) })
    };

    // lm_head may or may not be quantized
    let lm_head = if weights.contains_key("lm_head.scales") {
        MaybeQuantized::Quantized(make_quantized_linear(&weights, "lm_head", group_size, bits)?)
    } else if args.tie_word_embeddings {
        let weight = get_weight(&weights, "model.embed_tokens.weight")?;
        MaybeQuantized::Original(nn::Linear {
            weight: Param::new(weight),
            bias: Param::new(None),
        })
    } else {
        let weight = get_weight(&weights, "lm_head.weight")?;
        MaybeQuantized::Original(nn::Linear {
            weight: Param::new(weight),
            bias: Param::new(None),
        })
    };

    let mistral_model = MistralModel {
        embed_tokens,
        layers,
        norm: nn::RmsNorm {
            weight: Param::new(get_weight(&weights, "model.norm.weight")?),
            eps: args.rms_norm_eps,
        },
    };

    let model = Model {
        args,
        model: mistral_model,
        lm_head,
    };

    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

/// Initialize KV cache for a model
pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

/// Pipelined token generator
pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<C>,
    temp: f32,
    state: GenerateState<'a>,
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Pipelined { current_y: Array },
    Done,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        if cache.is_empty() {
            *cache = init_cache(model.model.layers.len());
        }
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }

    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        use mlx_rs::ops::indexing::{IndexOp, NewAxis};
        let inputs = y.index((.., NewAxis));
        let input = ModelInput {
            inputs: &inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        sample(&logits, self.temp)
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::ops::indexing::IndexOp;
        use mlx_rs::transforms::{async_eval, eval};

        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                // Start async eval and force completion for first token
                tri!(async_eval([&y]));
                tri!(eval([&y]));

                // Compute next token and start its async eval
                let next_y = tri!(self.compute_next(&y));
                tri!(async_eval([&next_y]));

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(y))
            }
            GenerateState::Pipelined { current_y } => {
                let next_y = tri!(self.compute_next(&current_y));
                tri!(async_eval([&next_y]));

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}
