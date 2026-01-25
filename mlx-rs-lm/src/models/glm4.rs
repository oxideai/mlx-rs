//! GLM-4 model implementation
//!
//! This module implements the GLM-4 architecture with support for:
//! - Partial RoPE (rotary position embedding on partial dimensions)
//! - Fused gate_up_proj in MLP
//! - Extra LayerNorms (post_self_attn, post_mlp)
//! - Quantized model loading

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        indexing::{IndexOp, NewAxis},
        split,
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_attention_mask,
        rope::FloatOrString,
        AttentionMask,
        SdpaMask,
    },
};

/// Quantization configuration for the model
#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Partial rotary factor - GLM4 uses 0.5 (RoPE on half of dimensions)
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// Whether attention layers have bias (GLM4 has QKV bias)
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Quantization config (present for quantized models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_max_position_embeddings() -> i32 { 32768 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_partial_rotary_factor() -> f32 { 0.5 }
fn default_attention_bias() -> bool { true }

/// GLM4 Attention with partial RoPE
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,  // Dimensions to apply RoPE to
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

impl Glm4Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        // Partial RoPE: only apply to first rope_dim dimensions
        let rope_dim = (head_dim as f32 * args.partial_rotary_factor) as i32;

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        // O projection typically has no bias in GLM4
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        // RoPE for partial dimensions
        let rope = nn::RopeBuilder::new(rope_dim)
            .base(args.rope_theta)
            .traditional(true)  // GLM4 uses traditional RoPE
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

pub struct Glm4AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<Glm4AttentionInput<'_, C>> for Glm4Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Glm4AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Glm4AttentionInput { x, mask, mut cache } = input;

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

        // Apply partial RoPE - only to first rope_dim dimensions
        if let Some(cache) = cache.as_mut() {
            // Split into rotary and pass-through parts
            let q_rot = queries.index((.., .., .., ..self.rope_dim));
            let q_pass = queries.index((.., .., .., self.rope_dim..));
            let k_rot = keys.index((.., .., .., ..self.rope_dim));
            let k_pass = keys.index((.., .., .., self.rope_dim..));

            let q_input = nn::RopeInputBuilder::new(&q_rot)
                .offset(cache.offset())
                .build()?;
            let q_rot = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&k_rot)
                .offset(cache.offset())
                .build()?;
            let k_rot = self.rope.forward(k_input)?;

            // Concatenate back
            queries = mlx_rs::ops::concatenate_axis(&[q_rot, q_pass], -1)?;
            keys = mlx_rs::ops::concatenate_axis(&[k_rot, k_pass], -1)?;

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            let q_rot = queries.index((.., .., .., ..self.rope_dim));
            let q_pass = queries.index((.., .., .., self.rope_dim..));
            let k_rot = keys.index((.., .., .., ..self.rope_dim));
            let k_pass = keys.index((.., .., .., self.rope_dim..));

            let q_rot = self.rope.forward(nn::RopeInput::new(&q_rot))?;
            let k_rot = self.rope.forward(nn::RopeInput::new(&k_rot))?;

            queries = mlx_rs::ops::concatenate_axis(&[q_rot, q_pass], -1)?;
            keys = mlx_rs::ops::concatenate_axis(&[k_rot, k_pass], -1)?;
        }

        // Determine mask mode: use Causal for prefill (L > 1), None for decode (L == 1)
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, sdpa_mask,
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

/// GLM4 MLP with fused gate_up_proj
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Mlp {
    #[quantizable]
    #[param]
    pub gate_up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Glm4Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        // Fused gate and up projection: output is 2 * hidden_dim
        let gate_up_proj = nn::LinearBuilder::new(dim, 2 * hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_up_proj: MaybeQuantized::Original(gate_up_proj),
            down_proj: MaybeQuantized::Original(down_proj),
        })
    }
}

impl Module<&Array> for Glm4Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let x = self.gate_up_proj.forward(input)?;
        // Split into gate and up parts
        let parts = split(&x, 2, -1)?;
        let gate = &parts[0];
        let up_states = &parts[1];
        let down_input = nn::silu(gate.clone())?.multiply(up_states)?;
        self.down_proj.forward(&down_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// GLM4 Decoder Layer with extra LayerNorms
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4DecoderLayer {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Glm4Attention,
    #[quantizable]
    #[param]
    pub mlp: Glm4Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub post_self_attn_layernorm: nn::RmsNorm,
    #[param]
    pub post_mlp_layernorm: nn::RmsNorm,
}

impl Glm4DecoderLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Glm4Attention::new(args)?;
        let mlp = Glm4Mlp::new(args.hidden_size, args.intermediate_size)?;

        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_self_attn_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_mlp_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }
}

impl<C> Module<Glm4AttentionInput<'_, C>> for Glm4DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Glm4AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Glm4AttentionInput { x, mask, cache } = input;

        // Self attention with post normalization
        let attn_input = Glm4AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let attn_output = self.self_attn.forward(attn_input)?;
        let x = x.add(self.post_self_attn_layernorm.forward(&attn_output)?)?;

        // MLP with post normalization
        let residual = x.clone();
        let mlp_input = self.post_attention_layernorm.forward(&x)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;
        let x = self.post_mlp_layernorm.forward(&mlp_output)?.add(residual)?;

        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        <Glm4Attention as Module<Glm4AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.post_self_attn_layernorm.training_mode(mode);
        self.post_mlp_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Model {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<Glm4DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Glm4Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..num_hidden_layers)
            .map(|_| Glm4DecoderLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct Glm4ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<Glm4ModelInput<'_, C>> for Glm4Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Glm4ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Glm4ModelInput { inputs, mask, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"))
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = Glm4AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <Glm4DecoderLayer as Module<Glm4AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Glm4Model,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Glm4Model::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self { args, model, lm_head })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C> Module<Glm4ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Glm4ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Glm4Model as Module<Glm4ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

// ============================================================================
// Loading functions
// ============================================================================

pub fn load_glm4_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_glm4_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_glm4_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_glm4_model_args(model_dir)?;

    // Check if this is a quantized model
    if model_args.quantization.is_some() {
        return load_glm4_model_quantized(model_dir, &model_args);
    }

    let mut model = Model::new(model_args)?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(weights_index)?;
    let weight_map: WeightMap = serde_json::from_str(&json)?;

    let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

    for weight_file in weight_files {
        let weights_filename = model_dir.join(weight_file);
        model.load_safetensors(&weights_filename)?;
    }

    Ok(model)
}

// ============================================================================
// Quantized model loading
// ============================================================================

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(weights_index)?;
    let weight_map: WeightMap = serde_json::from_str(&json)?;

    let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

    let mut all_weights: HashMap<String, Array> = HashMap::new();

    for weight_file in weight_files {
        let weights_filename = model_dir.join(weight_file);
        let loaded = Array::load_safetensors(&weights_filename)?;
        all_weights.extend(loaded);
    }

    Ok(all_weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::Message(format!("Weight not found: {}", key)))
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

    // Check for optional linear bias (separate from quantization biases)
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

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);

    Ok(qe)
}

fn load_glm4_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::Message("No quantization config".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    let weights = load_all_weights(model_dir)?;

    // Calculate rope_dim for partial RoPE
    let rope_dim = (args.head_dim as f32 * args.partial_rotary_factor) as i32;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let layer_prefix = format!("model.layers.{}", i);

        // Build attention with partial RoPE
        let attention = Glm4Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim: args.head_dim,
            rope_dim,
            scale: (args.head_dim as f32).sqrt().recip(),
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
            // RopeBuilder::build() returns Result<_, Infallible> - can never fail
            rope: nn::RopeBuilder::new(rope_dim)
                .base(args.rope_theta)
                .traditional(true)
                .build()
                .unwrap(),
        };

        // Build MLP with fused gate_up_proj
        let mlp = Glm4Mlp {
            gate_up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.gate_up_proj", layer_prefix), group_size, bits
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.down_proj", layer_prefix), group_size, bits
            )?),
        };

        // Build decoder layer with all 4 LayerNorms
        let block = Glm4DecoderLayer {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
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
            post_self_attn_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_self_attn_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            post_mlp_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_mlp_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
        };

        layers.push(block);
    }

    let glm4_model = Glm4Model {
        vocab_size: args.vocab_size,
        num_hidden_layers: args.num_hidden_layers,
        embed_tokens: MaybeQuantized::Quantized(make_quantized_embedding(
            &weights, "model.embed_tokens", group_size, bits
        )?),
        layers,
        norm: nn::RmsNorm {
            weight: Param::new(get_weight(&weights, "model.norm.weight")?),
            eps: args.rms_norm_eps,
        },
    };

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            &weights, "lm_head", group_size, bits
        )?))
    } else {
        None
    };

    let model = Model {
        args: args.clone(),
        model: glm4_model,
        lm_head,
    };

    model.eval()?;

    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
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
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = Glm4ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = Glm4ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits, self.temp));

                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
        }
    }
}
