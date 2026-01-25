use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, Param},
    nn,
    ops::concatenate_axis,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

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

/// HuggingFace-style config.json format
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(alias = "hidden_size")]
    pub dim: i32,
    #[serde(alias = "num_hidden_layers")]
    pub n_layers: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(alias = "intermediate_size")]
    pub hidden_dim: i32,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: i32,
    #[serde(alias = "num_key_value_heads")]
    pub n_kv_heads: i32,
    #[serde(alias = "rms_norm_eps")]
    pub norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_head_dim() -> i32 { 128 }  // Mistral default
fn default_rope_theta() -> Option<f32> { Some(10000.0) }

impl ModelArgs {
    pub const DEFAULT_ROPE_THETA: f32 = 10000.0;

    /// Compute head_dim from dim and n_heads if not specified
    pub fn head_dim(&self) -> i32 {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.dim / self.n_heads
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    repeats: i32,
    scale: f32,

    #[quantizable]
    #[param]
    wq: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wk: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wv: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wo: MaybeQuantized<nn::Linear>,

    #[param]
    rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads;
        let head_dim = args.head_dim();
        let repeats = n_heads / n_kv_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let wq = nn::LinearBuilder::new(args.dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let wk = nn::LinearBuilder::new(args.dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let wv = nn::LinearBuilder::new(args.dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let wo = nn::LinearBuilder::new(n_heads * head_dim, args.dim)
            .bias(false)
            .build()?;
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)  // Mistral uses non-traditional RoPE
            .base(args.rope_theta.unwrap_or(ModelArgs::DEFAULT_ROPE_THETA))
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats,
            scale,
            wq: MaybeQuantized::new(wq),
            wk: MaybeQuantized::new(wk),
            wv: MaybeQuantized::new(wv),
            wo: MaybeQuantized::new(wo),
            rope,
        })
    }
}

struct AttentionInput<'a> {
    x: &'a Array,
    mask: Option<ScaledDotProductAttentionMask<'a>>,
    cache: Option<(&'a Array, &'a Array)>,
}

struct AttentionOutput {
    output: Array,
    cache: (Array, Array),
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = AttentionOutput;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        // NOTE: this will panic if the input shape is not correct
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut queries = self.wq.forward(x)?;
        let mut keys = self.wk.forward(x)?;
        let mut values = self.wv.forward(x)?;

        // Prepare the queries, keys, and values for the attention computation
        queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        match cache {
            Some((key_cache, value_cache)) => {
                let offset = key_cache.shape()[2];
                queries = self.rope.forward((&queries, offset))?;
                keys = self.rope.forward((&keys, offset))?;
                keys = concatenate_axis(&[key_cache, &keys], 2)?;
                values = concatenate_axis(&[value_cache, &values], 2)?;
            }
            None => {
                queries = self.rope.forward(&queries)?;
                keys = self.rope.forward(&keys)?;
            }
        }

        let output = scaled_dot_product_attention(queries, &keys, &values, self.scale, mask)?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let output = self.wo.forward(&output)?;

        Ok(AttentionOutput {
            output,
            cache: (keys, values),
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.wq.training_mode(mode);
        self.wk.training_mode(mode);
        self.wv.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct FeedForward {
    #[quantizable]
    #[param]
    w1: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w2: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w3: MaybeQuantized<nn::Linear>,
}

impl FeedForward {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let w1 = nn::LinearBuilder::new(args.dim, args.hidden_dim)
            .bias(false)
            .build()?;
        let w2 = nn::LinearBuilder::new(args.hidden_dim, args.dim)
            .bias(false)
            .build()?;
        let w3 = nn::LinearBuilder::new(args.dim, args.dim)
            .bias(false)
            .build()?;
        Ok(Self {
            w1: MaybeQuantized::new(w1),
            w2: MaybeQuantized::new(w2),
            w3: MaybeQuantized::new(w3),
        })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &'_ Array) -> Result<Self::Output, Self::Error> {
        let w2_input = nn::silu(self.w1.forward(x)?)?.multiply(self.w3.forward(x)?)?;
        self.w2.forward(&w2_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.w1.training_mode(mode);
        self.w2.training_mode(mode);
        self.w3.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct TransformerBlock {
    n_heads: i32,
    dim: i32,

    #[quantizable]
    #[param]
    attention: Attention,

    #[quantizable]
    #[param]
    feed_forward: FeedForward,

    #[param]
    attention_norm: nn::RmsNorm,

    #[param]
    ffn_norm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let dim = args.dim;

        let attention = Attention::new(args)?;
        let feed_forward = FeedForward::new(args)?;
        let attention_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        let ffn_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        Ok(Self {
            n_heads,
            dim,
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

impl Module<AttentionInput<'_>> for TransformerBlock {
    type Output = AttentionOutput;

    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let norm_x = self.attention_norm.forward(x)?;
        let attention_input = AttentionInput {
            x: &norm_x,
            mask,
            cache,
        };
        let attention_output = self.attention.forward(attention_input)?;

        let r = attention_output.output;
        let cache = attention_output.cache;

        let h = x.add(r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        let output = h.add(r)?;

        Ok(AttentionOutput { output, cache })
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
        self.feed_forward.training_mode(mode);
        self.attention_norm.training_mode(mode);
        self.ffn_norm.training_mode(mode);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MistralError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mistral {
    vocab_size: i32,
    n_layers: i32,

    #[quantizable]
    #[param]
    tok_embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    layers: Vec<TransformerBlock>,

    #[param]
    norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    output: MaybeQuantized<nn::Linear>,
}

impl Mistral {
    pub fn new(args: &ModelArgs) -> Result<Self, MistralError> {
        let vocab_size = args.vocab_size;
        if vocab_size <= 0 {
            // We would still have to check for the zero case even if we switch to u32
            return Err(MistralError::InvalidVocabSize(vocab_size));
        }
        let n_layers = args.n_layers;

        let tok_embeddings = nn::Embedding::new(vocab_size, args.dim)?;
        let layers = (0..n_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.dim)
            .eps(args.norm_eps)
            .build()?;
        let output = nn::LinearBuilder::new(args.dim, vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            vocab_size,
            n_layers,
            tok_embeddings: MaybeQuantized::new(tok_embeddings),
            layers,
            norm,
            output: MaybeQuantized::new(output),
        })
    }
}

pub struct MistralInput<'a> {
    pub inputs: &'a Array,
    pub cache: &'a [Option<(Array, Array)>],
}
pub struct MistralOutput {
    pub logits: Array,
    pub cache: Vec<Option<(Array, Array)>>,
}

impl Module<MistralInput<'_>> for Mistral {
    type Output = MistralOutput;

    type Error = MistralError;

    fn forward(&mut self, input: MistralInput<'_>) -> Result<Self::Output, Self::Error> {
        let MistralInput { inputs, cache } = input;

        let mut h = self.tok_embeddings.forward(inputs)?;

        let mut mask = None;
        if h.shape()[1] > 1 {
            let mask_ = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
            let mask_ = mask_.as_dtype(h.dtype())?;
            mask = Some(mask_);
        }

        let mut out_cache = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_entry = cache.get(i).and_then(Option::as_ref).map(|(k, v)| (k, v));
            let input = AttentionInput {
                x: &h,
                mask: mask.as_ref().map(Into::into),
                cache: cache_entry,
            };
            let output = layer.forward(input)?;
            h = output.output;
            out_cache.push(Some(output.cache));
        }

        let output = self.output.forward(&self.norm.forward(&h)?)?;

        Ok(MistralOutput {
            logits: output,
            cache: out_cache,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.tok_embeddings.training_mode(mode);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
        self.norm.training_mode(mode);
        self.output.training_mode(mode);
    }
}

// ============================================================================
// Quantized model loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, MistralError> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| MistralError::Exception(Exception::custom(format!("Weight not found: {}", key))))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, MistralError> {
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

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, MistralError> {
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

pub fn load_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Mistral, MistralError> {
    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| MistralError::Exception(Exception::custom("No quantization config")))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    // Load all weights
    let weights = load_all_weights(model_dir)?;

    let head_dim = args.head_dim();
    let n_heads = args.n_heads;
    let n_kv_heads = args.n_kv_heads;

    let mut layers = Vec::with_capacity(args.n_layers as usize);

    for i in 0..args.n_layers {
        let layer_prefix = format!("model.layers.{}", i);

        // Build attention
        let attention = Attention {
            n_heads,
            n_kv_heads,
            repeats: n_heads / n_kv_heads,
            scale: (head_dim as f32).powf(-0.5),
            wq: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.q_proj", layer_prefix), group_size, bits
            )?),
            wk: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.k_proj", layer_prefix), group_size, bits
            )?),
            wv: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.v_proj", layer_prefix), group_size, bits
            )?),
            wo: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", layer_prefix), group_size, bits
            )?),
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(args.rope_theta.unwrap_or(ModelArgs::DEFAULT_ROPE_THETA))
                .build()
                .unwrap(),
        };

        // Build feed forward
        let feed_forward = FeedForward {
            w1: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.gate_proj", layer_prefix), group_size, bits
            )?),
            w2: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.down_proj", layer_prefix), group_size, bits
            )?),
            w3: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.up_proj", layer_prefix), group_size, bits
            )?),
        };

        let block = TransformerBlock {
            n_heads,
            dim: args.dim,
            attention,
            feed_forward,
            attention_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", layer_prefix))?),
                eps: args.norm_eps,
            },
            ffn_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", layer_prefix))?),
                eps: args.norm_eps,
            },
        };

        layers.push(block);
    }

    // Embedding may or may not be quantized - check for scales
    let tok_embeddings = if weights.contains_key("model.embed_tokens.scales") {
        MaybeQuantized::Quantized(make_quantized_embedding(
            &weights, "model.embed_tokens", group_size, bits
        )?)
    } else {
        // Non-quantized embedding
        let weight = get_weight(&weights, "model.embed_tokens.weight")?;
        MaybeQuantized::Original(nn::Embedding {
            weight: Param::new(weight),
        })
    };

    // lm_head may or may not be quantized
    let output = if weights.contains_key("lm_head.scales") {
        MaybeQuantized::Quantized(make_quantized_linear(
            &weights, "lm_head", group_size, bits
        )?)
    } else if args.tie_word_embeddings {
        // Tied weights - use embedding weight as linear
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

    let model = Mistral {
        vocab_size: args.vocab_size,
        n_layers: args.n_layers,
        tok_embeddings,
        layers,
        norm: nn::RmsNorm {
            weight: Param::new(get_weight(&weights, "model.norm.weight")?),
            eps: args.norm_eps,
        },
        output,
    };

    Ok(model)
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, MistralError> {
    use std::collections::HashSet;

    // Try sharded weights first
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| MistralError::Exception(Exception::custom(format!("Failed to read index: {}", e))))?;
        let index: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| MistralError::Exception(Exception::custom(format!("Failed to parse index: {}", e))))?;

        let weight_map = index["weight_map"].as_object()
            .ok_or_else(|| MistralError::Exception(Exception::custom("Invalid weight index")))?;

        let weight_files: HashSet<&str> = weight_map.values()
            .filter_map(|v| v.as_str())
            .collect();

        let mut all_weights: HashMap<String, Array> = HashMap::new();

        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            let loaded = Array::load_safetensors(&weights_filename)
                .map_err(|e| MistralError::Exception(Exception::custom(format!("Failed to load {}: {:?}", weight_file, e))))?;
            all_weights.extend(loaded);
        }

        return Ok(all_weights);
    }

    // Try single file
    let weights_file = model_dir.join("model.safetensors");
    if weights_file.exists() {
        let loaded = Array::load_safetensors(&weights_file)
            .map_err(|e| MistralError::Exception(Exception::custom(format!("Failed to load weights: {:?}", e))))?;
        return Ok(loaded);
    }

    Err(MistralError::Exception(Exception::custom("No weights file found")))
}
