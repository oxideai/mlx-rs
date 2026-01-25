//! GLM-4.5 MoE (Mixture of Experts) model implementation
//!
//! This module implements the GLM-4.5 MoE architecture with:
//! - Partial RoPE (rotary position embedding on partial dimensions)
//! - Mixture of Experts with top-k routing
//! - Shared experts + routed experts
//! - 3-bit quantization support

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
        indexing::{IndexOp, NewAxis, take_axis, take_along_axis},
        sigmoid,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    metal_kernels::fused_swiglu,
    utils::{
        rope::FloatOrString,
        SdpaMask,
    },
};

// Note: Compiled functions were tested but didn't improve performance.
// Individual MLX operations have same speed in Rust and Python.
// The remaining performance gap (~3.3x) is likely due to:
// 1. Missing `mode` parameter in gather_qmm C binding
// 2. Different cache update patterns
// 3. Graph structure differences at the integration level

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
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,

    // MoE specific fields
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub n_routed_experts: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,
    #[serde(default)]
    pub use_qk_norm: bool,

    /// Quantization config (present for quantized models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_max_position_embeddings() -> i32 { 131072 }
fn default_rope_theta() -> f32 { 1000000.0 }
fn default_partial_rotary_factor() -> f32 { 0.5 }
fn default_attention_bias() -> bool { true }
fn default_num_experts_per_tok() -> i32 { 8 }
fn default_first_k_dense_replace() -> i32 { 1 }
fn default_routed_scaling_factor() -> f32 { 1.0 }
fn default_n_group() -> i32 { 1 }
fn default_topk_group() -> i32 { 1 }

/// GLM4 MoE Attention with partial RoPE
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub scale: f32,
    pub use_qk_norm: bool,

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
    #[param]
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();
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
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        // GLM4 MoE uses traditional=false for RoPE
        let rope = nn::RopeBuilder::new(rope_dim)
            .base(args.rope_theta)
            .traditional(false)
            .build()
            .unwrap();

        let (q_norm, k_norm) = if args.use_qk_norm {
            (
                Some(nn::RmsNormBuilder::new(head_dim).eps(args.rms_norm_eps).build()?),
                Some(nn::RmsNormBuilder::new(head_dim).eps(args.rms_norm_eps).build()?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            scale,
            use_qk_norm: args.use_qk_norm,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
            q_norm,
            k_norm,
        })
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,  // Removed Option wrapper - cache is always present during generation
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

        let mut queries = queries.reshape(&[B, L, self.n_heads, -1])?;
        let mut keys = keys.reshape(&[B, L, self.n_kv_heads, -1])?;

        // Apply QK norm if enabled
        if self.use_qk_norm {
            if let Some(ref mut q_norm) = self.q_norm {
                queries = q_norm.forward(&queries)?;
            }
            if let Some(ref mut k_norm) = self.k_norm {
                keys = k_norm.forward(&keys)?;
            }
        }

        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply partial RoPE with cache offset
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

        // Determine mask mode: use Causal for prefill (L > 1), None for decode (L == 1)
        // If explicit mask is provided, use it; otherwise use optimized causal mode for prefill
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),  // Prefill: use hardware-optimized causal
            None => None,  // Decode: no mask needed
        };

        let output = crate::utils::scaled_dot_product_attention(
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

/// Standard MLP (used for dense layers and shared experts)
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MLP {
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

impl MLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            up_proj: MaybeQuantized::Original(up_proj),
            down_proj: MaybeQuantized::Original(down_proj),
        })
    }
}

impl Module<&Array> for MLP {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // SwiGLU activation: silu(gate) * up - using fused Metal kernel
        let activated = fused_swiglu(&up, &gate)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// MoE Gate for expert routing
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoEGate {
    pub top_k: i32,
    pub n_routed_experts: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub n_group: i32,
    pub topk_group: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub e_score_correction_bias: Param<Array>,
}

impl MoEGate {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let weight = Array::zeros::<f32>(&[args.n_routed_experts, args.hidden_size])?;
        let e_score_correction_bias = Array::zeros::<f32>(&[args.n_routed_experts])?;

        Ok(Self {
            top_k: args.num_experts_per_tok,
            n_routed_experts: args.n_routed_experts,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
            n_group: args.n_group,
            topk_group: args.topk_group,
            weight: Param::new(weight),
            e_score_correction_bias: Param::new(e_score_correction_bias),
        })
    }

    /// Returns (expert_indices, expert_weights) for top-k routing
    /// Non-compiled version to test overhead.
    pub fn route(&self, x: &Array) -> Result<(Array, Array), Exception> {
        // x: [B, L, D] -> gates: [B, L, n_experts]
        let gates = x.matmul(&(*self.weight).t())?;

        // Compute sigmoid scores once
        let orig_scores = sigmoid(&gates.as_dtype(Dtype::Float32)?)?;
        let scores_with_bias = orig_scores.add(&*self.e_score_correction_bias)?;

        // Top-k selection
        let neg_scores = scores_with_bias.negative()?;
        let partitioned_inds = mlx_rs::ops::argpartition_axis(&neg_scores, self.top_k - 1, -1)?;
        let inds = partitioned_inds.index((.., .., ..self.top_k));
        let selected_scores = take_along_axis(&orig_scores, &inds, -1)?;

        // Normalize and scale
        let scaling_arr = array!(self.routed_scaling_factor);
        let final_scores = if self.norm_topk_prob && self.top_k > 1 {
            let denom = selected_scores.sum_axis(-1, true)?;
            let normalized = selected_scores.divide(&denom)?;
            normalized.multiply(&scaling_arr)?
        } else {
            selected_scores.multiply(&scaling_arr)?
        };

        Ok((inds, final_scores))
    }
}

/// Quantized Switch Linear for MoE experts
/// Stores stacked weights for all experts: [n_experts, out_dim, in_dim]
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub num_experts: i32,
    pub input_dims: i32,
    pub output_dims: i32,
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl QuantizedSwitchLinear {
    /// Apply gather_qmm with already-expanded input.
    /// x: [..., groups, D], indices: [..., k] -> output: [..., k, out_dim]
    /// Note: groups should be 1 (broadcasts to k) or match k
    /// If sorted_indices is true, assumes indices are pre-sorted for optimized memory access.
    pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array, Exception> {
        mlx_rs::ops::gather_qmm(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            None::<&Array>,        // lhs_indices - not used
            Some(indices),         // rhs_indices - expert selection
            true,                  // transpose
            self.group_size,
            self.bits,
            None::<&str>,          // mode - default "affine"
            sorted_indices,        // sorted_indices - enables optimized kernels
        )
    }
}

/// Sort tokens by their expert indices for coalesced memory access.
/// Returns (sorted_x, sorted_indices, inverse_order).
///
/// This optimization groups tokens going to the same expert together,
/// dramatically improving memory bandwidth utilization.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let indices_shape = indices.shape();
    let m = *indices_shape.last().unwrap() as i32;  // k (num experts per token)

    // Flatten indices: [B, L, k] -> [B*L*k]
    let indices_flat = indices.flatten(None, None)?;

    // Get sort order: argsort gives indices that would sort the array
    let order = mlx_rs::ops::argsort(&indices_flat)?;

    // Get inverse order for unsorting later
    let inv_order = mlx_rs::ops::argsort(&order)?;

    // Flatten x from [B, L, 1, 1, D] to [B*L, 1, D] then reorder
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;
    let x_flat = x.reshape(&[-1, 1, d])?;  // [B*L, 1, D]

    // Reorder x: x_flat[order // m] selects the token for each sorted position
    // order // m gives the token index (since each token has m expert slots)
    let token_order = order.floor_divide(mlx_rs::array!(m))?;

    // Use take_axis to gather elements along axis 0
    let x_sorted = take_axis(&x_flat, &token_order, 0)?;

    // Reorder indices using take_axis
    let indices_sorted = take_axis(&indices_flat, &order, 0)?;

    Ok((x_sorted, indices_sorted, inv_order))
}

/// Unsort the output back to original token order.
fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> Result<Array, Exception> {
    // x is [B*L*k, 1, D], reorder and reshape back to [B, L, k, 1, D]
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;

    // Flatten to [B*L*k, D] for indexing
    let x_flat = x.reshape(&[-1, d])?;

    // Reorder back to original order using take_axis
    let x_unsorted = take_axis(&x_flat, inv_order, 0)?;

    // Reshape to original shape [B, L, k, 1, D]
    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.push(1);
    new_shape.push(d);
    x_unsorted.reshape(&new_shape)
}

/// SwitchGLU MLP for routed experts
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    #[param]
    pub gate_proj: QuantizedSwitchLinear,
    #[param]
    pub up_proj: QuantizedSwitchLinear,
    #[param]
    pub down_proj: QuantizedSwitchLinear,
}

impl SwitchGLU {
    /// Apply SwitchGLU experts using efficient gather_qmm operations.
    /// Following Python MLX-LM pattern exactly, including sorting optimization.
    /// x: [B, L, D], indices: [B, L, k] -> output: [B, L, k, D]
    pub fn forward_experts(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let indices_shape = indices.shape();
        let b = indices_shape[0];
        let l = indices_shape[1];
        let k = indices_shape[2];

        // Expand x as in Python: [B, L, D] -> [B, L, 1, 1, D]
        let x_expanded = mlx_rs::ops::expand_dims(x, -2)?;  // [B, L, 1, D]
        let x_expanded = mlx_rs::ops::expand_dims(&x_expanded, -2)?;  // [B, L, 1, 1, D]

        // Use sorting optimization when we have many tokens (indices.size >= 64)
        // This groups tokens by expert for coalesced memory access
        let indices_size = b * l * k;
        let do_sort = indices_size >= 64;

        if do_sort {
            // Sort tokens by expert indices for better memory access
            let (x_sorted, indices_sorted, inv_order) = gather_sort(&x_expanded, indices)?;

            // x_sorted is [B*L*k, 1, D] - no extra expand_dims needed (matching Python)
            // Gate and Up projections with sorted data
            let gate = self.gate_proj.apply(&x_sorted, &indices_sorted, true)?;
            let up = self.up_proj.apply(&x_sorted, &indices_sorted, true)?;

            // SwiGLU activation: silu(gate) * up - using fused Metal kernel
            let activated = fused_swiglu(&up, &gate)?;

            // Down projection
            let output = self.down_proj.apply(&activated, &indices_sorted, true)?;

            // Unsort back to original order
            let output_unsorted = scatter_unsort(&output, &inv_order, &[b as i32, l as i32, k as i32])?;

            // Squeeze: [B, L, k, 1, D] -> [B, L, k, D]
            let shape = output_unsorted.shape();
            output_unsorted.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
        } else {
            // No sorting for small batches
            let gate = self.gate_proj.apply(&x_expanded, indices, false)?;
            let up = self.up_proj.apply(&x_expanded, indices, false)?;

            // SwiGLU activation: silu(gate) * up - using fused Metal kernel
            let activated = fused_swiglu(&up, &gate)?;

            // Down projection
            let output = self.down_proj.apply(&activated, indices, false)?;

            // Squeeze: [B, L, k, 1, D] -> [B, L, k, D]
            let shape = output.shape();
            if shape.len() == 5 {
                output.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
            } else {
                Ok(output)
            }
        }
    }
}

/// Mixture of Experts block
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoE {
    pub num_experts_per_tok: i32,
    pub has_shared_experts: bool,

    #[param]
    pub gate: MoEGate,
    #[param]
    pub switch_mlp: SwitchGLU,
    #[param]
    pub shared_experts: Option<MLP>,
}

impl Module<&Array> for MoE {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Get routing decisions
        let (indices, scores) = self.gate.route(x)?;

        // Apply routed experts
        let expert_out = self.switch_mlp.forward_experts(x, &indices)?;

        // Weight by scores: [B, L, k, D] * [B, L, k, 1] -> sum over k
        // Note: scores are float32 (for sigmoid precision), so convert back to input dtype
        let scores_expanded = scores.index((.., .., .., NewAxis));
        let weighted = expert_out.multiply(&scores_expanded)?;
        let mut y = weighted.sum_axis(2, false)?.as_dtype(x.dtype())?;

        // Add shared experts if present
        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            y = y.add(&shared_out)?;
        }

        Ok(y)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(ref mut shared) = self.shared_experts {
            shared.training_mode(mode);
        }
    }
}

/// Decoder layer (can be dense or MoE)
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub is_moe: bool,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: Option<MLP>,
    #[param]
    pub moe: Option<MoE>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
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

        // MLP or MoE
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };

        h.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        if let Some(ref mut mlp) = self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(ref mut moe) = self.moe {
            moe.training_mode(mode);
        }
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LanguageModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<C>,  // Removed Option wrapper - pre-allocated before generation
}

impl<C> Module<ModelInput<'_, C>> for LanguageModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        // Don't create mask here - let Attention module determine mask mode:
        // - Causal mode for prefill (L > 1) - hardware optimized
        // - No mask for decode (L == 1)
        // Only use explicit mask if provided (e.g., for sliding window)
        let mask = mask.cloned();

        // Cache must be pre-allocated before calling forward
        assert!(!cache.is_empty(), "Cache must be pre-allocated with init_cache()");

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c,  // Direct reference, no Option unwrapping
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: LanguageModel,

    #[quantizable]
    #[param]
    pub lm_head: MaybeQuantized<nn::Linear>,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        self.lm_head.forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        <LanguageModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        self.lm_head.training_mode(mode);
    }
}

// ============================================================================
// Loading functions
// ============================================================================

pub fn load_glm4_moe_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
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

fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let shape = weight.shape();
    let num_experts = shape[0] as i32;
    let output_dims = shape[1] as i32;
    // input_dims is derived from scales
    let scales_shape = scales.shape();
    let input_dims = (scales_shape[2] as i32) * group_size;

    Ok(QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits,
        weight: Param::new(weight),
        scales: Param::new(scales),
        biases: Param::new(biases),
    })
}

fn make_quantized_mlp(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MLP, Error> {
    Ok(MLP {
        gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.gate_proj", prefix), group_size, bits
        )?),
        up_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.up_proj", prefix), group_size, bits
        )?),
        down_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.down_proj", prefix), group_size, bits
        )?),
    })
}

pub fn load_glm4_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::Message("GLM-4.5 MoE requires quantized model".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    eprintln!("Loading weights for {}-bit quantized model...", bits);
    let weights = load_all_weights(model_dir)?;

    let rope_dim = (args.head_dim as f32 * args.partial_rotary_factor) as i32;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let layer_prefix = format!("model.layers.{}", i);
        let is_moe = i >= args.first_k_dense_replace;

        // Build attention
        let attention = Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim: args.head_dim,
            rope_dim,
            scale: (args.head_dim as f32).sqrt().recip(),
            use_qk_norm: args.use_qk_norm,
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
            rope: nn::RopeBuilder::new(rope_dim)
                .base(args.rope_theta)
                .traditional(false)
                .build()
                .unwrap(),
            q_norm: None,
            k_norm: None,
        };

        let (mlp, moe) = if is_moe {
            // Build MoE
            let gate = MoEGate {
                top_k: args.num_experts_per_tok,
                n_routed_experts: args.n_routed_experts,
                routed_scaling_factor: args.routed_scaling_factor,
                norm_topk_prob: args.norm_topk_prob,
                n_group: args.n_group,
                topk_group: args.topk_group,
                weight: Param::new(get_weight(&weights, &format!("{}.mlp.gate.weight", layer_prefix))?),
                e_score_correction_bias: Param::new(get_weight(&weights, &format!("{}.mlp.gate.e_score_correction_bias", layer_prefix))?),
            };

            let switch_mlp = SwitchGLU {
                gate_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.gate_proj", layer_prefix), group_size, bits
                )?,
                up_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.up_proj", layer_prefix), group_size, bits
                )?,
                down_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.down_proj", layer_prefix), group_size, bits
                )?,
            };

            // Shared experts
            let shared_experts = if args.n_shared_experts > 0 {
                Some(make_quantized_mlp(
                    &weights, &format!("{}.mlp.shared_experts", layer_prefix), group_size, bits
                )?)
            } else {
                None
            };

            let moe = MoE {
                num_experts_per_tok: args.num_experts_per_tok,
                has_shared_experts: args.n_shared_experts > 0,
                gate,
                switch_mlp,
                shared_experts,
            };

            (None, Some(moe))
        } else {
            // Dense MLP
            let mlp = make_quantized_mlp(
                &weights, &format!("{}.mlp", layer_prefix), group_size, bits
            )?;
            (Some(mlp), None)
        };

        let layer = DecoderLayer {
            layer_idx: i,
            is_moe,
            self_attn: attention,
            mlp,
            moe,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
        };

        layers.push(layer);
    }

    let language_model = LanguageModel {
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

    let lm_head = MaybeQuantized::Quantized(make_quantized_linear(
        &weights, "lm_head", group_size, bits
    )?);

    let model = Model {
        args,
        model: language_model,
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
    cache: &'a mut Vec<C>,  // Removed Option wrapper
    temp: f32,
    state: GenerateState<'a>,
}

/// Initialize KV cache for a model with the given number of layers
pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,  // Removed Option wrapper
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        // Ensure cache is pre-allocated
        if cache.is_empty() {
            *cache = init_cache(model.model.num_hidden_layers as usize);
        }
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

/// State machine for pipelined token generation (matches Python's async pattern)
pub enum GenerateState<'a> {
    /// Initial state: need to process prompt
    Prefill { prompt_token: &'a Array },
    /// First decode: y computed, need to start pipeline
    FirstDecode { y: Array },
    /// Pipelined decode: current_y ready to return, next_y computing
    Pipelined { current_y: Array },
    /// Finished
    Done,
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    /// Compute the next token given current token
    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
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

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        // Use a dummy value to take ownership of state
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                // Process prompt and get first token
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                // Start async eval and force completion for first token
                tri!(mlx_rs::transforms::async_eval([&y]));
                tri!(mlx_rs::transforms::eval([&y]));  // Force eval like Python does for first token

                // Compute next token and start its async eval
                let next_y = tri!(self.compute_next(&y));
                tri!(mlx_rs::transforms::async_eval([&next_y]));

                // Return first token, store next for pipeline
                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(y))
            }
            GenerateState::FirstDecode { y } => {
                // This state is no longer used - we skip directly to Pipelined
                self.state = GenerateState::Done;
                Some(Ok(y))
            }
            GenerateState::Pipelined { current_y } => {
                // current_y's async_eval was started in previous iteration
                // Compute next token while current_y finalizes
                let next_y = tri!(self.compute_next(&current_y));

                // Start async eval for next token (background computation)
                tri!(mlx_rs::transforms::async_eval([&next_y]));

                // Return current (its async_eval should be done by now)
                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}
