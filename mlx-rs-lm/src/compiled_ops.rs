//! Compiled operations for improved kernel fusion
//!
//! This module provides compiled versions of key operations that benefit from
//! kernel fusion when executed repeatedly (e.g., during token generation).
//!
//! The key insight from Python mlx-lm is that `@mx.compile` decorators on
//! internal functions like `group_expert_select` and `swiglu` provide significant
//! speedups by fusing multiple GPU kernel launches into single kernels.

use mlx_rs::{Array, error::Exception};
use mlx_rs::transforms::compile::compile;

// ============================================================================
// Compiled SwiGLU activation
// ============================================================================

/// SwiGLU activation: silu(gate) * x
///
/// This is called 45 times per forward pass in GLM-4.5 MoE (once per MoE layer).
/// Compiling it fuses the silu and multiply operations.
fn swiglu_inner(inputs: &[Array]) -> Result<Vec<Array>, Exception> {
    let x = &inputs[0];
    let gate = &inputs[1];

    let activated = mlx_rs::nn::silu(gate)?;
    let result = activated.multiply(x)?;

    Ok(vec![result])
}

/// Compiled SwiGLU activation
///
/// Uses mlx compile which caches at the C level based on function ID.
pub fn compiled_swiglu(x: &Array, gate: &Array) -> Result<Array, Exception> {
    let inputs = [x.clone(), gate.clone()];

    // Use compile with shapeless=true for variable input sizes
    let mut compiled = compile(swiglu_inner, Some(true));
    let result = compiled(&inputs)?;
    Ok(result.into_iter().next().unwrap())
}

// ============================================================================
// Compiled MoE Routing
// ============================================================================

/// MoE routing parameters
pub struct MoERouteParams {
    pub top_k: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

/// Inner MoE routing function - matches Python's group_expert_select
///
/// inputs[0] = gates (pre-computed: x @ weight.T)
/// inputs[1] = e_score_correction_bias
fn moe_route_inner(inputs: &[Array]) -> Result<Vec<Array>, Exception> {
    let gates = &inputs[0];
    let bias = &inputs[1];

    // Fixed parameters (could be passed via closure if needed)
    let top_k = 8i32;
    let scaling_factor = 1.0f32;
    let norm_topk_prob = true;

    // scores = sigmoid(gates.astype(float32))
    let scores = mlx_rs::ops::sigmoid(&gates.as_dtype(mlx_rs::Dtype::Float32)?)?;
    let orig_scores = scores.clone();

    // scores = scores + bias
    let scores_with_bias = scores.add(bias)?;

    // Top-k selection via argpartition
    let neg_scores = scores_with_bias.negative()?;
    let partitioned = mlx_rs::ops::argpartition_axis(&neg_scores, top_k - 1, -1)?;
    let inds = mlx_rs::ops::indexing::IndexOp::index(&partitioned, (.., .., ..top_k));

    // Select original scores for top-k indices
    let selected = mlx_rs::ops::indexing::take_along_axis(&orig_scores, &inds, -1)?;

    // Normalize if configured
    let final_scores = if norm_topk_prob && top_k > 1 {
        let denom = selected.sum_axis(-1, true)?;
        let normalized = selected.divide(&denom)?;
        normalized.multiply(mlx_rs::array!(scaling_factor))?
    } else {
        selected.multiply(mlx_rs::array!(scaling_factor))?
    };

    Ok(vec![inds, final_scores])
}

/// Compiled MoE routing function
///
/// Returns (expert_indices, expert_weights) for top-k routing.
/// First call compiles; subsequent calls use cached version.
///
/// Note: Uses shapeless=false because slice operations don't support shapeless mode.
pub fn compiled_moe_route(gates: &Array, bias: &Array) -> Result<(Array, Array), Exception> {
    let inputs = [gates.clone(), bias.clone()];

    // Use compiled version with shapeless=false (slice doesn't support shapeless)
    let mut compiled = compile(moe_route_inner, Some(false));
    let result = compiled(&inputs)?;

    Ok((result[0].clone(), result[1].clone()))
}

// ============================================================================
// Compiled full forward step (for use in generation loop)
// ============================================================================

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::compile::compile_with_state;
use crate::cache::CacheState;
use crate::models::glm4_moe::{Model, ModelInput};
use mlx_rs::module::Module;
use std::cell::RefCell;

// Thread-local storage for the model during compiled generation
// This allows using a free function (which is Copy) with compile_with_state
thread_local! {
    static COMPILED_MODEL: RefCell<Option<*mut Model>> = const { RefCell::new(None) };
}

/// Decode step function that can be compiled with compile_with_state
///
/// This function accesses the model via thread-local storage, allowing it to be
/// a free function (which is Copy) as required by compile_with_state.
///
/// SAFETY: The model pointer must remain valid for the duration of the compiled step.
fn decode_step_inner(cache: &mut CacheState, inputs: &[Array]) -> Result<Vec<Array>, Exception> {
    let input = &inputs[0];

    COMPILED_MODEL.with(|model_cell| {
        let model_ptr = model_cell.borrow();
        let model_ptr = model_ptr.expect("Model not set for compiled decode step");

        // SAFETY: We trust that the model pointer is valid (set by CompiledDecodeStep)
        let model = unsafe { &mut *model_ptr };

        let model_input = ModelInput {
            inputs: input,
            mask: None,
            cache: &mut cache.0,
        };

        let logits = model.forward(model_input)?;

        // Get last token logits and argmax
        let last_logits = IndexOp::index(&logits, (.., -1, ..));
        let next_token = mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, true)?;

        Ok(vec![next_token])
    })
}

/// A compiled decode step that fuses the entire forward pass + argmax
///
/// This uses compile_with_state to capture the full computation graph,
/// enabling kernel fusion across the entire decode step.
///
/// # Example
/// ```ignore
/// let mut compiled_step = CompiledDecodeStep::new(&mut model, shapeless);
/// let next_token = compiled_step.step(&mut cache_state, &input)?;
/// ```
pub struct CompiledDecodeStep {
    model_ptr: *mut Model,
    compiled_fn: Box<dyn FnMut(&mut CacheState, &[Array]) -> Result<Vec<Array>, Exception>>,
}

impl CompiledDecodeStep {
    /// Create a new compiled decode step
    ///
    /// # Arguments
    /// * `model` - Mutable reference to the model (must remain valid for lifetime of this struct)
    /// * `shapeless` - Whether to use shapeless compilation (true avoids recompilation on shape change)
    ///
    /// # Safety
    /// The model reference must remain valid for as long as this CompiledDecodeStep is used.
    pub fn new(model: &mut Model, shapeless: bool) -> Self {
        let model_ptr = model as *mut Model;

        // Set up the thread-local model pointer
        COMPILED_MODEL.with(|cell| {
            *cell.borrow_mut() = Some(model_ptr);
        });

        // Create the compiled function
        // Note: shapeless=false is safer as some operations don't support shapeless
        let compiled_fn = compile_with_state(decode_step_inner, Some(shapeless));

        Self {
            model_ptr,
            compiled_fn: Box::new(compiled_fn),
        }
    }

    /// Execute one decode step
    ///
    /// Takes the current token(s) and returns the next token via argmax.
    pub fn step(&mut self, cache: &mut CacheState, input: &Array) -> Result<Array, Exception> {
        // Ensure thread-local model is set (in case of thread changes)
        COMPILED_MODEL.with(|cell| {
            *cell.borrow_mut() = Some(self.model_ptr);
        });

        let inputs = [input.clone()];
        let result = (self.compiled_fn)(cache, &inputs)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl Drop for CompiledDecodeStep {
    fn drop(&mut self) {
        // Clear the thread-local model pointer
        COMPILED_MODEL.with(|cell| {
            *cell.borrow_mut() = None;
        });
    }
}

/// Compiled generation step that includes forward pass and argmax sampling
///
/// This captures the entire decode computation graph for maximum fusion.
/// (Legacy version without compile_with_state - kept for comparison)
pub fn create_compiled_step<F>(
    mut forward_fn: F,
) -> impl FnMut(&Array) -> Result<Array, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    move |input: &Array| {
        let logits = forward_fn(input)?;
        // Get last token logits and argmax
        let last_logits = IndexOp::index(&logits, (.., -1, ..));
        mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_swiglu() {
        let x = mlx_rs::random::normal::<f32>(&[1, 1, 4096], None, None, None).unwrap();
        let gate = mlx_rs::random::normal::<f32>(&[1, 1, 4096], None, None, None).unwrap();

        let result = compiled_swiglu(&x, &gate).unwrap();
        assert_eq!(result.shape(), x.shape());
    }

    #[test]
    fn test_moe_route() {
        let gates = mlx_rs::random::normal::<f32>(&[1, 1, 64], None, None, None).unwrap();
        let bias = mlx_rs::Array::zeros::<f32>(&[64]).unwrap();

        let (inds, scores) = compiled_moe_route(&gates, &bias).unwrap();

        // Should return top-8 indices and scores
        assert_eq!(inds.shape()[2], 8);
        assert_eq!(scores.shape()[2], 8);
    }
}
