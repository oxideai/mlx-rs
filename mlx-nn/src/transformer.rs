use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;
use mlx_rs::{error::Exception, module::Module, ops::{matmul, softmax}, prelude::Builder, Array};

use crate::{error::MultiHeadAttentionBuildError, Linear, LinearBuilder};

/// Builder for the [`MultiHeadAttention`] module
#[derive(Debug, Clone, Builder)]
#[builder(
    build_with = build_multi_head_attention,
    err = MultiHeadAttentionBuildError,
)]
pub struct MultiHeadAttentionBuilder {
    /// Model dimensions and default for the other dimensions if they are not supplied
    pub dims: i32,

    /// Number of attention heads
    pub num_heads: i32,
    
    /// Input dimensions of queries
    #[builder(optional, default = None)]
    pub query_input_dims: Option<i32>,
    
    /// Input dimensions of keys
    #[builder(optional, default = None)]
    pub key_input_dims: Option<i32>,
    
    /// Input dimensions of values
    #[builder(optional, default = None)]
    pub value_input_dims: Option<i32>,
    
    /// Dimensions of values after the projection
    #[builder(optional, default = None)]
    pub value_dims: Option<i32>,
    
    /// Dimensions new values will be projected to
    #[builder(optional, default = None)]
    pub value_output_dims: Option<i32>,
    
    /// If `true`, use a bias in the [`Linear`] layers
    #[builder(optional, default = MultiHeadAttention::DEFAULT_BIAS)]
    pub bias: bool,
}

fn build_multi_head_attention(builder: MultiHeadAttentionBuilder) -> Result<MultiHeadAttention, MultiHeadAttentionBuildError> {
    if builder.dims % builder.num_heads != 0 {
        return Err(MultiHeadAttentionBuildError::InvalidNumHeads(builder.num_heads));
    }

    let dims = builder.dims;
    let bias = builder.bias;
    let query_input_dims = builder.query_input_dims.unwrap_or(builder.dims);
    let key_input_dims = builder.key_input_dims.unwrap_or(builder.dims);
    let value_input_dims = builder.value_input_dims.unwrap_or(builder.dims);
    let value_dims = builder.value_dims.unwrap_or(builder.dims);
    let value_output_dims = builder.value_output_dims.unwrap_or(builder.dims);

    let num_heads = builder.num_heads;

    let query_proj = LinearBuilder::new(query_input_dims, dims)
        .bias(bias)
        .build()?;
    let key_proj = LinearBuilder::new(key_input_dims, dims)
        .bias(bias)
        .build()?;
    let value_proj = LinearBuilder::new(value_input_dims, value_dims)
        .bias(bias)
        .build()?;
    let output_proj = LinearBuilder::new(value_dims, value_output_dims)
        .bias(bias)
        .build()?;

    Ok(MultiHeadAttention {
        num_heads,
        query_proj,
        key_proj,
        value_proj,
        output_proj,
    })
}

/// Implements the scaled dot product attention with multiple heads.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    pub num_heads: i32,

    /// Query projection layer
    #[param]
    pub query_proj: Linear,

    /// Key projection layer
    #[param]
    pub key_proj: Linear,

    /// Value projection layer
    #[param]
    pub value_proj: Linear,

    /// Output projection layer
    #[param]
    pub output_proj: Linear,
}

impl MultiHeadAttention {
    /// Default value for the `bias` field
    pub const DEFAULT_BIAS: bool = false;
}

/// Input to the [`MultiHeadAttention`] module
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionInput<'a> {
    /// Queries
    pub queries: &'a Array,

    /// Keys
    pub keys: &'a Array,

    /// Values
    pub values: &'a Array,

    /// Mask
    pub mask: Option<&'a Array>,
}

impl<'a> From<(&'a Array, &'a Array, &'a Array)> for MultiHeadAttentionInput<'a> {
    fn from((queries, keys, values): (&'a Array, &'a Array, &'a Array)) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask: None,
        }
    }
}

impl<'a> From<(&'a Array, &'a Array, &'a Array, &'a Array)> for MultiHeadAttentionInput<'a> {
    fn from((queries, keys, values, mask): (&'a Array, &'a Array, &'a Array, &'a Array)) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask: Some(mask),
        }
    }
}

impl<'a, Input> Module<Input> for MultiHeadAttention 
where 
    Input: Into<MultiHeadAttentionInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let input = input.into();

        let queries = self.query_proj.forward(input.queries)?;
        let keys = self.key_proj.forward(input.keys)?;
        let values = self.value_proj.forward(input.values)?;

        let B = queries.dim(0);
        let L = queries.dim(1);
        let S = keys.dim(1);

        let queries = queries.reshape(&[B, L, self.num_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = keys.reshape(&[B, S, self.num_heads, -1])?
            .transpose(&[0, 2, 3, 1])?;
        let values = values.reshape(&[B, S, self.num_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;

        // Dimensions are [batch x num_heads x sequence x hidden_dim]
        let scale = f32::sqrt(1.0 / queries.dim(-1) as f32);
        let mut scores = (queries * scale).matmul(&keys)?;
        if let Some(mask) = input.mask {
            scores = scores.add(mask.as_dtype(scores.dtype()))?;
        }
        scores = softmax(&scores, &[-1], None);
        let value_hat = matmul(&scores, &values)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;
            
        self.output_proj.forward(&value_hat)
    }

    fn training_mode(&mut self, mode: bool) {
        self.query_proj.training_mode(mode);
        self.key_proj.training_mode(mode);
        self.value_proj.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}