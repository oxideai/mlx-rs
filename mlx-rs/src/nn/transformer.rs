use std::borrow::Cow;

use dyn_clone::DynClone;
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;
use crate::{
    error::Exception,
    module::Module,
    ops::{matmul, softmax},
    prelude::Builder,
    Array,
};

use crate::nn::{
    error::{MultiHeadAttentionBuildError, TransformerBulidError},
    Dropout, DropoutBuilder, LayerNorm, Linear, LinearBuilder, Relu,
};

/// A marker trait for activation functions used in transformers.
pub trait Activation: std::fmt::Debug
where
    for<'a> Self: Module<&'a Array, Output = Array, Error = Exception> + DynClone,
{
}

impl<M> Activation for M where
    for<'a> M: Module<&'a Array, Output = Array, Error = Exception> + std::fmt::Debug + DynClone
{
}

/// Builder for the [`MultiHeadAttention`] module
#[derive(Debug, Clone, Builder)]
#[builder(
    build_with = build_multi_head_attention,
    err = MultiHeadAttentionBuildError,
    root = crate
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

fn build_multi_head_attention(
    builder: MultiHeadAttentionBuilder,
) -> Result<MultiHeadAttention, MultiHeadAttentionBuildError> {
    if builder.dims % builder.num_heads != 0 {
        return Err(MultiHeadAttentionBuildError::InvalidNumHeads(
            builder.num_heads,
        ));
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
#[module(root = crate)]
#[buildable(root = crate)]
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

impl<'a> From<(&'a Array, &'a Array, &'a Array, Option<&'a Array>)>
    for MultiHeadAttentionInput<'a>
{
    fn from(
        (queries, keys, values, mask): (&'a Array, &'a Array, &'a Array, Option<&'a Array>),
    ) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask,
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
    fn forward(&self, input: Input) -> Result<Self::Output, Self::Error> {
        let input = input.into();

        let queries = self.query_proj.forward(input.queries)?;
        let keys = self.key_proj.forward(input.keys)?;
        let values = self.value_proj.forward(input.values)?;

        let B = queries.dim(0);
        let L = queries.dim(1);
        let S = keys.dim(1);

        let queries = queries
            .reshape(&[B, L, self.num_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[B, S, self.num_heads, -1])?
            .transpose(&[0, 2, 3, 1])?;
        let values = values
            .reshape(&[B, S, self.num_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;

        // Dimensions are [batch x num_heads x sequence x hidden_dim]
        let scale = f32::sqrt(1.0 / queries.dim(-1) as f32);
        let mut scores = (queries * scale).matmul(&keys)?;
        if let Some(mask) = input.mask {
            scores = scores.add(mask.as_dtype(scores.dtype())?)?;
        }
        scores = softmax(&scores, &[-1], None)?;
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

#[derive(Debug, Builder)]
#[builder(
    build_with = build_transformer_encoder_layer,
    err = TransformerBulidError,
    root = crate
)]
struct TransformerEncoderLayerBuilder {
    pub dimensions: i32,
    pub num_heads: i32,

    #[builder(optional, default = None)]
    pub mlp_dimensions: Option<i32>,

    #[builder(optional, default = Self::DEFAULT_DROPOUT)]
    pub dropout: f32,

    #[builder(optional, default = None)]
    pub activation: Option<Box<dyn Activation>>,

    pub norm_first: bool,
}

impl Clone for TransformerEncoderLayerBuilder {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            num_heads: self.num_heads,
            mlp_dimensions: self.mlp_dimensions,
            dropout: self.dropout,
            activation: self
                .activation
                .as_ref()
                .map(|a| dyn_clone::clone_box(a.as_ref())),
            norm_first: self.norm_first,
        }
    }
}

// The const are placed in the builder because the encoder layer is not public anyway
impl TransformerEncoderLayerBuilder {
    const DEFAULT_DROPOUT: f32 = 0.0;
}

fn build_transformer_encoder_layer(
    builder: TransformerEncoderLayerBuilder,
) -> Result<TransformerEncoderLayer, TransformerBulidError> {
    let dimensions = builder.dimensions;
    let num_heads = builder.num_heads;
    let mlp_dimensions = builder.mlp_dimensions.unwrap_or(4 * dimensions);
    let dropout = builder.dropout;
    let attention = MultiHeadAttention::new(dimensions, num_heads)?;
    let ln1 = LayerNorm::new(dimensions)?;
    let ln2 = LayerNorm::new(dimensions)?;
    let linear1 = Linear::new(dimensions, mlp_dimensions)?;
    let linear2 = Linear::new(mlp_dimensions, dimensions)?;
    let dropout1 = DropoutBuilder::new().p(dropout).build()?;
    let dropout2 = DropoutBuilder::new().p(dropout).build()?;
    let activation = builder.activation.unwrap_or(Box::new(Relu));
    let norm_first = builder.norm_first;

    Ok(TransformerEncoderLayer {
        attention,
        ln1,
        ln2,
        linear1,
        linear2,
        dropout1,
        dropout2,
        activation,
        norm_first,
    })
}

/// Transformer encoder layer.
#[derive(Debug, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
struct TransformerEncoderLayer {
    /// Multi-head attention module
    #[param]
    pub attention: MultiHeadAttention,

    /// First layer norm module
    #[param]
    pub ln1: LayerNorm,

    /// Second layer norm module
    #[param]
    pub ln2: LayerNorm,

    /// First linear module
    #[param]
    pub linear1: Linear,

    /// Second linear module
    #[param]
    pub linear2: Linear,

    /// Dropout module for the first layer
    #[param]
    pub dropout1: Dropout,

    /// Dropout module for the second layer
    #[param]
    pub dropout2: Dropout,

    /// Activation function
    #[param]
    pub activation: Box<dyn Activation>,

    /// If `true`, apply the layer norm before the first linear layer
    pub norm_first: bool,
}

impl Clone for TransformerEncoderLayer {
    fn clone(&self) -> Self {
        Self {
            attention: self.attention.clone(),
            ln1: self.ln1.clone(),
            ln2: self.ln2.clone(),
            linear1: self.linear1.clone(),
            linear2: self.linear2.clone(),
            dropout1: self.dropout1.clone(),
            dropout2: self.dropout2.clone(),
            activation: dyn_clone::clone_box(&*self.activation),
            norm_first: self.norm_first,
        }
    }
}

struct TransformerEncoderInput<'a> {
    pub x: &'a Array,
    pub mask: &'a Array,
}

impl<'a> From<(&'a Array, &'a Array)> for TransformerEncoderInput<'a> {
    fn from((x, mask): (&'a Array, &'a Array)) -> Self {
        TransformerEncoderInput { x, mask }
    }
}

impl<'a, T> Module<T> for TransformerEncoderLayer
where
    T: Into<TransformerEncoderInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&self, input: T) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let x = input.x;
        let mask = input.mask;

        if self.norm_first {
            let mut y = self.ln1.forward(x)?;
            y = self.attention.forward((&y, &y, &y, mask))?;
            y = self.dropout1.forward(&y)?;
            let x = x.add(&y)?;

            y = self.ln2.forward(&x)?;
            y = self.linear1.forward(&y)?;
            y = self.activation.forward(&y)?;
            y = self.dropout2.forward(&y)?;
            y = self.linear2.forward(&y)?;
            y = x.add(&y)?;

            Ok(y)
        } else {
            let mut y = self.attention.forward((x, x, x, mask))?;
            y = self.dropout1.forward(&y)?;
            let mut x = x.add(&y)?;
            x = self.ln1.forward(&x)?;

            y = self.linear1.forward(&x)?;
            y = self.activation.forward(&y)?;
            y = self.dropout2.forward(&y)?;
            y = self.linear2.forward(&y)?;
            y = x.add(&y)?;
            y = self.ln2.forward(&y)?;

            Ok(y)
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <MultiHeadAttention as Module<MultiHeadAttentionInput<'a>>>::training_mode(
            &mut self.attention,
            mode,
        );
        self.ln1.training_mode(mode);
        self.ln2.training_mode(mode);
        self.linear1.training_mode(mode);
        self.linear2.training_mode(mode);
        self.dropout1.training_mode(mode);
        self.dropout2.training_mode(mode);
        self.activation.training_mode(mode);
    }
}

#[derive(Debug, Builder)]
#[builder(
    build_with = build_transformer_encoder,
    err = TransformerBulidError,
    root = crate
)]
struct TransformerEncoderBuilder {
    pub layer_count: usize,
    pub dimensions: i32,
    pub num_heads: i32,

    #[builder(optional, default = None)]
    pub mlp_dimensions: Option<i32>,

    #[builder(optional, default = Self::DEFAULT_DROPOUT)]
    pub dropout: f32,

    #[builder(optional, default = None)]
    pub activation: Option<Box<dyn Activation>>,

    pub norm_first: bool,
}

impl TransformerEncoderBuilder {
    const DEFAULT_DROPOUT: f32 = 0.0;
}

impl Clone for TransformerEncoderBuilder {
    fn clone(&self) -> Self {
        Self {
            layer_count: self.layer_count,
            dimensions: self.dimensions,
            num_heads: self.num_heads,
            mlp_dimensions: self.mlp_dimensions,
            dropout: self.dropout,
            activation: self
                .activation
                .as_ref()
                .map(|a| dyn_clone::clone_box(a.as_ref())),
            norm_first: self.norm_first,
        }
    }
}

fn build_transformer_encoder(
    builder: TransformerEncoderBuilder,
) -> Result<TransformerEncoder, TransformerBulidError> {
    let layer_count = builder.layer_count;
    let dimensions = builder.dimensions;
    let num_heads = builder.num_heads;
    let norm_first = builder.norm_first;
    let activation = builder.activation.unwrap_or(Box::new(Relu));

    let layers = (0..layer_count)
        .map(|_| {
            TransformerEncoderLayerBuilder::new(dimensions, num_heads, norm_first)
                .mlp_dimensions(builder.mlp_dimensions)
                .dropout(builder.dropout)
                .activation(dyn_clone::clone_box(&*activation))
                .build()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ln = LayerNorm::new(dimensions)?;

    Ok(TransformerEncoder { layers, ln })
}

#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
struct TransformerEncoder {
    #[param]
    pub layers: Vec<TransformerEncoderLayer>,

    #[param]
    pub ln: LayerNorm,
}

impl<'a, T> Module<T> for TransformerEncoder
where
    T: Into<TransformerEncoderInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&self, input: T) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let x = input.x;
        let mask = input.mask;

        let mut x = Cow::Borrowed(x);

        for l in &self.layers {
            x = Cow::Owned(l.forward((&*x, mask))?);
        }

        self.ln.forward(&*x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers.iter_mut().for_each(|layer| {
            <TransformerEncoderLayer as Module<TransformerEncoderInput>>::training_mode(
                layer, mode,
            );
        });
        self.ln.training_mode(mode);
    }
}

#[derive(Debug, Builder)]
#[builder(
    build_with = build_transformer_decoder_layer,
    err = TransformerBulidError,
    root = crate
)]
struct TransformerDecoderLayerBuilder {
    pub dimensions: i32,
    pub num_heads: i32,
    #[builder(optional, default = None)]
    pub ml_dimensions: Option<i32>,
    #[builder(optional, default = Self::DEFAULT_DROPOUT)]
    pub dropout: f32,
    #[builder(optional, default = None)]
    pub activation: Option<Box<dyn Activation>>,
    pub norm_first: bool,
}

impl TransformerDecoderLayerBuilder {
    const DEFAULT_DROPOUT: f32 = 0.0;
}

impl Clone for TransformerDecoderLayerBuilder {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            num_heads: self.num_heads,
            ml_dimensions: self.ml_dimensions,
            dropout: self.dropout,
            activation: self
                .activation
                .as_ref()
                .map(|a| dyn_clone::clone_box(a.as_ref())),
            norm_first: self.norm_first,
        }
    }
}

fn build_transformer_decoder_layer(
    builder: TransformerDecoderLayerBuilder,
) -> Result<TransformerDecoderLayer, TransformerBulidError> {
    let dimensions = builder.dimensions;
    let num_heads = builder.num_heads;
    let mlp_dimensions = builder.ml_dimensions.unwrap_or(4 * dimensions);
    let dropout = builder.dropout;

    let self_attention = MultiHeadAttention::new(dimensions, num_heads)?;
    let cross_attention = MultiHeadAttention::new(dimensions, num_heads)?;
    let ln1 = LayerNorm::new(dimensions)?;
    let ln2 = LayerNorm::new(dimensions)?;
    let ln3 = LayerNorm::new(dimensions)?;
    let linear1 = Linear::new(dimensions, mlp_dimensions)?;
    let linear2 = Linear::new(mlp_dimensions, dimensions)?;
    let dropout1 = DropoutBuilder::new().p(dropout).build()?;
    let dropout2 = DropoutBuilder::new().p(dropout).build()?;
    let dropout3 = DropoutBuilder::new().p(dropout).build()?;
    let activation = builder.activation.unwrap_or(Box::new(Relu));
    let norm_first = builder.norm_first;

    Ok(TransformerDecoderLayer {
        self_attention,
        cross_attention,
        ln1,
        ln2,
        ln3,
        linear1,
        linear2,
        dropout1,
        dropout2,
        dropout3,
        activation,
        norm_first,
    })
}

#[derive(Debug, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
struct TransformerDecoderLayer {
    #[param]
    pub self_attention: MultiHeadAttention,

    #[param]
    pub cross_attention: MultiHeadAttention,

    #[param]
    pub ln1: LayerNorm,

    #[param]
    pub ln2: LayerNorm,

    #[param]
    pub ln3: LayerNorm,

    #[param]
    pub linear1: Linear,

    #[param]
    pub linear2: Linear,

    #[param]
    pub dropout1: Dropout,

    #[param]
    pub dropout2: Dropout,

    #[param]
    pub dropout3: Dropout,

    #[param]
    pub activation: Box<dyn Activation>,

    pub norm_first: bool,
}

impl Clone for TransformerDecoderLayer {
    fn clone(&self) -> Self {
        Self {
            self_attention: self.self_attention.clone(),
            cross_attention: self.cross_attention.clone(),
            ln1: self.ln1.clone(),
            ln2: self.ln2.clone(),
            ln3: self.ln3.clone(),
            linear1: self.linear1.clone(),
            linear2: self.linear2.clone(),
            dropout1: self.dropout1.clone(),
            dropout2: self.dropout2.clone(),
            dropout3: self.dropout3.clone(),
            activation: dyn_clone::clone_box(&*self.activation),
            norm_first: self.norm_first,
        }
    }
}

struct TransformerDecoderInput<'a> {
    pub x: &'a Array,
    pub memory: &'a Array,
    pub x_mask: &'a Array,
    pub memory_mask: &'a Array,
}

impl<'a> From<(&'a Array, &'a Array, &'a Array, &'a Array)> for TransformerDecoderInput<'a> {
    fn from(
        (x, memory, x_mask, memory_mask): (&'a Array, &'a Array, &'a Array, &'a Array),
    ) -> Self {
        TransformerDecoderInput {
            x,
            memory,
            x_mask,
            memory_mask,
        }
    }
}

impl<'a, T> Module<T> for TransformerDecoderLayer
where
    T: Into<TransformerDecoderInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&self, input: T) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let x = input.x;
        let memory = input.memory;
        let x_mask = input.x_mask;
        let memory_mask = input.memory_mask;

        if self.norm_first {
            let mut y = self.ln1.forward(x)?;
            y = self.self_attention.forward((&y, &y, &y, x_mask))?;
            y = self.dropout1.forward(&y)?;
            let x = x.add(&y)?;

            y = self.ln2.forward(&x)?;
            y = self
                .cross_attention
                .forward((&y, memory, memory, memory_mask))?;
            y = self.dropout2.forward(&y)?;
            let x = x.add(&y)?;

            y = self.ln3.forward(&x)?;
            y = self.linear1.forward(&y)?;
            y = self.activation.forward(&y)?;
            y = self.dropout3.forward(&y)?;
            y = self.linear2.forward(&y)?;
            x.add(&y)
        } else {
            let mut y = self.self_attention.forward((x, x, x, x_mask))?;
            y = self.dropout1.forward(&y)?;
            let mut x = x.add(&y)?;
            x = self.ln1.forward(&x)?;

            y = self
                .cross_attention
                .forward((&y, memory, memory, memory_mask))?;
            y = self.dropout2.forward(&y)?;
            x = x.add(&y)?;
            x = self.ln2.forward(&x)?; // TODO: https://github.com/ml-explore/mlx/issues/1636

            y = self.linear1.forward(&x)?;
            y = self.activation.forward(&y)?;
            y = self.dropout3.forward(&y)?;
            y = self.linear2.forward(&y)?;
            y = x.add(&y)?;
            self.ln3.forward(&y)
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <MultiHeadAttention as Module<MultiHeadAttentionInput>>::training_mode(
            &mut self.self_attention,
            mode,
        );
        <MultiHeadAttention as Module<MultiHeadAttentionInput>>::training_mode(
            &mut self.cross_attention,
            mode,
        );
        self.ln1.training_mode(mode);
        self.ln2.training_mode(mode);
        self.ln3.training_mode(mode);
        self.linear1.training_mode(mode);
        self.linear2.training_mode(mode);
        self.dropout1.training_mode(mode);
        self.dropout2.training_mode(mode);
        self.dropout3.training_mode(mode);
        self.activation.training_mode(mode);
    }
}

#[derive(Debug, Builder)]
#[builder(
    build_with = build_transformer_decoder,
    err = TransformerBulidError,
    root = crate
)]
struct TransformerDecoderBuilder {
    pub layer_count: usize,
    pub dimensions: i32,
    pub num_heads: i32,

    #[builder(optional, default = None)]
    pub mlp_dimensions: Option<i32>,

    #[builder(optional, default = Self::DEFAULT_DROPOUT)]
    pub dropout: f32,

    #[builder(optional, default = None)]
    pub activation: Option<Box<dyn Activation>>,

    pub norm_first: bool,
}

impl TransformerDecoderBuilder {
    const DEFAULT_DROPOUT: f32 = 0.0;
}

impl Clone for TransformerDecoderBuilder {
    fn clone(&self) -> Self {
        Self {
            layer_count: self.layer_count,
            dimensions: self.dimensions,
            num_heads: self.num_heads,
            mlp_dimensions: self.mlp_dimensions,
            dropout: self.dropout,
            activation: self
                .activation
                .as_ref()
                .map(|a| dyn_clone::clone_box(a.as_ref())),
            norm_first: self.norm_first,
        }
    }
}

fn build_transformer_decoder(
    builder: TransformerDecoderBuilder,
) -> Result<TransformerDecoder, TransformerBulidError> {
    let layer_count = builder.layer_count;
    let dimensions = builder.dimensions;
    let num_heads = builder.num_heads;
    let norm_first = builder.norm_first;

    let activation = builder.activation.unwrap_or(Box::new(Relu));

    let layers = (0..layer_count)
        .map(|_| {
            TransformerDecoderLayerBuilder::new(dimensions, num_heads, norm_first)
                .ml_dimensions(builder.mlp_dimensions)
                .dropout(builder.dropout)
                .activation(dyn_clone::clone_box(&*activation))
                .build()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ln = LayerNorm::new(dimensions)?;

    Ok(TransformerDecoder { layers, ln })
}

#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
struct TransformerDecoder {
    #[param]
    pub layers: Vec<TransformerDecoderLayer>,

    #[param]
    pub ln: LayerNorm,
}

impl<'a, T> Module<T> for TransformerDecoder
where
    T: Into<TransformerDecoderInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&self, input: T) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let x = input.x;
        let memory = input.memory;
        let x_mask = input.x_mask;
        let memory_mask = input.memory_mask;

        let mut x = Cow::Borrowed(x);

        for l in &self.layers {
            x = Cow::Owned(l.forward((&*x, memory, x_mask, memory_mask))?);
        }

        self.ln.forward(&*x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers.iter_mut().for_each(|layer| {
            <TransformerDecoderLayer as Module<TransformerDecoderInput>>::training_mode(
                layer, mode,
            );
        });
        self.ln.training_mode(mode);
    }
}

/// Builder for the [`Transformer`] module
#[derive(Debug, Builder)]
#[builder(
    build_with = build_transformer,
    err = TransformerBulidError,
    root = crate
)]
pub struct TransformerBuilder {
    /// number of expected features in the encoder/decoder
    #[builder(optional, default = Transformer::DEFAULT_DIMENSIONS)]
    pub dimensions: i32,

    /// number of attention heads
    #[builder(optional, default = Transformer::DEFAULT_NUM_HEADS)]
    pub num_heads: i32,

    /// number of layers in the encoder
    #[builder(optional, default = Transformer::DEFAULT_ENCODER_LAYERS_COUNT)]
    pub encoder_layer_count: usize,

    /// number of layers in the decoder
    #[builder(optional, default = Transformer::DEFAULT_DECODER_LAYERS_COUNT)]
    pub decoder_layer_count: usize,

    /// hidden dimensions of the MLP block in each layer. Defaults to `4 * dimensions`
    /// if not specified
    #[builder(optional, default = None)]
    pub mlp_dimensions: Option<i32>,

    /// dropout value for the encode and decoder. Dropout is used after each attention layer
    /// and the activation in the MLP layer
    #[builder(optional, default = Transformer::DEFAULT_DROPOUT)]
    pub dropout: f32,

    /// the activation layer for the MLP hidden layer
    #[builder(optional, default = None)]
    pub activation: Option<Box<dyn Activation>>,

    /// if `true` encode and decoder layers will perform layer normalization before
    /// attention and MLP operations, otherwise after
    #[builder(optional, default = Transformer::DEFAULT_NORM_FIRST)]
    pub norm_first: bool,
}

impl Clone for TransformerBuilder {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            num_heads: self.num_heads,
            encoder_layer_count: self.encoder_layer_count,
            decoder_layer_count: self.decoder_layer_count,
            mlp_dimensions: self.mlp_dimensions,
            dropout: self.dropout,
            activation: self
                .activation
                .as_ref()
                .map(|a| dyn_clone::clone_box(a.as_ref())),
            norm_first: self.norm_first,
        }
    }
}

fn build_transformer(builder: TransformerBuilder) -> Result<Transformer, TransformerBulidError> {
    let dimensions = builder.dimensions;
    let num_heads = builder.num_heads;
    let encoder_layer_count = builder.encoder_layer_count;
    let decoder_layer_count = builder.decoder_layer_count;
    let mlp_dimensions = builder.mlp_dimensions;
    let dropout = builder.dropout;
    let activation = builder.activation.unwrap_or(Box::new(Relu));
    let norm_first = builder.norm_first;

    let encoder =
        TransformerEncoderBuilder::new(encoder_layer_count, dimensions, num_heads, norm_first)
            .mlp_dimensions(mlp_dimensions)
            .dropout(dropout)
            .activation(dyn_clone::clone_box(&*activation))
            .build()?;
    let decoder =
        TransformerDecoderBuilder::new(decoder_layer_count, dimensions, num_heads, norm_first)
            .mlp_dimensions(mlp_dimensions)
            .dropout(dropout)
            .activation(dyn_clone::clone_box(&*activation))
            .build()?;

    Ok(Transformer { encoder, decoder })
}

/// Implements a standard Transformer model.
///
/// The implementation is based on "Attention Is All You Need"
/// <https://arxiv.org/abs/1706.03762>.
///
/// The Transformer model contains an encoder and a decoder. The encoder
/// processes the input sequence and the decoder generates the output sequence.
/// The interaction between encoder and decoder happens through the attention
/// mechanism.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Transformer {
    /// Encoder module
    #[param]
    encoder: TransformerEncoder, // TODO: visibility?

    /// Decoder module
    #[param]
    decoder: TransformerDecoder, // TODO: visibility?
}

impl Transformer {
    /// Default value for `dimensions`
    pub const DEFAULT_DIMENSIONS: i32 = 512;

    /// Default value for `num_heads`
    pub const DEFAULT_NUM_HEADS: i32 = 8;

    /// Default number of encoder layers
    pub const DEFAULT_ENCODER_LAYERS_COUNT: usize = 6;

    /// Default number of decoder layers
    pub const DEFAULT_DECODER_LAYERS_COUNT: usize = 6;

    /// Default value for dropout
    pub const DEFAULT_DROPOUT: f32 = 0.0;

    /// Default value for `activation`
    pub const DEFAULT_NORM_FIRST: bool = false;
}

/// Input to the [`Transformer`] module
#[derive(Debug, Clone)]
pub struct TransformerInput<'a> {
    /// Source
    pub source: &'a Array,

    /// Target
    pub target: &'a Array,

    /// Source mask
    pub source_mask: &'a Array,

    /// Target mask
    pub target_mask: &'a Array,

    /// Memory mask
    pub memory_mask: &'a Array,
}

impl<'a> From<(&'a Array, &'a Array, &'a Array, &'a Array, &'a Array)> for TransformerInput<'a> {
    fn from(
        (source, target, source_mask, target_mask, memory_mask): (
            &'a Array,
            &'a Array,
            &'a Array,
            &'a Array,
            &'a Array,
        ),
    ) -> Self {
        TransformerInput {
            source,
            target,
            source_mask,
            target_mask,
            memory_mask,
        }
    }
}

impl<'a, T> Module<T> for Transformer
where
    T: Into<TransformerInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&self, input: T) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let source = input.source;
        let target = input.target;
        let source_mask = input.source_mask;
        let target_mask = input.target_mask;
        let memory_mask = input.memory_mask;

        let memory = self.encoder.forward((source, source_mask))?;
        self.decoder
            .forward((target, &memory, target_mask, memory_mask))
    }

    fn training_mode(&mut self, mode: bool) {
        <TransformerEncoder as Module<TransformerEncoderInput>>::training_mode(
            &mut self.encoder,
            mode,
        );
        <TransformerDecoder as Module<TransformerDecoderInput>>::training_mode(
            &mut self.decoder,
            mode,
        );
    }
}
