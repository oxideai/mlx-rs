use mlx_rs::{builder::Builder, error::Exception, fast::scaled_dot_product_attention, macros::ModuleParameters, module::Module, nn, ops::concatenate, Array};

pub struct RopeTheta(pub f32);

impl Default for RopeTheta {
    fn default() -> Self {
        RopeTheta(10000.0)
    }
}

pub struct ModelArgs {
    pub dim: i32,
    pub n_layers: i32,
    pub head_dim: i32,
    pub hidden_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: RopeTheta,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    repeats: i32,
    scale: f32,

    #[param]
    wq: nn::Linear,

    #[param]
    wk: nn::Linear,

    #[param]
    wv: nn::Linear,

    #[param]
    wo: nn::Linear,

    #[param]
    rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads;
        let repeats = n_heads / n_kv_heads;
        let scale = (args.head_dim as f32).powf(-0.5);

        let wq = nn::LinearBuilder::new(args.dim, n_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wk = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wv = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wo = nn::LinearBuilder::new(n_heads * args.head_dim, args.dim)
            .bias(false)
            .build()?;
        let rope = nn::RopeBuilder::new(args.head_dim)
            .traditional(true)
            .base(args.rope_theta.0)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats,
            scale,
            wq,
            wk,
            wv,
            wo,
            rope,
        })
    }
}

struct AttentionInput<'a> {
    x: &'a Array,
    mask: Option<&'a Array>,
    kv_cache: Option<(&'a Array, &'a Array)>,
}

struct AttentionOutput {
    output: Array,
    kv_cache: (Array, Array),
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = AttentionOutput;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, kv_cache } = input;

        // NOTE: this will panic if the input shape is not correct
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut queries = self.wq.forward(x)?;
        let mut keys = self.wk.forward(x)?;
        let mut values = self.wv.forward(x)?;

        // Prepare the queries, keys, and values for the attention computation
        queries = queries.reshape(&[B, L, self.n_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;
        keys = keys.reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;
        values = values.reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose(&[0, 2, 1, 3])?;

        match kv_cache {
            Some((key_cache, value_cache)) => {
                let offset = key_cache.shape()[2];
                queries = self.rope.forward((&queries, offset))?;
                keys = self.rope.forward((&keys, offset))?;
                keys = concatenate(&[key_cache, &keys], 2)?;
                values = concatenate(&[value_cache, &values], 2)?;
            },
            None => {
                queries = self.rope.forward(&queries)?;
                keys = self.rope.forward(&keys)?;
            }
        }

        let output = scaled_dot_product_attention(queries, &keys, &values, self.scale, mask, None)?;
        let output = output.transpose(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;
        let output = self.wo.forward(&output)?;

        Ok(AttentionOutput {
            output,
            kv_cache: (keys, values),
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.wq.training_mode(mode);
        self.wk.training_mode(mode);
        self.wv.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct FeedForward {
    #[param]
    w1: nn::Linear,

    #[param]
    w2: nn::Linear,

    #[param]
    w3: nn::Linear,
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
        Ok(Self { w1, w2, w3 })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &'_ Array) -> Result<Self::Output, Self::Error> {
        let w2_input = nn::silu(self.w1.forward(x)?)?
            .multiply(self.w3.forward(x)?)?;
        self.w2.forward(&w2_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.w1.training_mode(mode);
        self.w2.training_mode(mode);
        self.w3.training_mode(mode);
    }
}