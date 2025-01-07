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