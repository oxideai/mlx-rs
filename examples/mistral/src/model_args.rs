use crate::config::Config;


#[derive(Debug, Clone)]
pub struct RopeTheta(pub f32);

impl Default for RopeTheta {
    fn default() -> Self {
        RopeTheta(10000.0)
    }
}

#[derive(Debug, Clone)]
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

impl From<Config> for ModelArgs {
    fn from(value: Config) -> Self {
        Self {
            dim: value.hidden_size,
            n_layers: value.num_hidden_layers,
            head_dim: value.hidden_size / value.num_attention_heads,
            hidden_dim: value.intermediate_size,
            n_heads: value.num_attention_heads,
            n_kv_heads: value.num_key_value_heads,
            norm_eps: value.rms_norm_eps,
            vocab_size: value.vocab_size,
            rope_theta: RopeTheta(value.rope_theta),
        }
    }
}