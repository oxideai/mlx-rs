// {
//     "architectures": [
//         "MistralForCausalLM"
//     ],
//     "bos_token_id": 1,
//     "eos_token_id": 2,
//     "hidden_act": "silu",
//     "hidden_size": 4096,
//     "initializer_range": 0.02,
//     "intermediate_size": 14336,
//     "max_position_embeddings": 32768,
//     "model_type": "mistral",
//     "num_attention_heads": 32,
//     "num_hidden_layers": 32,
//     "num_key_value_heads": 8,
//     "rms_norm_eps": 1e-05,
//     "rope_theta": 10000.0,
//     "sliding_window": 4096,
//     "tie_word_embeddings": false,
//     "torch_dtype": "bfloat16",
//     "transformers_version": "4.34.0.dev0",
//     "use_cache": true,
//     "vocab_size": 32000
// }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub hidden_act: String,
    pub hidden_size: i32,
    pub initializer_range: f32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    pub model_type: String,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i32,
}