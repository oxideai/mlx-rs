use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Repo,
};
use mlx_rs::{
    array,
    module::{Module, ModuleParametersExt},
    ops::indexing::{argmax_axis, IndexOp, NewAxis},
    random::categorical,
    transforms::{eval, async_eval},
    Array, Stream,
};
use tokenizers::Tokenizer;

mod model;

use model::{Mistral, MistralInput, MistralOutput, ModelArgs, load_model_quantized};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Mistral inference example")]
pub struct Cli {
    /// The message to be processed by the model
    #[clap(long, default_value = "In the begging the Unverse was created.")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[clap(long, default_value = "100")]
    max_tokens: usize,

    /// The sampling temperature
    #[clap(long, default_value = "0.0")]
    temp: f32,

    /// The batch size of tokens to generate
    #[clap(long, default_value = "10")]
    tokens_per_eval: usize,

    /// The PRNG seed
    #[clap(long, default_value = "0")]
    seed: u64,
}

fn build_hf_api() -> Result<Api> {
    let cache_dir = std::env::var("HF_CACHE_DIR").ok();

    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    builder.build().map_err(Into::into)
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let t = Tokenizer::from_file(tokenizer_filename)?;

    Ok(t)
}

fn get_model_args(repo: &ApiRepo) -> Result<ModelArgs> {
    let model_args_filename = repo.get("config.json")?;
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

fn download_weights(repo: &ApiRepo) -> Result<std::path::PathBuf> {
    // Download config first
    let config_path = repo.get("config.json")?;

    // Check for sharded weights
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;
        if let Some(weight_map) = index["weight_map"].as_object() {
            let weight_files: std::collections::HashSet<&str> = weight_map.values()
                .filter_map(|v| v.as_str())
                .collect();
            for weight_file in &weight_files {
                let _ = repo.get(weight_file)?;
            }
        }
    } else {
        // Single file model
        let _ = repo.get("model.safetensors")?;
    }

    Ok(config_path.parent().unwrap().to_path_buf())
}

fn sample(logits: &Array, temp: f32) -> Result<Array> {
    match temp {
        0.0 => argmax_axis(logits, -1, None).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical(logits, None, None, None).map_err(Into::into)
        }
    }
}

fn synchronize(stream: &Stream) {
    unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()); }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

struct Generate<'a> {
    model: &'a mut Mistral,
    temp: f32,
    state: GenerateState<'a>,
}

enum GenerateState<'a> {
    /// Initial state: need to process prompt
    Prefill {
        prompt_token: &'a Array,
    },
    /// Pipelined decode: current_y ready to return, next is computing
    Pipelined {
        current_y: Array,
        cache: Vec<Option<(Array, Array)>>,
    },
    /// Finished
    Done,
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Mistral, prompt_token: &'a Array, temp: f32) -> Self {
        Self {
            model,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }

    /// Compute the next token given current token and cache
    fn compute_next(&mut self, y: &Array, cache: &[Option<(Array, Array)>]) -> Result<(Array, Vec<Option<(Array, Array)>>)> {
        let next_token = y.index((.., NewAxis));
        let input = MistralInput {
            inputs: &next_token,
            cache,
        };
        let MistralOutput { logits, cache: new_cache } = self.model.forward(input)?;
        let logits = logits.squeeze_axes(&[1])?;
        let next_y = sample(&logits, self.temp)?;
        Ok((next_y, new_cache))
    }
}

impl Iterator for Generate<'_> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        // Take ownership of state
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                let initial_cache = Vec::with_capacity(0);
                let input = MistralInput {
                    inputs: prompt_token,
                    cache: &initial_cache,
                };
                let MistralOutput { logits, cache } = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                // Start async eval and force completion for first token
                tri!(async_eval([&y]));
                tri!(eval([&y]));

                // Compute next token and start its async eval
                let (next_y, new_cache) = tri!(self.compute_next(&y, &cache));
                tri!(async_eval([&next_y]));

                // Return first token, store next for pipeline
                self.state = GenerateState::Pipelined {
                    current_y: next_y,
                    cache: new_cache,
                };
                Some(Ok(y))
            }
            GenerateState::Pipelined { current_y, cache } => {
                // current_y's async_eval was started in previous iteration
                // Compute next token while current_y finalizes
                let (next_y, new_cache) = tri!(self.compute_next(&current_y, &cache));

                // Start async eval for next token (background computation)
                tri!(async_eval([&next_y]));

                // Return current (its async_eval should be done by now)
                self.state = GenerateState::Pipelined {
                    current_y: next_y,
                    cache: new_cache,
                };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}

fn main() -> Result<()> {
    // If you want to manually set the cache directory, you can set the HF_CACHE_DIR
    // environment variable or put it in a .env file located at the root of this example
    // (ie. examples/mistral/.env)
    let _ = dotenv::dotenv();
    let api = build_hf_api()?;

    // Parse args
    let cli = Cli::parse();

    mlx_rs::random::seed(cli.seed)?;

    // Use pre-quantized model for optimal performance
    let model_id = "mlx-community/Mistral-7B-Instruct-v0.2-4bit".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    println!("[INFO] Downloading model...");
    let tokenizer = get_tokenizer(&repo)?;
    let model_dir = download_weights(&repo)?;

    println!("[INFO] Loading model...");
    let args = get_model_args(&repo)?;
    let mut model = load_model_quantized(&model_dir, &args)?;

    let encoding = tokenizer.encode(&cli.prompt[..], true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    print!("{}", cli.prompt);

    let start_time = std::time::Instant::now();
    let generate = Generate::new(&mut model, &prompt_tokens, cli.temp);
    let mut token_ids = Vec::with_capacity(cli.max_tokens);

    // Use proper async pipelining: .item() syncs previous token while next computes
    for (i, token) in generate.enumerate() {
        let token = token?;
        let token_id = token.item::<u32>();  // This syncs the token (overlaps with next computation)
        token_ids.push(token_id);

        // Stream output every tokens_per_eval tokens
        if token_ids.len() % cli.tokens_per_eval == 0 {
            let s = tokenizer.decode(&token_ids[token_ids.len() - cli.tokens_per_eval..], true)?;
            print!("{s}");
        }

        if i >= cli.max_tokens - 1 {
            break;
        }
    }

    synchronize(&Stream::default());
    let generation_time = start_time.elapsed();

    // Print remaining tokens
    let remaining = token_ids.len() % cli.tokens_per_eval;
    if remaining > 0 {
        let s = tokenizer.decode(&token_ids[token_ids.len() - remaining..], true)?;
        print!("{s}");
    }
    println!();

    println!("------");
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)",
        token_ids.len(),
        generation_time.as_secs_f64(),
        token_ids.len() as f64 / generation_time.as_secs_f64());

    Ok(())
}
