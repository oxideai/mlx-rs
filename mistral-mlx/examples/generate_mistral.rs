//! Simple text generation with Mistral
//!
//! Run with: cargo run --release -p mistral-mlx --example generate_mistral -- --prompt "Hello"

use std::time::Instant;
use anyhow::Result;
use clap::Parser;
use hf_hub::api::sync::Api;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::Stream;
use mistral_mlx::{load_model, load_tokenizer, Generate, KVCache};

fn synchronize(stream: &Stream) {
    unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()); }
}

#[derive(Parser)]
#[command(about = "Mistral text generation")]
struct Args {
    /// Model repository ID
    #[arg(long, default_value = "mlx-community/Mistral-7B-Instruct-v0.2-4bit")]
    model: String,

    /// Input prompt
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value = "0.7")]
    temperature: f32,
}

fn download_model(model_id: &str) -> Result<std::path::PathBuf> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json")?;
    let _ = repo.get("tokenizer.json")?;

    // Download weights
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
        let _ = repo.get("model.safetensors")?;
    }

    Ok(config_path.parent().unwrap().to_path_buf())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Downloading model: {}", args.model);
    let model_dir = download_model(&args.model)?;

    println!("Loading model...");
    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;
    println!("Model loaded!");

    // Mistral Instruct format
    let formatted = format!("[INST] {} [/INST]", args.prompt);
    let encoding = tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let prompt_tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

    println!("Prompt ({} tokens): {}", encoding.get_ids().len(), args.prompt);
    println!("---");

    let start = Instant::now();
    let mut cache = Vec::new();

    let generator = Generate::<KVCache>::new(
        &mut model,
        &mut cache,
        args.temperature,
        &prompt_tokens,
    );

    let mut token_ids = Vec::new();
    let eos_token: u32 = 2;  // </s>

    for token in generator.take(args.max_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();

        if token_id == eos_token {
            break;
        }

        token_ids.push(token_id);
    }

    synchronize(&Stream::default());
    let gen_time = start.elapsed();

    let response = tokenizer.decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    println!("{}", response);
    println!("---");
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)",
        token_ids.len(),
        gen_time.as_secs_f64(),
        token_ids.len() as f64 / gen_time.as_secs_f64());

    Ok(())
}
