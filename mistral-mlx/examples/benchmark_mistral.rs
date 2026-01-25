//! Benchmark Mistral with proper async pipelining
//!
//! Run with: cargo run --release -p mistral-mlx --example benchmark_mistral

use std::time::Instant;
use anyhow::Result;
use hf_hub::api::sync::Api;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::async_eval;
use mlx_rs::module::Module;
use mlx_rs::{Array, Stream};
use mistral_mlx::{load_model, load_tokenizer, ModelInput, KVCache, init_cache, sample};

fn synchronize(stream: &Stream) {
    unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()); }
}

fn download_model(model_id: &str) -> Result<std::path::PathBuf> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json")?;
    let _ = repo.get("tokenizer.json")?;

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
    let model_id = "mlx-community/Mistral-7B-Instruct-v0.2-4bit";

    println!("Downloading model: {}", model_id);
    let model_dir = download_model(model_id)?;

    println!("Loading model...");
    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;
    println!("Model loaded!");

    let prompt = "What is the capital of France?";
    let formatted = format!("[INST] {} [/INST]", prompt);
    let encoding = tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    println!("Prompt ({} tokens): {}", encoding.get_ids().len(), prompt);

    let num_tokens = 100;
    let num_warmup = 10;
    let num_runs = 3;

    println!("Running benchmark ({} tokens, {} warmup, {} runs)...", num_tokens, num_warmup, num_runs);

    let mut run_results = Vec::new();

    for run in 0..num_runs {
        let mut cache: Vec<KVCache> = init_cache(model.model.layers.len());

        // Prefill
        let input = ModelInput { inputs: &prompt_tokens, mask: None, cache: &mut cache };
        let logits = model.forward(input)?;
        let mut y = sample(&logits.index((.., -1, ..)), 0.0)?;
        async_eval([&y])?;

        // Warmup
        for _ in 0..num_warmup {
            let inputs = y.index((.., NewAxis));
            let input = ModelInput { inputs: &inputs, mask: None, cache: &mut cache };
            let logits = model.forward(input)?;
            let next_y = sample(&logits, 0.0)?;
            async_eval([&next_y])?;
            let _ = y.item::<u32>();
            y = next_y;
        }

        // Timed run
        let start = Instant::now();
        for _ in 0..num_tokens {
            let inputs = y.index((.., NewAxis));
            let input = ModelInput { inputs: &inputs, mask: None, cache: &mut cache };
            let logits = model.forward(input)?;
            let next_y = sample(&logits, 0.0)?;
            async_eval([&next_y])?;
            let _ = y.item::<u32>();  // Sync previous (overlap!)
            y = next_y;
        }
        synchronize(&Stream::default());
        let elapsed = start.elapsed();
        let tps = num_tokens as f64 / elapsed.as_secs_f64();
        run_results.push(tps);

        println!("  Run {}: {:.1} tok/s", run + 1, tps);
    }

    let avg = run_results.iter().sum::<f64>() / run_results.len() as f64;
    let variance = run_results.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / run_results.len() as f64;
    let stddev = variance.sqrt();

    println!("---");
    println!("Result: {:.1} +/- {:.2} tok/s", avg, stddev);

    Ok(())
}
