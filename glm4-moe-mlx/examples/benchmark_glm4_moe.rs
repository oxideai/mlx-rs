//! Proper benchmark for GLM-4.5 MoE using async pipelining
//!
//! Run with: cargo run --release -p glm4-moe-mlx --example benchmark_glm4_moe

use std::time::Instant;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::async_eval;
use mlx_rs::module::Module;
use mlx_rs::{Array, Stream};
use glm4_moe_mlx::{load_model, load_tokenizer, ModelInput, KVCache, init_cache, sample, Error};

fn synchronize(stream: &Stream) {
    unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()); }
}

fn main() -> Result<(), Error> {
    let model_dir = std::env::args().nth(1)
        .unwrap_or_else(|| std::env::var("HOME").unwrap() + "/.cache/huggingface/hub/mlx-community--GLM-4.5-Air-3bit");

    println!("Loading GLM-4.5 MoE model from: {}", model_dir);
    let start = Instant::now();

    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let prompt = "请解释一下什么是人工智能，以及它在日常生活中的应用有哪些？";
    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    println!("Prompt ({} tokens): {}", encoding.get_ids().len(), prompt);

    let num_tokens = 100;
    let num_warmup = 10;
    let num_runs = 3;

    println!("Running benchmark ({} tokens, {} warmup, {} runs)...", num_tokens, num_warmup, num_runs);

    let mut run_results = Vec::new();

    for run in 0..num_runs {
        let mut cache: Vec<KVCache> = init_cache(model.model.num_hidden_layers as usize);

        // Prefill
        let input = ModelInput { inputs: &prompt_tokens, mask: None, cache: &mut cache };
        let logits = model.forward(input)?;
        let mut y = sample(&logits.index((.., -1, ..)), 0.0)?;
        async_eval([&y])?;

        // Warmup with proper pipelining
        for _ in 0..num_warmup {
            let inputs = y.index((.., NewAxis));
            let input = ModelInput { inputs: &inputs, mask: None, cache: &mut cache };
            let logits = model.forward(input)?;
            let next_y = sample(&logits, 0.0)?;
            async_eval([&next_y])?;
            let _ = y.item::<u32>();  // Sync previous token
            y = next_y;
        }

        // Timed run with proper pipelining
        let start = Instant::now();
        for _ in 0..num_tokens {
            let inputs = y.index((.., NewAxis));
            let input = ModelInput { inputs: &inputs, mask: None, cache: &mut cache };
            let logits = model.forward(input)?;
            let next_y = sample(&logits, 0.0)?;
            async_eval([&next_y])?;
            let _ = y.item::<u32>();  // Sync previous token (overlap!)
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
