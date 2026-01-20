use clap::Parser;
use anyhow::Result;
use hf_hub::api::sync::Api;
use mlx_rs_lm::models::glm4_moe::{load_glm4_moe_model, Generate};
use mlx_rs_lm::cache::ConcatKeyValueCache;
use mlx_lm_utils::tokenizer::Tokenizer;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::Stream;

/// Set the wired memory limit for better GPU performance.
/// This is a key optimization for MLX inference.
fn set_wired_limit_max() {
    unsafe {
        // Get metal device info to get max_recommended_working_set_size
        let info = mlx_sys::mlx_metal_device_info();
        let max_size = info.max_recommended_working_set_size;

        // Set wired limit to max recommended
        let mut old_limit: usize = 0;
        mlx_sys::mlx_set_wired_limit(&mut old_limit, max_size);

        // Enable compile mode for graph optimization
        mlx_sys::mlx_set_compile_mode(mlx_sys::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED);

        eprintln!("Set wired limit to {} MB (was {} MB)",
            max_size / (1024 * 1024),
            old_limit / (1024 * 1024));
    }
}

/// Clear the MLX memory cache
fn clear_cache() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

/// Synchronize the given stream
fn synchronize(stream: &Stream) {
    unsafe {
        mlx_sys::mlx_synchronize(stream.as_ptr());
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model repository ID or local path
    #[arg(long, default_value = "mlx-community/GLM-4.5-Air-3bit")]
    model: String,

    /// Input prompt
    #[arg(long)]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// System prompt (optional)
    #[arg(long)]
    system: Option<String>,

    /// Show debug information
    #[arg(long)]
    debug: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set wired memory limit for optimal GPU performance
    set_wired_limit_max();

    if args.debug {
        println!("Loading model: {}", args.model);
    }

    // 1. Initialize API and download model if needed
    let api = Api::new()?;
    let repo = api.model(args.model.clone());

    // Download essential files first
    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;
    let index_path = repo.get("model.safetensors.index.json")?;

    // Read the weight index and download all weight files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;
    let weight_map = index["weight_map"].as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid weight index"))?;

    // Collect unique weight files
    let weight_files: std::collections::HashSet<&str> = weight_map.values()
        .filter_map(|v| v.as_str())
        .collect();

    if args.debug {
        println!("Downloading {} weight files...", weight_files.len());
    }

    // Download all weight files
    for weight_file in &weight_files {
        if args.debug {
            println!("  - {}", weight_file);
        }
        repo.get(weight_file)?;
    }

    // Get model directory (parent of config.json)
    let model_dir = config_path.parent()
        .ok_or_else(|| anyhow::anyhow!("Could not determine model directory"))?;

    if args.debug {
        println!("Model directory: {}", model_dir.display());
        println!("Tokenizer: {}", tokenizer_path.display());
    }

    // 2. Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 3. Build GLM-4.5 prompt manually (same format as GLM-4)
    // GLM-4 format: [gMASK]<sop><|system|>\n{system}<|user|>\n{user}<|assistant|>
    let mut prompt_text = String::from("[gMASK]<sop>");

    if let Some(system_prompt) = &args.system {
        prompt_text.push_str("<|system|>\n");
        prompt_text.push_str(system_prompt);
    }

    prompt_text.push_str("<|user|>\n");
    prompt_text.push_str(&args.prompt);
    prompt_text.push_str("<|assistant|>");

    if args.debug {
        println!("Prompt text: {}", prompt_text);
    }

    // 4. Encode prompt
    let encoding = tokenizer.encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
    let prompt: Vec<u32> = encoding.get_ids().to_vec();

    if args.debug {
        println!("Input tokens: {}", prompt.len());
    }

    // 5. Load model
    println!("Loading model weights (this may take a while for MoE models)...");
    let mut model = load_glm4_moe_model(&model_dir)?;
    println!("Model loaded successfully!");

    // 6. Generate tokens
    println!("\nGenerating response...");
    let prompt_tokens = mlx_rs::Array::from(&prompt[..]).index(NewAxis);
    let start_time = std::time::Instant::now();
    let mut cache = Vec::new();

    // Note: Don't use with_new_default_stream() - it causes 2x slowdown
    let generate = Generate::<ConcatKeyValueCache>::new(
        &mut model,
        &mut cache,
        args.temperature,
        &prompt_tokens,
    );

    let mut output_tokens = Vec::new();

    // GLM-4 EOS tokens: 151329 (<|endoftext|>), 151336 (<|user|>), 151338 (<|observation|>)
    let eos_tokens: [u32; 3] = [151329, 151336, 151338];

    let mut token_start = std::time::Instant::now();
    for token in generate {
        let token = token?;
        let token_id = token.item::<u32>();
        if args.debug {
            let token_time = token_start.elapsed();
            eprintln!("DEBUG: Token {}: {} ({:.1}ms)", output_tokens.len(), token_id, token_time.as_secs_f64() * 1000.0);
            token_start = std::time::Instant::now();
        }

        // Check for EOS tokens
        if eos_tokens.contains(&token_id) {
            if args.debug {
                eprintln!("DEBUG: Hit EOS token {}", token_id);
            }
            break;
        }

        output_tokens.push(token.clone());

        // Clear cache periodically (every 256 tokens) like Python does
        if output_tokens.len() % 256 == 0 {
            clear_cache();
        }

        if output_tokens.len() >= args.max_tokens {
            break;
        }
    }

    // Synchronize before measuring time
    synchronize(&Stream::default());
    let generation_time = start_time.elapsed();

    // 7. Decode and print final response
    let token_ids: Vec<u32> = output_tokens
        .iter()
        .map(|t| t.item::<u32>())
        .collect();
    let response = tokenizer.decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

    println!("\nResponse:");
    println!("{}", response);

    println!("\nGeneration stats:");
    println!("  - Tokens generated: {}", output_tokens.len());
    println!("  - Time: {:.2}s", generation_time.as_secs_f64());
    println!("  - Tokens/sec: {:.2}", output_tokens.len() as f64 / generation_time.as_secs_f64());

    Ok(())
}
