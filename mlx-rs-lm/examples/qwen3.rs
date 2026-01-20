use clap::Parser;
use anyhow::Result;
use hf_hub::api::sync::Api;
use mlx_rs_lm::models::qwen3::{load_qwen3_model, Generate};
use mlx_rs_lm::cache::KVCache;
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Tokenizer,
};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use serde::Serialize;

/// Custom role type that supports system, user, and assistant
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model repository ID or local path
    #[arg(long, default_value = "mlx-community/Qwen3-4B-bf16")]
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

/// Set the wired memory limit for better GPU performance.
fn set_wired_limit_max() {
    unsafe {
        let info = mlx_sys::mlx_metal_device_info();
        let max_size = info.max_recommended_working_set_size;
        let mut old_limit: usize = 0;
        mlx_sys::mlx_set_wired_limit(&mut old_limit, max_size);
        mlx_sys::mlx_set_compile_mode(mlx_sys::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set wired memory limit and enable compilation for optimal performance
    set_wired_limit_max();

    if args.debug {
        println!("Loading model: {}", args.model);
    }

    // 1. Initialize API and download model if needed
    let api = Api::new()?;
    let repo = api.model(args.model.clone());

    // Download essential files first
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json")?;
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
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 3. Load chat template
    let model_chat_template = load_model_chat_template_from_file(&tokenizer_config_path)?
        .expect("Model chat template not found");

    // 4. Create conversations
    let mut conversations: Vec<Conversation<ChatRole, String>> = vec![];

    if let Some(system_prompt) = args.system {
        conversations.push(Conversation {
            role: ChatRole::System,
            content: system_prompt,
        });
    }

    conversations.push(Conversation {
        role: ChatRole::User,
        content: args.prompt,
    });

    // 5. Apply chat template
    let apply_args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: &args.model,
        chat_template_id: None,
        add_generation_prompt: Some(true),  // Important: add generation prompt
        continue_final_message: None,
    };

    let encodings = tokenizer.apply_chat_template_and_encode(model_chat_template, apply_args)?;
    let prompt: Vec<u32> = encodings
        .iter()
        .flat_map(|encoding| encoding.get_ids())
        .copied()
        .collect();

    if args.debug {
        println!("Input tokens: {}", prompt.len());
        // Show the rendered prompt (decoded)
        let rendered = tokenizer.decode(&prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode prompt: {}", e))?;
        println!("Rendered prompt:\n---\n{}\n---", rendered);
    }

    // 6. Load model
    println!("Loading model weights...");
    let mut model = load_qwen3_model(&model_dir)?;
    println!("Model loaded successfully!");

    // 7. Generate
    println!("\nGenerating response...");
    let prompt_tokens = mlx_rs::Array::from(&prompt[..]).index(NewAxis);
    let start_time = std::time::Instant::now();
    let mut cache = Vec::new();

    let generate = Generate::<KVCache>::new(
        &mut model,
        &mut cache,
        args.temperature,
        &prompt_tokens,
    );

    let mut output_tokens = Vec::new();

    for token in generate {
        let token = token?;
        let token_id = token.item::<u32>();
        if args.debug {
            eprintln!("DEBUG: Generated token {}: {}", output_tokens.len(), token_id);
        }
        output_tokens.push(token.clone());

        if output_tokens.len() >= args.max_tokens {
            break;
        }

        // Optional: stream output as it's generated
        // let decoded = tokenizer.decode(&[token.item::<u32>()], true)?;
        // print!("{}", decoded);
    }

    let generation_time = start_time.elapsed();

    // 8. Decode and print final response
    // Convert Array tokens to u32 for decoding
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