//! Pure Rust Paraformer ASR test - no Python dependencies
//!
//! Uses the unified MelFrontend from paraformer.rs for FunASR-compatible features.

use std::time::Instant;

use mlx_rs::module::Module;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use mlx_rs_lm::audio::{load_wav, resample};
use mlx_rs_lm::models::paraformer::{
    load_paraformer_model, parse_cmvn_file, DecoderInput, MelFrontend, ParaformerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wav_path = "/Users/yuechen/home/mofa-studio/models/setup-local-models/asr-validation/test.wav";
    let model_path = "/tmp/paraformer.safetensors";
    let cmvn_path = "/tmp/paraformer_cmvn.txt";
    let vocab_path = "/tmp/paraformer_vocab.txt";

    // Load audio
    println!("Loading audio: {}", wav_path);
    let (samples, src_rate) = load_wav(wav_path)?;
    let duration_secs = samples.len() as f32 / src_rate as f32;
    println!("Audio: {} samples, {} Hz, {:.2}s", samples.len(), src_rate, duration_secs);

    // Resample to 16kHz if needed
    let samples = if src_rate != 16000 {
        println!("Resampling from {} to 16000 Hz", src_rate);
        resample(&samples, src_rate, 16000)
    } else {
        samples
    };

    // Create audio array
    let audio = Array::from_slice(&samples, &[samples.len() as i32]);

    // Setup frontend with CMVN
    println!("\nSetting up MelFrontend...");
    let config = ParaformerConfig::default();
    let mut frontend = MelFrontend::new(&config);

    // Load and set CMVN parameters
    let (addshift, rescale) = parse_cmvn_file(cmvn_path)?;
    frontend.set_cmvn(addshift, rescale);

    // Extract features using MelFrontend (unified, FunASR-compatible)
    println!("Extracting features...");
    let features = frontend.forward(&audio)?;
    eval([&features])?;
    println!("Features shape: {:?}", features.shape());

    // Load model
    println!("\nLoading model...");
    let mut model = load_paraformer_model(model_path)?;
    model.training_mode(false);

    // Load vocabulary
    let vocab: Vec<String> = std::fs::read_to_string(vocab_path)?
        .lines()
        .map(|s| s.to_string())
        .collect();
    println!("Loaded {} tokens", vocab.len());

    // Run inference
    println!("\nRunning inference...");
    let start = Instant::now();

    let encoder_out = model.encoder.forward(&features)?;
    let (acoustic_embeds, _) = model.predictor.forward(&encoder_out)?;
    let logits = model.decoder.forward(DecoderInput {
        acoustic_embeds: &acoustic_embeds,
        encoder_out: &encoder_out,
    })?;
    eval([&logits])?;

    let elapsed = start.elapsed();

    // Get token IDs
    let token_ids = mlx_rs::argmax_axis!(logits, -1)?;
    let token_ids = token_ids.as_dtype(mlx_rs::Dtype::Int32)?;
    eval([&token_ids])?;
    let token_ids_vec: Vec<i32> = token_ids.try_as_slice::<i32>()?.to_vec();

    // Decode to text
    let text: String = token_ids_vec
        .iter()
        .filter_map(|&id| {
            let id = id as usize;
            if id < vocab.len() {
                let token = &vocab[id];
                if token == "<blank>" || token == "<s>" || token == "</s>" || token == "<unk>" {
                    None
                } else {
                    Some(token.clone())
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    // Calculate metrics
    let inference_ms = elapsed.as_millis();
    let rtf = (inference_ms as f32 / 1000.0) / duration_secs;

    println!("\n=== Results (Pure Rust with MelFrontend) ===");
    println!("Transcription: {}", text);
    println!("Expected:      目前的等级为二等站");
    println!("\nPerformance:");
    println!("  Audio duration: {:.2}s", duration_secs);
    println!("  Inference time: {} ms", inference_ms);
    println!("  RTF: {:.4}x", rtf);
    println!("  Speed: {:.1}x real-time", 1.0 / rtf);

    Ok(())
}
