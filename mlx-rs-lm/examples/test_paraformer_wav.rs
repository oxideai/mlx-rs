//! Test Paraformer with WAV file features
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use mlx_rs::module::Module;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use mlx_rs_lm::models::paraformer::{load_paraformer_model, DecoderInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pre-extracted features
    let mut file = File::open("/tmp/test_features.bin")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let features: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let n_frames = 50i32;
    let n_dim = 560i32;
    let expected_size = (n_frames * n_dim) as usize;
    
    println!("Loaded {} floats, expected {}", features.len(), expected_size);
    assert_eq!(features.len(), expected_size, "Feature size mismatch");
    
    let x = Array::from_slice(&features, &[1, n_frames, n_dim]);
    println!("Input features shape: [1, {}, {}]", n_frames, n_dim);
    println!("Features [0,0,:10]: {:?}", &features[0..10]);

    // Load model
    println!("\nLoading model...");
    let model_path = "/tmp/paraformer.safetensors";
    let mut model = load_paraformer_model(model_path)?;
    model.training_mode(false);

    // Load vocabulary
    let vocab: Vec<String> = std::fs::read_to_string("/tmp/paraformer_vocab.txt")?
        .lines()
        .map(|s| s.to_string())
        .collect();
    println!("Loaded {} tokens", vocab.len());

    // Run inference
    let start = Instant::now();
    
    let encoder_out = model.encoder.forward(&x)?;
    let (acoustic_embeds, token_num) = model.predictor.forward(&encoder_out)?;
    let logits = model.decoder.forward(DecoderInput {
        acoustic_embeds: &acoustic_embeds,
        encoder_out: &encoder_out,
    })?;
    eval([&logits])?;
    
    let elapsed = start.elapsed();
    println!("\nInference time: {:.1} ms", elapsed.as_millis());

    // Get token IDs
    let token_ids = mlx_rs::argmax_axis!(logits, -1)?;
    let token_ids = token_ids.as_dtype(mlx_rs::Dtype::Int32)?;
    eval([&token_ids])?;
    let token_ids_vec: Vec<i32> = token_ids.try_as_slice::<i32>()?.to_vec();
    
    println!("Token IDs: {:?}", &token_ids_vec);

    // Decode to text (filter special tokens)
    let text: String = token_ids_vec
        .iter()
        .filter_map(|&id| {
            let id = id as usize;
            if id < vocab.len() {
                let token = &vocab[id];
                // Filter special tokens: <blank>=0, <s>=1, </s>=2, <unk>=8403
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

    println!("\n=== Results ===");
    println!("Rust transcription: {}", text);
    println!("Expected (FunASR):  目前的等级为二等站");

    Ok(())
}
