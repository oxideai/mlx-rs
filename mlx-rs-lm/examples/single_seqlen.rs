// Test a single sequence length (specified via command line arg)
use mlx_rs_lm::models::glm4_moe::{load_glm4_moe_model, ModelInput, init_cache};
use mlx_rs_lm::cache::ConcatKeyValueCache;
use mlx_rs::module::Module;
use mlx_rs::Stream;
use std::time::Instant;

fn sync() {
    unsafe { mlx_sys::mlx_synchronize(Stream::default().as_ptr()); }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seq_len: i32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(127);

    let model_dir = std::path::Path::new("/Users/yuechen/.cache/huggingface/hub/models--mlx-community--GLM-4.5-Air-3bit/snapshots/c4367db4696015335df032b8df2227814b277077");

    unsafe {
        let info = mlx_sys::mlx_metal_device_info();
        let mut old_limit: usize = 0;
        mlx_sys::mlx_set_wired_limit(&mut old_limit, info.max_recommended_working_set_size);
        mlx_sys::mlx_set_compile_mode(mlx_sys::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED);
    }

    let mut model = load_glm4_moe_model(&model_dir)?;
    let num_layers = model.model.num_hidden_layers as usize;

    let prompt_data: Vec<u32> = (1u32..=(seq_len as u32)).collect();
    let prompt = mlx_rs::Array::from(&prompt_data[..]).reshape(&[1, seq_len])?;

    // Warmup (5 runs)
    for _ in 0..5 {
        let mut cache: Vec<ConcatKeyValueCache> = init_cache(num_layers);
        let input = ModelInput { inputs: &prompt, mask: None, cache: &mut cache };
        let logits = model.forward(input)?;
        mlx_rs::transforms::eval([&logits])?;
    }
    sync();

    // Measure (10 runs)
    let mut times = vec![];
    for _ in 0..10 {
        let mut cache: Vec<ConcatKeyValueCache> = init_cache(num_layers);
        let start = Instant::now();
        let input = ModelInput { inputs: &prompt, mask: None, cache: &mut cache };
        let logits = model.forward(input)?;
        mlx_rs::transforms::eval([&logits])?;
        sync();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("seq_len={}: avg={:.1}ms, min={:.1}ms, max={:.1}ms", seq_len, avg, min, max);

    Ok(())
}
