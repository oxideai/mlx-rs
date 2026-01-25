//! Voice Cloning Example
//!
//! Demonstrates the high-level VoiceCloner API for GPT-SoVITS.
//!
//! # Usage
//!
//! ```bash
//! # Basic usage with default reference voice
//! cargo run --example voice_clone --release -- "你好，世界！"
//!
//! # With custom reference audio
//! cargo run --example voice_clone --release -- "你好，世界！" --ref /path/to/reference.wav
//!
//! # Save to file
//! cargo run --example voice_clone --release -- "你好，世界！" --output /tmp/output.wav
//!
//! # Interactive mode
//! cargo run --example voice_clone --release -- --interactive
//! ```

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};

// Default reference audio
const DEFAULT_REF_AUDIO: &str = "/Users/yuechen/.dora/models/primespeech/moyoyo/ref_audios/doubao_ref_mix_new.wav";
// Reference text for doubao voice (must match the reference audio)
const DEFAULT_REF_TEXT: &str = "这家resturant的steak很有名，但是vegetable salad的price有点贵";

// Luo Xiang reference
const LUOXIANG_REF_AUDIO: &str = "/Users/yuechen/.dora/models/primespeech/moyoyo/ref_audios/luoxiang_ref.wav";
const LUOXIANG_REF_TEXT: &str = "复杂的问题背后也许没有统一的答案，选择站在正方还是反方，其实取决于你对一系列价值判断的回答。";

fn print_help() {
    println!("Voice Clone - GPT-SoVITS TTS");
    println!("============================");
    println!();
    println!("Usage:");
    println!("  voice_clone \"text to speak\"              Synthesize and play text (zero-shot mode)");
    println!("  voice_clone \"text\" --ref FILE            Use custom reference audio");
    println!("  voice_clone \"text\" --ref-text \"text\"     Reference transcript (enables few-shot mode)");
    println!("  voice_clone \"text\" --codes FILE.bin      Use pre-computed prompt semantic codes");
    println!("  voice_clone \"text\" --output FILE.wav     Save to WAV file");
    println!("  voice_clone --interactive                 Interactive mode");
    println!("  voice_clone --help                        Show this help");
    println!();
    println!("Examples:");
    println!("  voice_clone \"你好，世界！\"");
    println!("  voice_clone \"今天天气真好\" --ref my_voice.wav");
    println!("  voice_clone \"测试语音\" --output test.wav");
    println!();
    println!("Few-shot mode (better quality with reference transcript):");
    println!("  voice_clone \"你好\" --ref voice.wav --ref-text \"这是参考音频的文本\"");
    println!();
    println!("Few-shot with Python-extracted codes (best quality):");
    println!("  # First extract codes with Python:");
    println!("  python scripts/extract_prompt_semantic.py voice.wav codes.bin");
    println!("  # Then use them:");
    println!("  voice_clone \"你好\" --ref voice.wav --ref-text \"参考文本\" --codes codes.bin");
}

/// Parsed command line arguments
struct Args {
    text: Option<String>,
    ref_audio: Option<String>,
    ref_text: Option<String>,
    codes_path: Option<String>,
    tokens_path: Option<String>,  // Pre-computed semantic tokens (for testing)
    output: Option<String>,
    interactive: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().skip(1).collect();

    let mut text = None;
    let mut ref_audio = None;
    let mut ref_text = None;
    let mut codes_path = None;
    let mut tokens_path = None;
    let mut output = None;
    let mut interactive = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--ref" | "-r" => {
                if i + 1 < args.len() {
                    ref_audio = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--ref-text" | "-t" => {
                if i + 1 < args.len() {
                    ref_text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--codes" | "-c" => {
                if i + 1 < args.len() {
                    codes_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--tokens" => {
                if i + 1 < args.len() {
                    tokens_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--text" => {
                if i + 1 < args.len() {
                    text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--voice" => {
                // Set reference audio and text for known voices
                if i + 1 < args.len() {
                    let voice = &args[i + 1];
                    match voice.as_str() {
                        "doubao" => {
                            if ref_audio.is_none() { ref_audio = Some(DEFAULT_REF_AUDIO.to_string()); }
                            if ref_text.is_none() { ref_text = Some(DEFAULT_REF_TEXT.to_string()); }
                        }
                        "luoxiang" | "luo" => {
                            if ref_audio.is_none() { ref_audio = Some(LUOXIANG_REF_AUDIO.to_string()); }
                            if ref_text.is_none() { ref_text = Some(LUOXIANG_REF_TEXT.to_string()); }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            "--play" => {
                // Play is default behavior, ignore
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            arg if !arg.starts_with('-') => {
                if text.is_none() {
                    text = Some(arg.to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }

    Args { text, ref_audio, ref_text, codes_path, tokens_path, output, interactive }
}

fn synthesize_and_play(cloner: &mut VoiceCloner, text: &str, output: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Text: {}", text);
    println!("🎤 Reference: {}", cloner.reference_path().unwrap_or("none"));

    let start = Instant::now();
    let audio = cloner.synthesize(text)?;
    let gen_time = start.elapsed();

    println!("✅ Generated {} tokens in {:.1}ms", audio.num_tokens, gen_time.as_secs_f64() * 1000.0);
    println!("🔊 Duration: {:.2}s ({} samples)", audio.duration_secs(), audio.samples.len());

    // Save if output specified
    if let Some(path) = output {
        cloner.save_wav(&audio, path)?;
        println!("💾 Saved to: {}", path);
    }

    // Play audio
    println!("▶️  Playing...");
    cloner.play_blocking(&audio)?;

    Ok(())
}

fn interactive_mode(cloner: &mut VoiceCloner) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎙️  Voice Clone Interactive Mode");
    println!("================================");
    println!("Commands:");
    println!("  /ref <path>    - Change reference audio");
    println!("  /save <path>   - Save last audio to file");
    println!("  /quit          - Exit");
    println!("  <text>         - Synthesize and play text");
    println!();

    let mut last_audio = None;

    loop {
        print!("voice> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with("/ref ") {
            let path = &input[5..].trim();
            match cloner.set_reference_audio(path) {
                Ok(()) => println!("✅ Reference audio set to: {}", path),
                Err(e) => println!("❌ Error: {}", e),
            }
        } else if input.starts_with("/save ") {
            let path = &input[6..].trim();
            if let Some(ref audio) = last_audio {
                match cloner.save_wav(audio, path) {
                    Ok(()) => println!("💾 Saved to: {}", path),
                    Err(e) => println!("❌ Error: {}", e),
                }
            } else {
                println!("❌ No audio to save. Generate some text first.");
            }
        } else if input == "/quit" || input == "/exit" || input == "/q" {
            println!("👋 Goodbye!");
            break;
        } else if input.starts_with('/') {
            println!("❓ Unknown command. Try /ref, /save, or /quit");
        } else {
            // Synthesize text
            match cloner.synthesize(input) {
                Ok(audio) => {
                    println!("✅ {} tokens, {:.2}s", audio.num_tokens, audio.duration_secs());
                    if let Err(e) = cloner.play_blocking(&audio) {
                        println!("❌ Playback error: {}", e);
                    }
                    last_audio = Some(audio);
                }
                Err(e) => println!("❌ Synthesis error: {}", e),
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // Initialize voice cloner
    println!("🔧 Initializing VoiceCloner...");
    let start = Instant::now();
    let config = VoiceClonerConfig::default();
    let mut cloner = VoiceCloner::new(config)?;
    println!("   Models loaded in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Check HuBERT availability for few-shot mode
    if cloner.few_shot_available() {
        println!("   HuBERT available (few-shot mode supported)");
    } else {
        println!("   HuBERT not available (zero-shot mode only)");
    }

    // Set reference audio
    let ref_path = args.ref_audio.as_deref().unwrap_or(DEFAULT_REF_AUDIO);
    if !Path::new(ref_path).exists() {
        println!("❌ Reference audio not found: {}", ref_path);
        return Ok(());
    }

    let start = Instant::now();

    // Use few-shot mode if reference text is provided
    if let Some(ref ref_text) = args.ref_text {
        // Check if pre-computed codes are provided
        if let Some(ref codes_path) = args.codes_path {
            if !Path::new(codes_path).exists() {
                println!("❌ Codes file not found: {}", codes_path);
                return Ok(());
            }
            cloner.set_reference_with_precomputed_codes(ref_path, ref_text, codes_path)?;
            println!("   Reference loaded (few-shot with Python codes) in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
            println!("   Reference text: \"{}\"", ref_text);
            println!("   Codes file: {}", codes_path);
        } else {
            if !cloner.few_shot_available() {
                println!("❌ Few-shot mode requires HuBERT model");
                println!("   Tip: Use --codes with pre-computed codes from Python");
                return Ok(());
            }
            cloner.set_reference_audio_with_text(ref_path, ref_text)?;
            println!("   Reference loaded (few-shot mode) in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
            println!("   Reference text: \"{}\"", ref_text);
        }
    } else {
        cloner.set_reference_audio(ref_path)?;
        println!("   Reference loaded (zero-shot mode) in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    if args.interactive {
        interactive_mode(&mut cloner)?;
    } else if let Some(ref tokens_path) = args.tokens_path {
        // Use pre-computed tokens (for testing/debugging)
        use std::fs;
        let text = args.text.as_deref().unwrap_or("从季节上看，主要是增在秋粮");
        let bytes = fs::read(tokens_path)?;
        let tokens: Vec<i32> = bytes.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("\n📝 Text: {}", text);
        println!("🔢 Using {} pre-computed tokens from {}", tokens.len(), tokens_path);
        println!("   First 10: {:?}", &tokens[..tokens.len().min(10)]);

        let start = std::time::Instant::now();
        let audio = cloner.synthesize_from_tokens(text, &tokens)?;
        let gen_time = start.elapsed();

        println!("✅ Vocoded in {:.1}ms", gen_time.as_secs_f64() * 1000.0);
        println!("🔊 Duration: {:.2}s ({} samples)", audio.duration_secs(), audio.samples.len());

        println!("▶️  Playing...");
        cloner.play_blocking(&audio)?;
    } else if let Some(text) = args.text {
        synthesize_and_play(&mut cloner, &text, args.output.as_deref())?;
    } else {
        // Default demo
        let demo_texts = [
            "你好，欢迎使用语音克隆系统。",
            "今天天气真好，我们一起出去玩吧！",
            "这是一个测试句子，用来验证语音合成的效果。",
        ];

        println!("\n🎭 Voice Clone Demo");
        println!("==================");

        for text in demo_texts {
            synthesize_and_play(&mut cloner, text, None)?;
            println!();
        }
    }

    Ok(())
}
