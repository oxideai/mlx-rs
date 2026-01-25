# TTS Performance Benchmark: Rust MLX vs Python MPS

Benchmark comparing the Rust MLX implementation against Python dora-primespeech with MPS acceleration.

## Test Configuration

- **Date**: January 2025
- **Hardware**: Apple Silicon (M-series)
- **Voice**: doubao (few-shot mode)
- **Text**: 91 Chinese characters with decimals and percentages

**Test text**:
```
从季节上看，主要是增在秋粮，2025年秋粮增产163.6亿斤，占全年粮食增量九成多。
从区域上看，主要增在东北三省、内蒙古和新疆，这5个省粮食增产114.7亿斤，占全国增量接近70%。
```

## Benchmark Results (5 runs each)

### Synthesis Time

| Run | Rust MLX (ms) | Python MPS (ms) |
|-----|---------------|-----------------|
| 1   | 4,936         | 17,009 (warmup) |
| 2   | 4,886         | 9,537           |
| 3   | 4,889         | 9,430           |
| 4   | 4,953         | 9,608           |
| 5   | 4,861         | 9,665           |
| **Average** | **4,905** | **9,560** (excl. warmup) |
| **Std Dev** | ±47 (1.0%) | ±106 (1.1%) |

### Summary

| Metric | Rust (MLX) | Python (MPS) | Speedup |
|--------|------------|--------------|---------|
| Model load | 49ms | 3,895ms | **79x** |
| Synthesis (avg) | 4,905ms | 9,560ms | **1.95x** |
| Audio duration | 19.72s | 22.27s | similar |
| RTF (realtime factor) | **4.02x** | 2.33x | 1.7x |
| Tokens generated | 493 | N/A | - |
| Sample rate | 32kHz | 32kHz | same |

## Key Findings

1. **Rust is ~2x faster** at synthesis than Python with MPS
2. **Rust runs at 4x realtime** - generates 19.72s audio in 4.9s
3. **Python runs at 2.3x realtime** - generates 22.27s audio in 9.6s
4. **Model loading is 79x faster** in Rust (49ms vs 3.9s)
5. Both implementations show **excellent consistency** (±1-2% variance)

## Realtime Factor Comparison

```
Rust:   |████████████████████████████████████████| 4.02x realtime
Python: |███████████████████████|                  2.33x realtime
```

## Benchmark Commands

### Rust

```bash
TEXT='从季节上看，主要是增在秋粮，2025年秋粮增产163.6亿斤，占全年粮食增量九成多。从区域上看，主要增在东北三省、内蒙古和新疆，这5个省粮食增产114.7亿斤，占全国增量接近70%。'

echo "=== RUST (5 runs) ==="
for i in 1 2 3 4 5; do
  echo "Run $i:"
  cargo run --release --example voice_clone -- --text "$TEXT" --voice doubao 2>&1 | grep "Generated.*tokens in"
done
```

### Python

```bash
cd ~/home/mofa-studio/models/setup-local-models/primespeech-validation
python test_tts_direct.py --voice doubao --device mps
```

Or use this inline script:

```python
import sys, time, os
from pathlib import Path

primespeech_path = Path(os.path.expanduser('~/home/mofa-studio/node-hub/dora-primespeech')).resolve()
model_dir = Path(os.path.expanduser('~/.dora/models/primespeech')).resolve()
sys.path.insert(0, str(primespeech_path))
os.environ['PRIMESPEECH_MODEL_DIR'] = str(model_dir)

from dora_primespeech.moyoyo_tts_wrapper_streaming_fix import StreamingMoYoYoTTSWrapper

TEXT = '从季节上看，主要是增在秋粮，2025年秋粮增产163.6亿斤，占全年粮食增量九成多。从区域上看，主要增在东北三省、内蒙古和新疆，这5个省粮食增产114.7亿斤，占全国增量接近70%。'

wrapper = StreamingMoYoYoTTSWrapper(voice='doubao', device='mps', enable_streaming=False)

for i in range(1, 6):
    start = time.time()
    sample_rate, audio = wrapper.synthesize(TEXT, language='zh', speed=1.0)
    elapsed = time.time() - start
    print(f"Run {i}: {elapsed*1000:.1f}ms, Duration {len(audio)/sample_rate:.2f}s")
```

## Notes

- Python first run includes JIT compilation overhead (~17s vs ~9.5s for subsequent runs)
- Rust has minimal warmup effect due to ahead-of-time compilation
- Both implementations produce similar quality audio
- Audio duration difference (19.72s vs 22.27s) is due to different token generation patterns
