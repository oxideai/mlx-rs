#!/usr/bin/env python3
# Test a single sequence length (specified via command line arg)
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import time

def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 127

    model, tokenizer = load("mlx-community/GLM-4.5-Air-3bit")
    mx.set_memory_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    mx.set_default_device(mx.gpu)

    prompt = mx.array([[i for i in range(1, seq_len + 1)]], dtype=mx.uint32)

    # Warmup (5 runs)
    for _ in range(5):
        cache = make_prompt_cache(model)
        logits = model(prompt, cache=cache)
        mx.eval(logits)
    mx.synchronize()

    # Measure (10 runs)
    times = []
    for _ in range(10):
        cache = make_prompt_cache(model)
        start = time.perf_counter()
        logits = model(prompt, cache=cache)
        mx.eval(logits)
        mx.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    print(f"seq_len={seq_len}: avg={avg:.1f}ms, min={min_t:.1f}ms, max={max_t:.1f}ms")

if __name__ == "__main__":
    main()
