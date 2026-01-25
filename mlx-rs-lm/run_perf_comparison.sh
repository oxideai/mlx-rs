#!/bin/bash
#
# Performance Comparison: Python mlx-lm vs Rust mlx-rs
# This script runs both benchmarks and produces a comparison report.
#
# Usage: ./run_perf_comparison.sh [--python-only|--rust-only]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_OUTPUT="python_mlx_benchmark.json"
RUST_OUTPUT="rust_mlx_benchmark.json"

echo "============================================================"
echo "MLX Performance Comparison Suite"
echo "============================================================"
echo "Date: $(date)"
echo "Directory: $SCRIPT_DIR"
echo ""

run_python() {
    echo "============================================================"
    echo "Running Python mlx-lm benchmark..."
    echo "============================================================"
    python3 perf_comparison.py -o "$PYTHON_OUTPUT"
    echo ""
}

run_rust() {
    echo "============================================================"
    echo "Running Rust mlx-rs benchmark..."
    echo "============================================================"
    cd "$SCRIPT_DIR/.."
    cargo run --release --example perf_comparison 2>&1
    mv rust_mlx_benchmark.json "$SCRIPT_DIR/$RUST_OUTPUT" 2>/dev/null || true
    cd "$SCRIPT_DIR"
    echo ""
}

compare_results() {
    echo "============================================================"
    echo "PERFORMANCE COMPARISON SUMMARY"
    echo "============================================================"

    if [[ -f "$PYTHON_OUTPUT" ]] && [[ -f "$RUST_OUTPUT" ]]; then
        python3 - <<'EOF'
import json
import sys

with open("python_mlx_benchmark.json") as f:
    py = json.load(f)
with open("rust_mlx_benchmark.json") as f:
    rs = json.load(f)

print(f"\n{'Component':<30} {'Python (ms)':>15} {'Rust (ms)':>15} {'Ratio':>10}")
print("-" * 70)

components = [
    ("MoE Forward", "moe_forward", "moe_forward_ms"),
    ("Attention Forward", "attention_forward", "attention_forward_ms"),
    ("Full Forward (Decode)", "full_forward_decode", "full_forward_decode_ms"),
    ("Full Forward (Prefill 64)", "full_forward_prefill", "full_forward_prefill_ms"),
]

for name, py_key, rs_key in components:
    py_val = py.get("components", {}).get(py_key, {}).get("avg_ms", "N/A")
    rs_val = rs.get(rs_key, "N/A")

    if py_val != "N/A" and rs_val != "N/A":
        ratio = f"{float(rs_val)/float(py_val):.2f}x"
    else:
        ratio = "N/A"

    py_str = f"{py_val:.2f}" if isinstance(py_val, (int, float)) else str(py_val)
    rs_str = f"{rs_val:.2f}" if isinstance(rs_val, (int, float)) else str(rs_val)
    print(f"{name:<30} {py_str:>15} {rs_str:>15} {ratio:>10}")

print("-" * 70)

py_tps = py.get("e2e_tokens_per_sec", 0)
rs_tps = rs.get("e2e_tokens_per_sec", 0)
if py_tps and rs_tps:
    ratio = f"{py_tps/rs_tps:.2f}x"
    print(f"{'E2E Tokens/sec':<30} {py_tps:>15.1f} {rs_tps:>15.1f} {ratio:>10}")

print("\n" + "=" * 70)
if py_tps and rs_tps:
    gap = py_tps / rs_tps
    print(f"PERFORMANCE GAP: Python is {gap:.2f}x faster than Rust")
    print(f"  - Python: {py_tps:.1f} tok/s ({1000/py_tps:.1f}ms/tok)")
    print(f"  - Rust:   {rs_tps:.1f} tok/s ({1000/rs_tps:.1f}ms/tok)")
print("=" * 70)

# Save comparison report
comparison = {
    "timestamp": py.get("timestamp", ""),
    "python": py,
    "rust": rs,
    "gap_ratio": gap if py_tps and rs_tps else None
}
with open("comparison_report.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\nComparison saved to comparison_report.json")
EOF
    else
        echo "Missing benchmark files. Run both benchmarks first."
        [[ ! -f "$PYTHON_OUTPUT" ]] && echo "  Missing: $PYTHON_OUTPUT"
        [[ ! -f "$RUST_OUTPUT" ]] && echo "  Missing: $RUST_OUTPUT"
    fi
}

# Parse arguments
case "${1:-}" in
    --python-only)
        run_python
        ;;
    --rust-only)
        run_rust
        ;;
    --compare-only)
        compare_results
        ;;
    *)
        run_python
        run_rust
        compare_results
        ;;
esac
