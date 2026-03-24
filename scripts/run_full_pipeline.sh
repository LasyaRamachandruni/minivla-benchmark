#!/usr/bin/env bash
# MiniVLA Benchmark — Full Pipeline Runner
# Runs all stages: benchmark, optimize, ray-bench, compare
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-mock}"
N_BENCH="${2:-100}"
N_RAY="${3:-200}"
WORKERS="${4:-4}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "MiniVLA Benchmark — Full Pipeline"
echo "Model: $MODEL | Bench runs: $N_BENCH | Ray requests: $N_RAY | Workers: $WORKERS"
echo "============================================"

# Stage 1: Single-device benchmark
echo ""
echo "[Stage 1] Single-device benchmark..."
python3 cli.py benchmark \
    --model "$MODEL" \
    --N "$N_BENCH" \
    --output "$RESULTS_DIR/stage1_baseline.csv"

# Stage 2: Optimization pipeline
echo ""
echo "[Stage 2] Optimization pipeline..."
python3 cli.py optimize \
    --model "$MODEL" \
    --quant-type int8 \
    --prune-amount 0.3 \
    --N "$((N_BENCH / 2))" \
    --results-output "$RESULTS_DIR/stage2_optimized.csv"

# Stage 3: Ray distributed benchmark
echo ""
echo "[Stage 3] Ray distributed benchmark..."
python3 cli.py ray-bench \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --N "$N_RAY" \
    --scaling-test \
    --results-output "$RESULTS_DIR/stage3_ray.csv"

# Stage 4: Combine and compare all results
echo ""
echo "[Stage 4] Combining results..."

# Merge all CSVs (skip headers on subsequent files)
head -1 "$RESULTS_DIR/stage1_baseline.csv" > "$RESULTS_DIR/results_table.csv"
for f in "$RESULTS_DIR"/stage*.csv; do
    tail -n +2 "$f" >> "$RESULTS_DIR/results_table.csv"
done

python3 cli.py compare \
    --results "$RESULTS_DIR/results_table.csv" \
    --plot \
    --plot-output "$RESULTS_DIR/comparison.png"

echo ""
echo "============================================"
echo "Pipeline complete! Results in $RESULTS_DIR/"
echo "============================================"
