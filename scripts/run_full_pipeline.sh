#!/usr/bin/env bash
# Full benchmark: sample LIBERO frames once, run every variant on GPU (if present) and CPU,
# then write results/RESULTS.md and one plot per results JSON.
#
#   ./scripts/run_full_pipeline.sh [model] [gpu_frames] [cpu_frames]
#
# Defaults: smolvla_libero, 500 frames on GPU, 50 frames on CPU (SmolVLA takes seconds per
# sample on a 2-vCPU machine, so the CPU section uses a seeded subset of the same frames).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-smolvla_libero}"
GPU_FRAMES="${2:-500}"
CPU_FRAMES="${3:-50}"
CACHE="data_cache/libero_frames.npz"

python cli.py sample-libero --num-frames "$GPU_FRAMES" --cache "$CACHE"

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    python cli.py optimize --model "$MODEL" --device cuda --frames-cache "$CACHE" \
        --num-frames "$GPU_FRAMES" --eval-batch-size 16 --N 100 --warmup 10
fi

python cli.py optimize --model "$MODEL" --device cpu --frames-cache "$CACHE" \
    --num-frames "$CPU_FRAMES" --eval-batch-size 1 --N 50 --warmup 5

python cli.py compare --results "results/*.json" --markdown results/RESULTS.md
