# MiniVLA Benchmark

A CLI toolkit for benchmarking, optimizing, and scaling Vision-Language-Action (VLA) inference pipelines — from single-device optimization to distributed Ray-based execution.

## Motivation

VLA models combine vision, language, and action prediction into a single model for robotics and embodied AI. This toolkit provides a reproducible benchmarking framework to:

- Measure inference latency (p50/p95/p99), throughput, and memory usage
- Quantify the accuracy vs. performance tradeoff of ONNX export, INT8 quantization, and structured pruning
- Demonstrate horizontal scaling with Ray distributed actors

Inspired by the engineering principles described in ["Scaling VLA Pipelines for Robotics with Ray on Anyscale"](https://www.anyscale.com/blog) (Omar Shorbaji, Feb 2026).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (uses mock model by default)
chmod +x scripts/run_full_pipeline.sh
./scripts/run_full_pipeline.sh

# Or run stages individually:
python cli.py benchmark --model mock --N 100
python cli.py optimize --model mock --quant-type int8 --prune-amount 0.3
python cli.py ray-bench --model mock --workers 4 --N 200 --scaling-test
python cli.py compare --results results/results_table.csv --plot
```

## CLI Commands

### `benchmark` — Stage 1: Single-Device Inference

```bash
python cli.py benchmark --model mobilevlm --input sample.jpg --prompt "pick up the red block" --N 100
```

Measures per-run latency (p50, p95, p99), throughput (QPS), peak memory, and model size.

### `optimize` — Stage 2: Optimization Pipeline

```bash
python cli.py optimize --model mock --quant-type int8 --prune-amount 0.3 --output-dir optimized
```

Runs the full optimization pipeline:
1. ONNX export of baseline model
2. Dynamic INT8 quantization
3. Structured pruning (channel pruning on Linear/Conv2d layers)
4. Export pruned model → ONNX → quantize

Outputs a side-by-side comparison table.

### `ray-bench` — Stage 3: Distributed Ray Benchmark

```bash
python cli.py ray-bench --model mock --workers 4 --N 200 --scaling-test
```

- Wraps the inference pipeline as a Ray actor
- Distributes requests across workers round-robin
- `--scaling-test` compares 1, 2, and 4 workers automatically
- Reports throughput scaling and per-worker utilization

### `compare` — Stage 4: Results Comparison

```bash
python cli.py compare --results results/results_table.csv --plot
```

Displays a formatted comparison table and generates bar charts for latency, throughput, and model size.

## Supported Models

| Model | ID | Notes |
|-------|------|-------|
| **SmolVLM-256M** | `smolvlm` | 256M params, runs on CPU/MPS, recommended default |
| Mock VLA | `mock` | Lightweight CNN+Transformer for pipeline testing (no downloads) |
| MobileVLM v2 | `mobilevlm` | 1.7B params, runs on CPU/single GPU |
| LLaVA-1.5 | `llava` | 7B params, requires GPU |

## Project Structure

```
minivla-benchmark/
├── cli.py                     # Unified CLI entry point
├── models/
│   └── load_model.py          # Model loading (PyTorch, ONNX, mock)
├── pipeline/
│   ├── infer.py               # Single inference run
│   ├── benchmark.py           # Latency, throughput, memory measurement
│   ├── optimize.py            # ONNX export, quantization, pruning
│   └── evaluate.py            # Accuracy evaluation
├── ray_workers/
│   ├── actor.py               # Ray actor wrapping inference
│   └── distributed_bench.py   # Multi-worker benchmarking
├── results/                   # Benchmark outputs
└── scripts/
    └── run_full_pipeline.sh   # One-command full pipeline
```

## Results Table (Example Output)

| Configuration | Size (MB) | p50 (ms) | p95 (ms) | Throughput (QPS) | Accuracy (%) |
|---------------|-----------|----------|----------|------------------|--------------|
| Baseline PyTorch | 4.2 | 12.34 | 15.67 | 78.50 | 100.0 |
| ONNX Export | 4.1 | 8.91 | 11.23 | 108.30 | 100.0 |
| INT8 Quantized | 1.8 | 6.45 | 8.12 | 149.70 | 98.5 |
| Pruned + Quantized | 1.2 | 5.23 | 6.89 | 183.40 | 95.2 |
| Ray (4 workers) | — | 12.50 | 18.30 | 312.00 | — |

## Tech Stack

- **Models**: PyTorch, HuggingFace Transformers
- **Optimization**: ONNX Runtime, dynamic quantization, structured pruning
- **Distributed execution**: Ray Core (actors)
- **CLI**: Click
- **Benchmarking**: psutil, numpy, time.perf_counter
- **Visualization**: Matplotlib, tabulate
