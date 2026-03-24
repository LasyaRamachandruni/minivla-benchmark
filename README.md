# MiniVLA Benchmark

A CLI toolkit for benchmarking, optimizing, and scaling Vision-Language-Action (VLA) inference pipelines — from single-device optimization to distributed Ray-based execution.

## Motivation

VLA models combine vision, language, and action prediction into a single model for robotics and embodied AI. This toolkit provides a reproducible benchmarking framework to:

- Measure inference latency (p50/p95/p99), throughput, and memory usage
- Quantify the accuracy vs. performance tradeoff of INT8 quantization and structured pruning
- Demonstrate horizontal scaling with Ray distributed actors
- Profile pipeline-parallel execution (vision → language → action as separate actors)
- Serve models as HTTP endpoints with Ray Serve
- Process image datasets at scale with Ray Data

## Quick Start

```bash
# Install
pip install -e .

# Or just install dependencies
pip install -r requirements.txt

# Run the full pipeline
./scripts/run_full_pipeline.sh

# Or run stages individually:
python3 cli.py benchmark --model smolvlm --N 5 --warmup 2
python3 cli.py optimize --model smolvlm --quant-type int8 --prune-amount 0.3 --N 5
python3 cli.py ray-bench --model mock --workers 4 --N 200 --scaling-test
python3 cli.py pipeline-bench --model mock --N 100
python3 cli.py serve --model mock --port 8000 --replicas 2
python3 cli.py batch --model mock --N 50
python3 cli.py compare --results results/results_table.csv --plot
```

## CLI Commands

### `benchmark` — Single-Device Inference

```bash
python3 cli.py benchmark --model smolvlm --N 5 --warmup 2 --prompt "Describe this image"
```

Measures per-run latency (p50/p95/p99), throughput (QPS), peak memory, model size, and displays sample model output.

### `optimize` — Optimization Pipeline

```bash
python3 cli.py optimize --model smolvlm --quant-type int8 --prune-amount 0.3
```

Applies INT8 dynamic quantization and structured pruning, then benchmarks each variant side-by-side.

### `ray-bench` — Distributed Ray Benchmark

```bash
python3 cli.py ray-bench --model mock --workers 4 --N 200 --scaling-test
```

Distributes requests across Ray actors and generates a **scaling curve** showing throughput vs. workers with efficiency annotations.

### `pipeline-bench` — Pipeline-Parallel Inference

```bash
python3 cli.py pipeline-bench --model mock --N 100
```

Splits the VLA model into three pipeline stages (Vision Encoder → Language Decoder → Action Decoder), each running as a separate Ray actor. Measures per-stage latency and generates a breakdown chart showing where time is spent.

### `serve` — Ray Serve HTTP Endpoint

```bash
python3 cli.py serve --model mock --port 8000 --replicas 2

# Query the endpoint:
curl -X POST http://localhost:8000 \
    -H "Content-Type: application/json" \
    -d '{"prompt": "pick up the red block"}'
```

Wraps the model as a production-style Ray Serve deployment with autoscaling replicas.

### `batch` — Batch Inference with Ray Data

```bash
python3 cli.py batch --model mock --N 50 --image-dir ./my_images/
```

Processes a folder of images (or synthetic dataset) in parallel using Ray Data with stateful actor workers.

### `compare` — Results Comparison

```bash
python3 cli.py compare --results results/results_table.csv --plot
```

Displays formatted comparison table and generates bar charts.

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
├── cli.py                          # Unified CLI (8 commands)
├── pyproject.toml                  # pip install -e .
├── models/
│   └── load_model.py               # Model loading (PyTorch, ONNX, HuggingFace)
├── pipeline/
│   ├── infer.py                    # Single inference (forward pass + generate)
│   ├── benchmark.py                # Latency, throughput, memory (with tqdm)
│   ├── optimize.py                 # ONNX export, INT8 quantization, pruning
│   ├── evaluate.py                 # Accuracy evaluation
│   └── visualize.py                # Scaling curves, pipeline breakdowns
├── ray_workers/
│   ├── actor.py                    # Ray actor for distributed inference
│   ├── pipeline_actors.py          # Pipeline-parallel actors (vision/language/action)
│   ├── distributed_bench.py        # Multi-worker benchmarking
│   ├── serve_endpoint.py           # Ray Serve HTTP deployment
│   └── batch_inference.py          # Ray Data batch processing
├── results/                        # Benchmark outputs, charts
└── scripts/
    └── run_full_pipeline.sh        # One-command full pipeline
```

## SmolVLM-256M Results

| Configuration | Size (MB) | p50 Latency | Throughput (QPS) |
|---|---|---|---|
| Baseline PyTorch (MPS) | 489.2 | 5.9s | 0.17 |
| INT8 Quantized (CPU) | 113.8 | 33.8s | 0.03 |
| Pruned (MPS) | 489.2 | 5.2s | 0.19 |
| Pruned + Quantized (CPU) | 113.8 | 15.7s | 0.06 |

**Key findings:**
- Structured pruning gives 12% latency reduction on MPS
- INT8 quantization achieves 77% model size reduction (489 → 114 MB)
- Ray scaling requires distributed hardware for real throughput gains

## Tech Stack

- **Models**: PyTorch, HuggingFace Transformers (SmolVLM-256M)
- **Optimization**: ONNX Runtime, PyTorch dynamic quantization, structured pruning
- **Distributed**: Ray Core (actors, tasks), Ray Serve, Ray Data
- **CLI**: Click with tqdm progress bars
- **Visualization**: Matplotlib, tabulate
