"""Benchmarking utilities: latency, throughput, memory measurement."""

import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import psutil
import torch
from PIL import Image

from models.load_model import ModelInfo, create_sample_input
from pipeline.infer import run_inference


@dataclass
class BenchmarkResult:
    config_name: str
    model_name: str
    backend: str
    device: str
    size_mb: float
    num_runs: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    throughput_qps: float
    peak_memory_mb: float
    accuracy_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_row(self) -> list:
        return [
            self.config_name,
            f"{self.size_mb:.1f}",
            f"{self.p50_latency_ms:.2f}",
            f"{self.p95_latency_ms:.2f}",
            f"{self.p99_latency_ms:.2f}",
            f"{self.throughput_qps:.2f}",
            f"{self.peak_memory_mb:.1f}",
            f"{self.accuracy_pct:.1f}",
        ]

    @staticmethod
    def table_headers() -> list:
        return [
            "Configuration", "Size (MB)", "p50 (ms)", "p95 (ms)",
            "p99 (ms)", "Throughput (QPS)", "Peak Mem (MB)", "Accuracy (%)",
        ]


def measure_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    return mem


def measure_gpu_memory() -> float:
    """Get current GPU memory usage in MB (CUDA only)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def run_benchmark(
    model_info: ModelInfo,
    config_name: str = "Baseline PyTorch",
    num_runs: int = 100,
    warmup_runs: int = 10,
    image: Optional[Image.Image] = None,
    prompt: str = "pick up the red block",
    accuracy_pct: float = 0.0,
) -> BenchmarkResult:
    """Run a full benchmark: warmup + timed runs.

    Args:
        model_info: Loaded model info.
        config_name: Label for this benchmark configuration.
        num_runs: Number of timed inference runs.
        warmup_runs: Number of warmup runs (not timed).
        image: Input image (random if None).
        prompt: Text prompt for the model.
        accuracy_pct: Pre-computed accuracy to include in results.

    Returns:
        BenchmarkResult with all measurements.
    """
    inputs = create_sample_input(model_info.processor, model_info.device, image, prompt)

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mem_before = measure_memory_usage()

    # Warmup
    from tqdm import tqdm
    for _ in tqdm(range(warmup_runs), desc="  Warmup", leave=False):
        run_inference(model_info, inputs)

    # Timed runs
    latencies = []
    sample_output = None
    for i in tqdm(range(num_runs), desc="  Benchmark"):
        result = run_inference(model_info, inputs)
        latencies.append(result["latency_ms"])
        if sample_output is None:
            sample_output = result

    # Show sample output for real VLMs
    if sample_output and "text" in sample_output:
        print(f'  Sample output: "{sample_output["text"][:120]}"')

    mem_after = measure_memory_usage()
    gpu_mem = measure_gpu_memory()

    latencies = np.array(latencies)
    total_time_sec = latencies.sum() / 1000.0

    peak_mem = max(mem_after - mem_before, gpu_mem) if gpu_mem > 0 else mem_after

    return BenchmarkResult(
        config_name=config_name,
        model_name=model_info.name,
        backend=model_info.backend,
        device=model_info.device,
        size_mb=model_info.size_mb,
        num_runs=num_runs,
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        mean_latency_ms=float(np.mean(latencies)),
        throughput_qps=num_runs / total_time_sec if total_time_sec > 0 else 0,
        peak_memory_mb=peak_mem,
        accuracy_pct=accuracy_pct,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print a formatted results table."""
    from tabulate import tabulate

    headers = BenchmarkResult.table_headers()
    rows = [r.summary_row() for r in results]
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


def save_results_csv(results: List[BenchmarkResult], path: str):
    """Save results to CSV."""
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    headers = BenchmarkResult.table_headers()
    rows = [r.summary_row() for r in results]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Results saved to {path}")
