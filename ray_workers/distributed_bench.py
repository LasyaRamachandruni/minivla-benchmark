"""Distributed benchmarking with Ray workers."""

import os
import sys
import time
from typing import List, Optional

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_distributed_benchmark(
    model_name: str = "mock",
    model_path: Optional[str] = None,
    num_workers: int = 4,
    num_requests: int = 200,
    prompt: str = "pick up the red block",
) -> dict:
    """Run distributed inference benchmark across Ray workers.

    Spawns `num_workers` Ray actors, distributes `num_requests` across them,
    and measures throughput and latency scaling.

    Args:
        model_name: Model to load on each worker.
        model_path: Path to ONNX model (optional).
        num_workers: Number of parallel Ray actors.
        num_requests: Total number of inference requests.
        prompt: Text prompt for all requests.

    Returns:
        Dict with benchmark results and per-worker stats.
    """
    import ray
    from ray_workers.actor import VLAInferenceActor

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    print(f"Spawning {num_workers} Ray workers...")
    workers = [
        VLAInferenceActor.remote(model_name=model_name, model_path=model_path)
        for _ in range(num_workers)
    ]

    # Wait for all workers to be ready
    ready_checks = [w.health_check.remote() for w in workers]
    ray.get(ready_checks)
    print(f"All {num_workers} workers ready.")

    # Distribute requests round-robin
    print(f"Sending {num_requests} requests across {num_workers} workers...")
    pending_refs = []
    wall_start = time.perf_counter()

    for i in range(num_requests):
        worker = workers[i % num_workers]
        ref = worker.infer.remote(image_bytes=None, prompt=prompt)
        pending_refs.append(ref)

    # Collect all results
    results = ray.get(pending_refs)
    wall_elapsed = time.perf_counter() - wall_start

    # Analyze results
    latencies = np.array([r["latency_ms"] for r in results])
    wall_throughput = num_requests / wall_elapsed

    # Per-worker stats
    worker_stats = ray.get([w.get_stats.remote() for w in workers])

    # Cleanup
    for w in workers:
        ray.kill(w)

    summary = {
        "num_workers": num_workers,
        "num_requests": num_requests,
        "wall_time_sec": wall_elapsed,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_latency_ms": float(np.mean(latencies)),
        "throughput_qps": wall_throughput,
        "worker_stats": worker_stats,
    }

    return summary


def compare_scaling(
    model_name: str = "mock",
    model_path: Optional[str] = None,
    worker_counts: List[int] = None,
    num_requests: int = 200,
) -> List[dict]:
    """Compare throughput across different worker counts.

    Args:
        model_name: Model to benchmark.
        model_path: Optional ONNX model path.
        worker_counts: List of worker counts to test.
        num_requests: Requests per configuration.

    Returns:
        List of benchmark results, one per worker count.
    """
    if worker_counts is None:
        worker_counts = [1, 2, 4]

    all_results = []

    for n in worker_counts:
        print(f"\n{'='*60}")
        print(f"Benchmarking with {n} worker(s)...")
        print(f"{'='*60}")

        result = run_distributed_benchmark(
            model_name=model_name,
            model_path=model_path,
            num_workers=n,
            num_requests=num_requests,
        )
        all_results.append(result)

        print(f"  Throughput: {result['throughput_qps']:.2f} QPS")
        print(f"  p50 Latency: {result['p50_latency_ms']:.2f} ms")
        print(f"  p95 Latency: {result['p95_latency_ms']:.2f} ms")

    # Print scaling summary
    if len(all_results) > 1:
        baseline_qps = all_results[0]["throughput_qps"]
        print(f"\n{'='*60}")
        print("Scaling Summary")
        print(f"{'='*60}")
        for res, n in zip(all_results, worker_counts):
            speedup = res["throughput_qps"] / baseline_qps if baseline_qps > 0 else 0
            print(f"  {n} workers: {res['throughput_qps']:.2f} QPS "
                  f"({speedup:.2f}x vs 1 worker)")

    return all_results
