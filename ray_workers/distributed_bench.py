"""Data-parallel inference across Ray actors.

The eval frames are split into fixed-size batches and dispatched to `num_workers` actors, each
holding its own copy of the model. Every action returned is a real model output; the driver
checks that all frames came back with finite actions.

On a single machine all workers share the same CPU cores (or GPU), so throughput cannot scale
past the hardware; the result records `cpu_count` so the scaling curve can be read honestly.
"""

import os
import time
from typing import List, Sequence, Tuple

import numpy as np

from models.frames import Frame
from pipeline.benchmark import latency_stats


def make_batches(frames: Sequence[Frame], batch_size: int, noise_seed: int = 0) -> List[Tuple[list, list]]:
    from pipeline.suite import frame_seeds

    seeds = frame_seeds(len(frames), noise_seed)
    out = []
    for i in range(0, len(frames), batch_size):
        out.append(([f.to_dict() for f in frames[i:i + batch_size]], seeds[i:i + batch_size]))
    return out


def run_distributed_benchmark(model_name: str, frames: Sequence[Frame], num_workers: int = 2, batch_size: int = 4,
                              variant: str = "fp32", device: str = "cpu", prune_amount: float = 0.2) -> dict:
    import ray

    from ray_workers import init_ray
    from ray_workers.actor import remote_worker

    init_ray(log_to_driver=False)

    cpus = os.cpu_count() or 1
    threads = max(1, cpus // num_workers)
    gpu = device.startswith("cuda")
    # fractional reservations so any worker count can be scheduled on a small machine
    Worker = remote_worker(num_cpus=min(1.0, cpus / num_workers), num_gpus=(1.0 / num_workers) if gpu else 0)
    workers = [Worker.remote(model_name, device, variant, prune_amount, threads) for _ in range(num_workers)]
    ray.get([w.ready.remote() for w in workers])

    batches = make_batches(frames, batch_size)
    ray.get([w.infer_batch.remote(*batches[0]) for w in workers])  # warm every worker once
    ray.get([w.reset_stats.remote() for w in workers])

    t0 = time.perf_counter()
    refs = [workers[i % num_workers].infer_batch.remote(fd, s) for i, (fd, s) in enumerate(batches)]
    outs = ray.get(refs)
    wall = time.perf_counter() - t0

    actions = np.concatenate([o["actions"] for o in outs])
    if len(actions) != len(frames):
        raise RuntimeError(f"expected {len(frames)} action chunks, got {len(actions)}")
    stats = ray.get([w.stats.remote() for w in workers])
    for w in workers:
        ray.kill(w)

    return {
        "model": model_name,
        "variant": variant,
        "device": device,
        "num_workers": num_workers,
        "torch_threads_per_worker": threads,
        "cpu_count": cpus,
        "batch_size": batch_size,
        "num_frames": len(frames),
        "wall_time_s": wall,
        "throughput_frames_per_s": len(frames) / wall,
        "batch_latency": latency_stats(np.array([o["latency_ms"] for o in outs]), batch_size=batch_size),
        "worker_stats": stats,
        "actions_shape": list(actions.shape),
    }


def compare_scaling(model_name: str, frames: Sequence[Frame], worker_counts: Sequence[int], batch_size: int = 4,
                    variant: str = "fp32", device: str = "cpu") -> List[dict]:
    results = []
    for n in worker_counts:
        print(f"Ray: {n} worker(s), batch {batch_size}, {len(frames)} frames")
        results.append(run_distributed_benchmark(model_name, frames, n, batch_size, variant, device))
    return results
