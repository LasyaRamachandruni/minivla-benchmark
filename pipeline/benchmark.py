"""Latency measurement and the statistics we report.

Rules every comparison follows:
* the baseline and the variant run on the same device (CPU vs CPU, GPU vs GPU);
* warmup runs are discarded;
* at least 50 timed runs (enforced unless `allow_small_n=True`, which tests use);
* we report p50 / p95 / p99 / mean, a 95% CI for the mean (normal approximation) and
  bootstrap 95% CIs for p50 and p95;
* CUDA work is synchronised before the clock stops.
"""

import os
import platform
import subprocess
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

MIN_TIMED_RUNS = 50


def time_runs(fn: Callable[[], object], num_runs: int, warmup: int,
              sync: Optional[Callable[[], None]] = None, allow_small_n: bool = False) -> np.ndarray:
    """Call `fn` `warmup` times untimed, then `num_runs` times timed. Returns latencies in ms."""
    if num_runs < MIN_TIMED_RUNS and not allow_small_n:
        raise ValueError(f"num_runs={num_runs}: use at least {MIN_TIMED_RUNS} timed runs for reportable numbers")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    lat = np.empty(num_runs)
    for i in range(num_runs):
        t0 = time.perf_counter()
        fn()
        sync()
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return lat


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> List[float]:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    return [float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))]


def _p50(a):
    return float(np.percentile(a, 50))


def _p95(a):
    return float(np.percentile(a, 95))


def latency_stats(lat_ms: np.ndarray, batch_size: int = 1) -> dict:
    lat = np.asarray(lat_ms, dtype=float)
    n = len(lat)
    mean = float(lat.mean())
    std = float(lat.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "batch_size": batch_size,
        "mean_ms": mean,
        "std_ms": std,
        "mean_ci95_ms": [mean - half, mean + half],
        "p50_ms": _p50(lat),
        "p50_ci95_ms": bootstrap_ci(lat, _p50),
        "p95_ms": _p95(lat),
        "p95_ci95_ms": bootstrap_ci(lat, _p95, seed=1),
        "p99_ms": float(np.percentile(lat, 99)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "throughput_samples_per_s": float(1000.0 * batch_size / mean) if mean > 0 else 0.0,
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(pkg: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _nvidia_driver() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] or None
    except Exception:
        return None


def environment_info(device: str) -> dict:
    info = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "quantized_engine": torch.backends.quantized.engine,
        "lerobot": _version("lerobot"),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "numpy": np.__version__,
        "git_commit": _git_commit(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        info.update({
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 1),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "cuda_driver": _nvidia_driver(),
        })
    return info


def process_rss_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


class PeakMemory:
    """Peak CUDA memory allocated inside the block (MB); process RSS after the block on CPU."""

    def __init__(self, device: str):
        self.device = device
        self.result: Dict[str, float] = {}

    def __enter__(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            self.result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        self.result["process_rss_mb"] = process_rss_mb()
        return False
