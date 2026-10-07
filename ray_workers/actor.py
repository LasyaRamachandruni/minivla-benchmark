"""A Ray actor that runs real, batched VLA inference.

The worker logic is a plain class (`VLAWorker`) so it can be unit-tested without Ray;
`remote_worker()` wraps it with `ray.remote` at runtime. Workers only ever return actions
computed by the model from the frames they were given - there is no fallback output.
"""

import time
from typing import List, Optional, Sequence

import numpy as np


class VLAWorker:
    def __init__(self, model_name: str = "mock", device: str = "cpu", variant: str = "fp32",
                 prune_amount: float = 0.2, num_threads: Optional[int] = None):
        import torch

        from models.load_model import load_model
        from pipeline.optimize import build_variant

        if num_threads:
            torch.set_num_threads(num_threads)
        base = load_model(model_name, device=device)
        self.model, _ = build_variant(base, variant, prune_amount)
        self.variant = variant
        self.frames_done = 0
        self.busy_ms = 0.0
        self.batches = 0

    def infer_batch(self, frame_dicts: List[dict], seeds: Sequence[int]) -> dict:
        from models.frames import Frame

        if not frame_dicts:
            raise ValueError("empty batch")
        frames = [Frame.from_dict(d) for d in frame_dicts]
        t0 = time.perf_counter()
        actions = self.model.predict(frames, seeds, batch_size=len(frames))
        ms = (time.perf_counter() - t0) * 1000
        if not np.isfinite(actions).all():
            raise FloatingPointError("model produced non-finite actions")
        self.frames_done += len(frames)
        self.busy_ms += ms
        self.batches += 1
        return {"actions": actions, "latency_ms": ms, "batch_size": len(frames)}

    def stats(self) -> dict:
        return {"variant": self.variant, "frames": self.frames_done, "batches": self.batches,
                "busy_ms": self.busy_ms,
                "avg_batch_ms": self.busy_ms / self.batches if self.batches else 0.0}

    def reset_stats(self) -> None:
        self.frames_done, self.busy_ms, self.batches = 0, 0.0, 0

    def ready(self) -> bool:
        return True


def remote_worker(num_cpus: float = 1, num_gpus: float = 0):
    import ray

    return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(VLAWorker)
