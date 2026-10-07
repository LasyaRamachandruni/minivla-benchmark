"""Pipelined VLA inference: preprocess -> policy -> postprocess as three Ray actors.

The split follows the real structure of a LeRobot policy:
  1. PreprocessStage  (CPU): image conversion, tokenization, state normalization;
  2. PolicyStage      (CPU/GPU): the SmolVLA forward pass (VLM prefix + flow-matching steps);
  3. PostprocessStage (CPU): action un-normalization.

Requests are submitted back-to-back and Ray passes ObjectRefs between the stages, so batch i+1
can be preprocessed while batch i is in the policy. Stage classes are plain Python so they can
be tested without Ray; nothing here invents outputs - every action comes from the policy.
"""

import time
from typing import List, Sequence

import numpy as np

from models.frames import Frame


def _to_device(batch: dict, device: str) -> dict:
    import torch

    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _to_cpu(batch: dict) -> dict:
    import torch

    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class PreprocessStage:
    def __init__(self, model_name: str):
        from models.load_model import load_model

        # weights are not needed to preprocess
        self.model = load_model(model_name, device="cpu", load_policy=False) if model_name != "mock" else \
            load_model(model_name, device="cpu")

    def run(self, frame_dicts: List[dict], seeds: Sequence[int]) -> dict:
        t0 = time.perf_counter()
        frames = [Frame.from_dict(d) for d in frame_dicts]
        batch = _to_cpu(self.model.prepare(frames))
        return {"batch": batch, "seeds": list(seeds), "timings": {"preprocess": (time.perf_counter() - t0) * 1000}}


class PolicyStage:
    def __init__(self, model_name: str, device: str = "cpu", variant: str = "fp32", prune_amount: float = 0.2):
        from models.load_model import load_model
        from pipeline.optimize import build_variant

        base = load_model(model_name, device=device)
        self.model, _ = build_variant(base, variant, prune_amount)

    def run(self, item: dict) -> dict:
        import torch

        t0 = time.perf_counter()
        batch = _to_device(item["batch"], self.model.device)
        noise = self.model.make_noise(item["seeds"])
        with torch.inference_mode():
            actions = self.model.forward(batch, noise)
        self.model.synchronize()
        out = {"actions": actions.float().cpu(), "timings": dict(item["timings"])}
        out["timings"]["policy"] = (time.perf_counter() - t0) * 1000
        return out


class PostprocessStage:
    def __init__(self, model_name: str):
        from models.load_model import load_model

        self.model = load_model(model_name, device="cpu", load_policy=False) if model_name != "mock" else \
            load_model(model_name, device="cpu")

    def run(self, item: dict) -> dict:
        t0 = time.perf_counter()
        actions = self.model.postprocess(item["actions"]).float().numpy()
        timings = dict(item["timings"])
        timings["postprocess"] = (time.perf_counter() - t0) * 1000
        return {"actions": actions, "timings": timings}


def run_pipeline_benchmark(model_name: str, frames: Sequence[Frame], batch_size: int = 1, device: str = "cpu",
                           variant: str = "fp32") -> dict:
    import ray

    from ray_workers import init_ray
    from ray_workers.distributed_bench import make_batches

    init_ray(log_to_driver=False)
    gpu = device.startswith("cuda")
    # Fractional CPU reservations so all three stages fit on a 2-core machine.
    pre = ray.remote(num_cpus=0.25)(PreprocessStage).remote(model_name)
    pol = ray.remote(num_cpus=1, num_gpus=1 if gpu else 0)(PolicyStage).remote(model_name, device, variant)
    post = ray.remote(num_cpus=0.25)(PostprocessStage).remote(model_name)

    batches = make_batches(frames, batch_size)
    ray.get(post.run.remote(pol.run.remote(pre.run.remote(*batches[0]))))  # warmup

    t0 = time.perf_counter()
    submit, refs = [], []
    for fd, seeds in batches:
        submit.append(time.perf_counter())
        refs.append(post.run.remote(pol.run.remote(pre.run.remote(fd, seeds))))
    done_at = {}
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        done_at[ready[0]] = time.perf_counter()
    wall = time.perf_counter() - t0
    outs = ray.get(refs)
    for a in (pre, pol, post):
        ray.kill(a)

    actions = np.concatenate([o["actions"] for o in outs])
    if len(actions) != len(frames):
        raise RuntimeError(f"expected {len(frames)} action chunks, got {len(actions)}")
    e2e = np.array([(done_at[r] - s) * 1000 for r, s in zip(refs, submit)])
    stage_ms = {k: np.array([o["timings"][k] for o in outs]) for k in ("preprocess", "policy", "postprocess")}
    total = sum(v.mean() for v in stage_ms.values())
    return {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "num_frames": len(frames),
        "wall_time_s": wall,
        "throughput_frames_per_s": len(frames) / wall,
        "e2e_p50_ms": float(np.percentile(e2e, 50)),
        "e2e_p95_ms": float(np.percentile(e2e, 95)),
        "e2e_note": "submit-to-completion per batch; includes queueing behind earlier batches",
        "stage_stats": {
            k: {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)),
                "avg_ms": float(v.mean()), "pct_of_stage_total": float(v.mean() / total * 100)}
            for k, v in stage_ms.items()
        },
    }
