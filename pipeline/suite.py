"""Run the FP32 baseline and every optimized variant on one device, and write one results JSON.

For each variant, on the same device and the same frames:
1. predict action chunks for every evaluation frame (fixed per-frame noise seeds),
2. score them against LIBERO ground truth (only if the checkpoint shares LIBERO's action space)
   and against the FP32 baseline's own actions,
3. time single-sample inference (warmup + N timed runs) and record size / parameters / memory.
"""

import gc
import json
import os
import re
from typing import Callable, List, Optional

import numpy as np
import torch

from models.frames import Frame, stack_ground_truth
from models.vla import VLAModel
from pipeline.benchmark import PeakMemory, environment_info, latency_stats, time_runs
from pipeline.evaluate import action_errors, agreement, degradation
from pipeline.optimize import VariantUnavailable, build_variant

SCHEMA_VERSION = 1


def frame_seeds(n: int, noise_seed: int) -> List[int]:
    return [noise_seed * 1_000_003 + i for i in range(n)]


def benchmark_latency(model: VLAModel, frames: List[Frame], seeds: List[int], num_runs: int, warmup: int,
                      allow_small_n: bool = False) -> dict:
    """Latency of `prepare` (CPU preprocessing) and of `run` (policy + postprocess) on a fixed input."""
    batch = model.prepare(frames)
    noise = model.make_noise(seeds)

    def infer():
        with torch.inference_mode():
            model.run(batch, noise)

    def prep():
        model.prepare(frames)

    lat = time_runs(infer, num_runs, warmup, sync=model.synchronize, allow_small_n=allow_small_n)
    pre = time_runs(prep, num_runs, max(1, warmup // 2), sync=model.synchronize, allow_small_n=allow_small_n)
    return {"inference": latency_stats(lat, batch_size=len(frames)),
            "preprocess": latency_stats(pre, batch_size=len(frames)),
            "raw_inference_ms": [round(float(x), 4) for x in lat]}


def default_output_path(model_key: str, device: str, results_dir: str = "results") -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        tag = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower()).strip("-")
    else:
        tag = "cpu"
    return os.path.join(results_dir, f"{model_key}_{tag}.json")


def run_suite(
    base: VLAModel,
    frames: List[Frame],
    frames_info: dict,
    variants: List[str],
    num_runs: int = 100,
    warmup: int = 10,
    eval_batch_size: int = 1,
    latency_batch_size: int = 1,
    prune_amount: float = 0.2,
    noise_seed: int = 0,
    horizon: Optional[int] = None,
    out_path: Optional[str] = None,
    allow_small_n: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    device = base.device
    variants = ["fp32"] + [v for v in variants if v != "fp32"]
    seeds = frame_seeds(len(frames), noise_seed)
    alt_seeds = frame_seeds(len(frames), noise_seed + 1)

    gt, gt_pad = stack_ground_truth(frames)
    gt_ok = bool(base.spec.gt_comparable and gt is not None and gt.shape[-1] == base.action_dim)
    if gt is not None and not gt_ok:
        why = ("checkpoint is not trained on this dataset's action space" if not base.spec.gt_comparable
               else f"action dim {base.action_dim} != dataset {gt.shape[-1]}")
        log(f"Ground-truth metrics disabled for {base.name}: {why}")

    results = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment_info(device),
        "model": base.describe(),
        "dataset": frames_info,
        "protocol": {
            "device": device,
            "variants": variants,
            "timed_runs": num_runs,
            "warmup_runs": warmup,
            "latency_batch_size": latency_batch_size,
            "eval_batch_size": eval_batch_size,
            "eval_frames": len(frames),
            "noise_seed": noise_seed,
            "prune_amount": prune_amount,
            "ground_truth_metrics": gt_ok,
            "latency_input": "fixed: the first eval frame(s), same noise every run",
        },
        "variants": [],
    }

    ref_preds = None
    baseline_gt = None
    for v in variants:
        log(f"[{v}] building on {device}")
        try:
            model, build_info = build_variant(base, v, prune_amount)
        except VariantUnavailable as e:
            log(f"[{v}] skipped: {e}")
            results["variants"].append({"variant": v, "skipped": str(e)})
            continue

        entry = {"variant": v, "precision": model.precision, "device": model.device,
                 "size_mb": model.size_mb(), "num_parameters": model.num_parameters(), "build": build_info}
        offloaded = device.startswith("cuda") and model is not base
        if offloaded:  # keep the baseline's weights out of the variant's peak-memory number
            base.policy.to("cpu")
            torch.cuda.empty_cache()
        with PeakMemory(device) as mem:
            log(f"[{v}] predicting {len(frames)} frames")
            preds = model.predict(frames, seeds, batch_size=eval_batch_size)
            log(f"[{v}] timing {warmup} warmup + {num_runs} runs")
            entry["latency"] = benchmark_latency(model, frames[:latency_batch_size], seeds[:latency_batch_size],
                                                 num_runs, warmup, allow_small_n)
        entry["memory"] = mem.result

        if v == "fp32":
            ref_preds = preds
            alt = model.predict(frames, alt_seeds, batch_size=eval_batch_size)
            entry["noise_floor"] = agreement(alt, ref_preds, horizon)
        entry["agreement_with_fp32"] = agreement(preds, ref_preds, horizon)
        if gt_ok:
            entry["vs_ground_truth"] = action_errors(preds, gt, gt_pad, horizon)
            if v == "fp32":
                baseline_gt = entry["vs_ground_truth"]
            else:
                entry["degradation_vs_fp32"] = degradation(entry["vs_ground_truth"], baseline_gt)
        inf = entry["latency"]["inference"]
        log(f"[{v}] p50 {inf['p50_ms']:.2f} ms  p95 {inf['p95_ms']:.2f} ms  size {entry['size_mb']:.1f} MB  "
            f"rel-L2 vs fp32 {entry['agreement_with_fp32']['rel_l2_mean']:.4f}")
        results["variants"].append(entry)

        if model is not base:
            del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if offloaded:
            base.policy.to(device)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        log(f"Results written to {out_path}")
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
