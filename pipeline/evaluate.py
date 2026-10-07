"""Action-prediction metrics.

Two different questions, kept apart on purpose:

* **Error vs ground truth** (`action_errors`): how far the predicted action chunk is from the
  actions the LIBERO demonstrator took (MSE / L1, overall and per action dimension). Only
  meaningful for a checkpoint trained on LIBERO's action space.
* **Agreement with the FP32 model** (`agreement`): how far an optimized variant's actions are
  from the FP32 baseline's actions on the same frames with the same flow-matching noise
  (relative L2 error, MSE, L1). This isolates what the optimization changed and works for any
  checkpoint, including ones not trained on LIBERO.

`noise floor`: SmolVLA samples actions from noise, so FP32 itself gives different actions for a
different noise seed. The suite reports FP32-vs-FP32-with-other-noise agreement as a reference;
an optimization whose disagreement is below that floor changes outputs less than resampling does.
"""

from typing import Optional

import numpy as np


def _err_block(diff: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """diff: (..., A). mask: same leading shape as diff[..., 0], True = valid."""
    a = diff.shape[-1]
    flat = diff.reshape(-1, a)
    if mask is not None:
        flat = flat[mask.reshape(-1)]
    if len(flat) == 0:
        return {"mse": float("nan"), "l1": float("nan"), "per_dim_mse": [float("nan")] * a,
                "per_dim_l1": [float("nan")] * a, "count": 0}
    return {
        "mse": float(np.mean(flat ** 2)),
        "l1": float(np.mean(np.abs(flat))),
        "per_dim_mse": np.mean(flat ** 2, axis=0).tolist(),
        "per_dim_l1": np.mean(np.abs(flat), axis=0).tolist(),
        "count": int(len(flat)),
    }


def action_errors(pred: np.ndarray, gt: np.ndarray, gt_is_pad: Optional[np.ndarray] = None,
                  horizon: Optional[int] = None) -> dict:
    """Error of predicted chunks (N, chunk, A) against ground truth (N, H, A).

    `next_action` scores only the first predicted action against the action recorded at that
    frame; `chunk` scores the first `h = min(chunk, H, horizon)` steps, ignoring steps past the
    end of the episode.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape[-1] != gt.shape[-1]:
        raise ValueError(f"action dims differ: prediction {pred.shape[-1]} vs ground truth {gt.shape[-1]}")
    if gt_is_pad is None:
        gt_is_pad = np.zeros(gt.shape[:2], dtype=bool)
    h = min(pred.shape[1], gt.shape[1], horizon or gt.shape[1])
    return {
        "n_frames": int(len(pred)),
        "horizon": int(h),
        "next_action": _err_block(pred[:, 0] - gt[:, 0]),
        "chunk": _err_block(pred[:, :h] - gt[:, :h], ~gt_is_pad[:, :h]),
    }


def agreement(pred: np.ndarray, ref: np.ndarray, horizon: Optional[int] = None) -> dict:
    """How closely `pred` matches `ref` (both (N, chunk, A)) over the first `horizon` steps."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    h = min(pred.shape[1], horizon or pred.shape[1])
    d = pred[:, :h] - ref[:, :h]
    num = np.sqrt((d ** 2).sum(axis=(1, 2)))
    den = np.sqrt((ref[:, :h] ** 2).sum(axis=(1, 2)))
    rel = num / np.maximum(den, 1e-12)
    block = _err_block(d)
    return {
        "n_frames": int(len(pred)),
        "horizon": int(h),
        "rel_l2_mean": float(rel.mean()),
        "rel_l2_median": float(np.median(rel)),
        "rel_l2_p95": float(np.percentile(rel, 95)),
        "mse": block["mse"],
        "l1": block["l1"],
        "max_abs": float(np.abs(d).max()),
        "per_dim_mse": block["per_dim_mse"],
    }


def degradation(metrics: dict, baseline: dict) -> dict:
    """Change in ground-truth error relative to the FP32 baseline (positive = worse)."""
    out = {}
    for part in ("next_action", "chunk"):
        for m in ("mse", "l1"):
            v, b = metrics[part][m], baseline[part][m]
            out[f"{part}_{m}_delta"] = v - b
            out[f"{part}_{m}_delta_pct"] = (v - b) / b * 100.0 if b else float("nan")
    return out
