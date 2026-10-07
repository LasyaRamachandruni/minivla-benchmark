"""Offline batch inference over the evaluation frames with Ray Data.

Frames become rows of a Ray Dataset (one tensor column per camera, plus state / task / index);
a stateful mapper holds the model and predicts a whole batch per call. Outputs are the model's
action chunks, optionally saved to an .npz together with ground truth for later analysis.
"""

import time
from typing import Optional, Sequence

import numpy as np

from models.frames import Frame, stack_ground_truth


class BatchPredictor:
    """Ray Data callable class: numpy batch -> action chunks."""

    def __init__(self, model_name: str, device: str, camera_keys: Sequence[str], noise_seed: int = 0):
        from models.load_model import load_model

        self.model = load_model(model_name, device=device)
        self.camera_keys = list(camera_keys)
        self.noise_seed = noise_seed

    def __call__(self, batch: dict) -> dict:
        n = len(batch["idx"])
        frames = [
            Frame(images={k: np.asarray(batch[f"cam{j}"][i], dtype=np.uint8) for j, k in enumerate(self.camera_keys)},
                  state=batch["state"][i], task=str(batch["task"][i]))
            for i in range(n)
        ]
        idx = [int(i) for i in batch["idx"]]
        seeds = [self.noise_seed * 1_000_003 + i for i in idx]  # same scheme as pipeline.suite.frame_seeds
        actions = self.model.predict(frames, seeds, batch_size=n)
        return {"idx": np.asarray(idx), "actions": actions}


def run_batch_inference(model_name: str, frames: Sequence[Frame], batch_size: int = 8, concurrency: int = 1,
                        device: str = "cpu", out_path: Optional[str] = None) -> dict:
    import ray.data

    from ray_workers import init_ray

    init_ray(log_to_driver=False)
    camera_keys = list(frames[0].images)
    rows = [{"idx": i, "state": f.state, "task": f.task, **{f"cam{j}": f.images[k] for j, k in enumerate(camera_keys)}}
            for i, f in enumerate(frames)]
    ds = ray.data.from_items(rows)

    t0 = time.perf_counter()
    out = ds.map_batches(
        BatchPredictor,
        fn_constructor_kwargs={"model_name": model_name, "device": device, "camera_keys": camera_keys},
        batch_size=batch_size,
        batch_format="numpy",
        concurrency=concurrency,
        num_gpus=1 if device.startswith("cuda") else 0,
    ).take_all()
    wall = time.perf_counter() - t0

    out.sort(key=lambda r: int(r["idx"]))
    actions = np.stack([r["actions"] for r in out])
    if len(actions) != len(frames):
        raise RuntimeError(f"expected {len(frames)} outputs, got {len(actions)}")
    summary = {"model": model_name, "num_frames": len(frames), "batch_size": batch_size,
               "concurrency": concurrency, "wall_time_s": wall, "throughput_frames_per_s": len(frames) / wall,
               "actions_shape": list(actions.shape)}
    print(f"Ray Data: {len(frames)} frames in {wall:.1f}s ({summary['throughput_frames_per_s']:.2f} frames/s)")
    if out_path:
        gt, pad = stack_ground_truth(list(frames))
        extra = {} if gt is None else {"gt_actions": gt, "gt_is_pad": pad}
        np.savez_compressed(out_path, actions=actions, **extra)
        print(f"Saved predictions to {out_path}")
    return summary
