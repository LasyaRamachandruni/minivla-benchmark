"""Evaluation data: a fixed, seeded sample of LIBERO frames with ground-truth actions.

The default source is the LeRobot-format dataset `HuggingFaceVLA/libero` (1,693 episodes,
40 tasks, 10 fps, two 256x256 cameras, 8-D state, 7-D action). Only the episodes we sample
are downloaded. The sample is chosen in two seeded steps:

1. `episodes_per_task` episodes from every task (so all 40 tasks are covered), then
2. `num_frames` frames spread evenly over those episodes.

Each frame carries the next `horizon` ground-truth actions (padded at episode ends) so a
predicted action chunk can be scored against what the demonstrator actually did. Sampled
frames are cached to an .npz file so later runs (and the CPU section of the Colab notebook)
reuse exactly the same frames without touching the network.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.frames import Frame

DEFAULT_DATASET = "HuggingFaceVLA/libero"


def select_episodes(episode_tasks: Dict[int, str], episodes_per_task: int, seed: int,
                    max_tasks: Optional[int] = None) -> List[int]:
    """Seeded choice of `episodes_per_task` episodes for each task (or a seeded subset of tasks)."""
    rng = np.random.RandomState(seed)
    by_task: Dict[str, List[int]] = {}
    for ep, task in sorted(episode_tasks.items()):
        by_task.setdefault(task, []).append(ep)
    tasks = sorted(by_task)
    if max_tasks is not None and max_tasks < len(tasks):
        tasks = sorted(rng.choice(tasks, size=max_tasks, replace=False).tolist())
    chosen = []
    for task in tasks:
        eps = by_task[task]
        k = min(episodes_per_task, len(eps))
        chosen.extend(rng.choice(eps, size=k, replace=False).tolist())
    return sorted(int(e) for e in chosen)


def stratified_frame_sample(row_episode: np.ndarray, num_frames: int, seed: int) -> List[int]:
    """Row indices: `num_frames` rows spread as evenly as possible over the episodes present."""
    rng = np.random.RandomState(seed)
    row_episode = np.asarray(row_episode)
    episodes = np.unique(row_episode)
    rows_by_ep = {ep: np.flatnonzero(row_episode == ep) for ep in episodes}
    num_frames = min(num_frames, len(row_episode))

    quota = {ep: 0 for ep in episodes}
    remaining = num_frames
    # hand out frames round-robin in a seeded order, skipping episodes that are exhausted
    order = rng.permutation(episodes)
    while remaining > 0:
        progressed = False
        for ep in order:
            if remaining == 0:
                break
            if quota[ep] < len(rows_by_ep[ep]):
                quota[ep] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    picked = []
    for ep in episodes:
        if quota[ep]:
            picked.extend(rng.choice(rows_by_ep[ep], size=quota[ep], replace=False).tolist())
    return sorted(int(i) for i in picked)


def _episode_tasks(meta) -> Dict[int, str]:
    eps = meta.episodes
    out = {}
    for ep, tasks in zip(eps["episode_index"], eps["tasks"]):
        tasks = list(tasks) if not isinstance(tasks, str) else [tasks]
        out[int(ep)] = tasks[0]
    return out


def _to_uint8_hwc(img) -> np.ndarray:
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    return np.ascontiguousarray(arr)


def load_libero_frames(
    repo_id: str = DEFAULT_DATASET,
    num_frames: int = 500,
    episodes_per_task: int = 1,
    seed: int = 0,
    horizon: int = 10,
    revision: Optional[str] = None,
    max_tasks: Optional[int] = None,
    root: Optional[str] = None,
) -> Tuple[List[Frame], dict]:
    """Download the sampled episodes of a LeRobot dataset and return frames + a description of the sample.

    `root` points at a local copy of the dataset instead of the Hub.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    episodes = select_episodes(_episode_tasks(meta), episodes_per_task, seed, max_tasks)
    fps = meta.fps
    ds = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        revision=revision,
        delta_timestamps={"action": [i / fps for i in range(horizon)]},
    )
    row_episode = ds.hf_dataset.data.column("episode_index").to_numpy()
    rows = stratified_frame_sample(row_episode, num_frames, seed)
    camera_keys = list(meta.camera_keys)

    frames = []
    for r in rows:
        item = ds[r]
        actions = item["action"].numpy().reshape(horizon, -1)
        is_pad = item.get("action_is_pad")
        frames.append(Frame(
            images={k: _to_uint8_hwc(item[k]) for k in camera_keys},
            state=item["observation.state"].numpy(),
            task=item["task"],
            gt_actions=actions,
            gt_is_pad=np.zeros(horizon, dtype=bool) if is_pad is None else is_pad.numpy().astype(bool),
            episode_index=int(item["episode_index"]),
            frame_index=int(item["frame_index"]),
            source=repo_id,
        ))
    info = {
        "dataset": repo_id,
        "dataset_revision": getattr(ds, "revision", revision),
        "seed": seed,
        "num_frames": len(frames),
        "episodes_per_task": episodes_per_task,
        "max_tasks": max_tasks,
        "episodes": episodes,
        "num_tasks": len({f.task for f in frames}),
        "horizon": horizon,
        "fps": fps,
        "camera_keys": camera_keys,
    }
    return frames, info


def save_frames(path: str, frames: List[Frame], info: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    camera_keys = list(frames[0].images)
    arrays = {f"img::{k}": np.stack([f.images[k] for f in frames]) for k in camera_keys}
    arrays["state"] = np.stack([f.state for f in frames])
    has_gt = all(f.has_ground_truth for f in frames)
    if has_gt:
        arrays["gt_actions"] = np.stack([f.gt_actions for f in frames])
        arrays["gt_is_pad"] = np.stack([f.gt_is_pad for f in frames])
    arrays["episode_index"] = np.array([f.episode_index for f in frames])
    arrays["frame_index"] = np.array([f.frame_index for f in frames])
    arrays["task"] = np.array([f.task for f in frames])
    arrays["source"] = np.array([f.source for f in frames])
    arrays["info_json"] = np.array(json.dumps(info))
    np.savez_compressed(path, **arrays)


def load_frames(path: str) -> Tuple[List[Frame], dict]:
    with np.load(path, allow_pickle=False) as z:
        camera_keys = [k.split("::", 1)[1] for k in z.files if k.startswith("img::")]
        imgs = {k: z[f"img::{k}"] for k in camera_keys}
        n = len(z["state"])
        has_gt = "gt_actions" in z.files
        frames = [
            Frame(
                images={k: imgs[k][i] for k in camera_keys},
                state=z["state"][i],
                task=str(z["task"][i]),
                gt_actions=z["gt_actions"][i] if has_gt else None,
                gt_is_pad=z["gt_is_pad"][i] if has_gt else None,
                episode_index=int(z["episode_index"][i]),
                frame_index=int(z["frame_index"][i]),
                source=str(z["source"][i]),
            )
            for i in range(n)
        ]
        info = json.loads(str(z["info_json"]))
    return frames, info


def get_frames(cache_path: Optional[str], repo_id: str = DEFAULT_DATASET, num_frames: int = 500,
               episodes_per_task: int = 1, seed: int = 0, horizon: int = 10,
               max_tasks: Optional[int] = None) -> Tuple[List[Frame], dict]:
    """Load frames from the cache if it exists, otherwise sample them from the Hub and cache them.

    Asking for fewer frames than the cache holds returns a seeded subset of it, still spread
    over all cached episodes (and therefore tasks).
    """
    if cache_path and os.path.exists(cache_path):
        frames, info = load_frames(cache_path)
        if num_frames < len(frames):
            frames = subsample(frames, num_frames, seed)
            info = {**info, "num_frames": len(frames), "subsampled_from_cache": True}
        return frames, info
    frames, info = load_libero_frames(repo_id, num_frames, episodes_per_task, seed, horizon, max_tasks=max_tasks)
    if cache_path:
        save_frames(cache_path, frames, info)
    return frames, info


def subsample(frames: Sequence[Frame], n: int, seed: int) -> List[Frame]:
    """Seeded subset of n frames, stratified over episodes, in the original order."""
    rows = stratified_frame_sample(np.array([f.episode_index for f in frames]), n, seed)
    return [frames[i] for i in rows]
