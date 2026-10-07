"""A single observation (camera images + robot state + instruction) and its ground truth.

Frames are the unit passed between data loading, evaluation, benchmarking and the Ray
workers. Images are stored as uint8 HWC arrays so they are compact to cache and to ship
between processes; models convert them to whatever their policy expects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(eq=False)  # identity equality; field-wise == on arrays is ambiguous
class Frame:
    images: Dict[str, np.ndarray]  # dataset camera key -> uint8 (H, W, 3)
    state: np.ndarray  # float32 (state_dim,)
    task: str  # natural-language instruction
    gt_actions: Optional[np.ndarray] = None  # float32 (horizon, action_dim), ground truth from the dataset
    gt_is_pad: Optional[np.ndarray] = None  # bool (horizon,), True where the horizon ran past the episode end
    episode_index: int = -1
    frame_index: int = -1
    source: str = "unknown"  # dataset repo id, or "synthetic"

    def __post_init__(self):
        for key, img in self.images.items():
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"image {key!r} must be uint8 (H, W, 3), got {img.dtype} {img.shape}")
        self.state = np.asarray(self.state, dtype=np.float32)
        if self.gt_actions is not None:
            self.gt_actions = np.asarray(self.gt_actions, dtype=np.float32)
            if self.gt_actions.ndim == 1:
                self.gt_actions = self.gt_actions[None]
            if self.gt_is_pad is None:
                self.gt_is_pad = np.zeros(len(self.gt_actions), dtype=bool)

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_actions is not None

    def to_dict(self) -> dict:
        """Plain-python form for Ray / JSON transport (arrays become lists)."""
        return {
            "images": {k: v for k, v in self.images.items()},
            "state": self.state,
            "task": self.task,
            "gt_actions": self.gt_actions,
            "gt_is_pad": self.gt_is_pad,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            images={k: np.asarray(v, dtype=np.uint8) for k, v in d["images"].items()},
            state=np.asarray(d["state"], dtype=np.float32),
            task=d["task"],
            gt_actions=None if d.get("gt_actions") is None else np.asarray(d["gt_actions"], dtype=np.float32),
            gt_is_pad=None if d.get("gt_is_pad") is None else np.asarray(d["gt_is_pad"], dtype=bool),
            episode_index=int(d.get("episode_index", -1)),
            frame_index=int(d.get("frame_index", -1)),
            source=d.get("source", "unknown"),
        )


def synthetic_frames(
    n: int,
    camera_keys=("observation.images.image", "observation.images.image2"),
    state_dim: int = 8,
    image_size: int = 64,
    seed: int = 0,
    tasks: List[str] = ("pick up the black bowl", "open the top drawer"),
) -> List[Frame]:
    """Random frames for tests and for the mock model only.

    They carry no ground-truth actions, so nothing evaluated on them can be reported as accuracy.
    """
    rng = np.random.RandomState(seed)
    frames = []
    for i in range(n):
        frames.append(Frame(
            images={k: rng.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8) for k in camera_keys},
            state=rng.randn(state_dim).astype(np.float32),
            task=tasks[i % len(tasks)],
            source="synthetic",
            frame_index=i,
        ))
    return frames


def stack_ground_truth(frames: List[Frame]):
    """Return (gt_actions, gt_is_pad) stacked over frames, or (None, None) if any frame lacks GT."""
    if not frames or not all(f.has_ground_truth for f in frames):
        return None, None
    return np.stack([f.gt_actions for f in frames]), np.stack([f.gt_is_pad for f in frames])
