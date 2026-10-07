import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests never download anything


@pytest.fixture
def frames_with_gt():
    """Synthetic frames with made-up ground truth, to exercise the metric code paths only."""
    from models.frames import synthetic_frames

    frames = synthetic_frames(12, seed=3)
    rng = np.random.RandomState(0)
    for i, f in enumerate(frames):
        f.gt_actions = rng.randn(10, 7).astype(np.float32)
        f.gt_is_pad = np.zeros(10, dtype=bool)
        f.gt_is_pad[10 - (i % 3):] = True
        f.episode_index = i // 4
    return frames
