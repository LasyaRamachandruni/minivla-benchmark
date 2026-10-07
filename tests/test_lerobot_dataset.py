"""Load frames through LeRobot's real dataset code from a tiny LIBERO-shaped dataset written
locally (no Hub access). Skipped when lerobot is not installed."""

import numpy as np
import pytest

pytest.importorskip("lerobot")

pytestmark = pytest.mark.integration

FEATURES = {
    "observation.images.image": {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
TASKS = ["pick up the black bowl", "open the top drawer", "turn on the stove"]


@pytest.fixture(scope="module")
def libero_like(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path_factory.mktemp("ds") / "libero_tiny"
    ds = LeRobotDataset.create("local/libero_tiny", fps=10, features=FEATURES, root=root, use_videos=False)
    rng = np.random.RandomState(0)
    for ep in range(6):  # two episodes per task, 12 frames each
        for t in range(12):
            ds.add_frame({
                "observation.images.image": rng.randint(0, 256, (32, 32, 3), dtype=np.uint8),
                "observation.images.image2": rng.randint(0, 256, (32, 32, 3), dtype=np.uint8),
                "observation.state": np.full(8, ep, dtype=np.float32),
                # action encodes (episode, step) so the ground-truth window can be checked exactly
                "action": np.array([ep, t, 0, 0, 0, 0, 1], dtype=np.float32),
                "task": TASKS[ep % 3],
            })
        ds.save_episode()
    ds.finalize()
    return str(root)


def test_load_frames_from_lerobot_dataset(libero_like):
    from pipeline.data import load_libero_frames

    frames, info = load_libero_frames("local/libero_tiny", num_frames=9, episodes_per_task=1, seed=0,
                                      horizon=4, root=libero_like)
    assert len(frames) == 9 and info["num_tasks"] == 3 and len(info["episodes"]) == 3
    for f in frames:
        assert f.images["observation.images.image"].shape == (32, 32, 3)
        assert f.images["observation.images.image"].dtype == np.uint8
        assert f.task == TASKS[f.episode_index % 3]
        assert f.gt_actions.shape == (4, 7)
        assert (f.gt_actions[:, 0] == f.episode_index).all()
        steps = f.frame_index + np.arange(4)
        valid = ~f.gt_is_pad
        np.testing.assert_array_equal(f.gt_actions[valid, 1], steps[valid])
        assert valid.sum() == min(4, 12 - f.frame_index)  # padded past the episode end
