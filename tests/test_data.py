import numpy as np
import pytest

from models.frames import Frame, synthetic_frames
from pipeline.data import load_frames, save_frames, select_episodes, stratified_frame_sample, subsample


def test_select_episodes_covers_every_task_and_is_seeded():
    episode_tasks = {i: f"task {i % 5}" for i in range(50)}
    a = select_episodes(episode_tasks, episodes_per_task=2, seed=7)
    assert a == select_episodes(episode_tasks, episodes_per_task=2, seed=7)
    assert a != select_episodes(episode_tasks, episodes_per_task=2, seed=8)
    assert len(a) == 10
    assert sorted({episode_tasks[e] for e in a}) == [f"task {i}" for i in range(5)]


def test_select_episodes_max_tasks():
    episode_tasks = {i: f"task {i % 5}" for i in range(50)}
    eps = select_episodes(episode_tasks, 1, seed=0, max_tasks=3)
    assert len(eps) == 3 and len({episode_tasks[e] for e in eps}) == 3


def test_stratified_sample_spreads_over_episodes():
    rows = np.repeat(np.arange(10), [50, 5, 50, 50, 50, 50, 50, 50, 50, 50])
    idx = stratified_frame_sample(rows, 100, seed=0)
    assert len(idx) == len(set(idx)) == 100
    counts = np.bincount(rows[idx], minlength=10)
    assert counts[1] == 5  # short episode fully used
    assert counts.max() - counts[counts != 5].min() <= 1  # the rest are balanced
    assert idx == stratified_frame_sample(rows, 100, seed=0)


def test_stratified_sample_caps_at_dataset_size():
    assert len(stratified_frame_sample(np.zeros(7, dtype=int), 100, seed=0)) == 7


def test_frame_cache_roundtrip(tmp_path, frames_with_gt):
    path = str(tmp_path / "frames.npz")
    save_frames(path, frames_with_gt, {"dataset": "unit-test", "seed": 3})
    frames, info = load_frames(path)
    assert info["dataset"] == "unit-test"
    assert len(frames) == len(frames_with_gt)
    for a, b in zip(frames, frames_with_gt):
        assert a.task == b.task and a.episode_index == b.episode_index
        np.testing.assert_array_equal(a.images["observation.images.image"], b.images["observation.images.image"])
        np.testing.assert_array_equal(a.gt_actions, b.gt_actions)
        np.testing.assert_array_equal(a.gt_is_pad, b.gt_is_pad)


def test_subsample_is_stratified_and_ordered(frames_with_gt):
    sub = subsample(frames_with_gt, 6, seed=0)
    assert len(sub) == 6
    assert sorted({f.episode_index for f in sub}) == [0, 1, 2]
    positions = [frames_with_gt.index(f) for f in sub]
    assert positions == sorted(positions)


def test_frame_rejects_non_uint8_images():
    with pytest.raises(ValueError):
        Frame(images={"cam": np.zeros((4, 4, 3), dtype=np.float32)}, state=np.zeros(8), task="x")


def test_synthetic_frames_have_no_ground_truth():
    assert not any(f.has_ground_truth for f in synthetic_frames(3))
