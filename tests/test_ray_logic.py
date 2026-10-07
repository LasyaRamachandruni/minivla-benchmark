"""The Ray actors' logic, exercised without Ray (the classes are plain Python)."""

import numpy as np
import pytest

from models.frames import synthetic_frames
from models.load_model import load_model
from ray_workers.actor import VLAWorker
from ray_workers.distributed_bench import make_batches
from ray_workers.pipeline_actors import PolicyStage, PostprocessStage, PreprocessStage
from ray_workers.serve_endpoint import decode_image, encode_image, parse_request


def test_worker_returns_real_model_actions():
    frames = synthetic_frames(4)
    (fd, seeds), = make_batches(frames, batch_size=4)
    out = VLAWorker("mock").infer_batch(fd, seeds)
    expected = load_model("mock", device="cpu").predict(frames, seeds, batch_size=4)
    np.testing.assert_allclose(out["actions"], expected, atol=1e-6)
    with pytest.raises(ValueError):
        VLAWorker("mock").infer_batch([], [])


def test_pipeline_stages_compose_to_the_same_actions():
    frames = synthetic_frames(3)
    (fd, seeds), = make_batches(frames, batch_size=3)
    out = PostprocessStage("mock").run(PolicyStage("mock").run(PreprocessStage("mock").run(fd, seeds)))
    expected = load_model("mock", device="cpu").predict(frames, seeds, batch_size=3)
    np.testing.assert_allclose(out["actions"], expected, atol=1e-6)
    assert set(out["timings"]) == {"preprocess", "policy", "postprocess"}


def test_serve_request_validation_and_image_roundtrip():
    f = synthetic_frames(1)[0]
    keys = list(f.images)
    body = {"instruction": f.task, "state": f.state.tolist(), "images": {k: encode_image(v) for k, v in f.images.items()}}
    parsed = parse_request(body, keys)
    np.testing.assert_array_equal(parsed.images[keys[0]], f.images[keys[0]])  # PNG is lossless
    assert decode_image(encode_image(f.images[keys[1]])).shape == f.images[keys[1]].shape
    for broken, msg in [({**body, "images": {}}, "missing images"),
                        ({k: v for k, v in body.items() if k != "state"}, "state"),
                        ({k: v for k, v in body.items() if k != "instruction"}, "instruction")]:
        with pytest.raises(ValueError, match=msg):
            parse_request(broken, keys)


def test_make_batches_covers_all_frames_once():
    frames = synthetic_frames(10)
    batches = make_batches(frames, 4)
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    assert sum((b[1] for b in batches), []) == list(range(10))
