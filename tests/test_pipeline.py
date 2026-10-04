"""Offline tests for the benchmarking and optimization pipeline, using the mock VLA model."""

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest
import torch

from models.load_model import MockProcessor, load_model
from pipeline.benchmark import BenchmarkResult, run_benchmark
from pipeline.optimize import (
    apply_pytorch_dynamic_quantization,
    apply_structured_pruning,
    export_to_onnx,
    quantize_dynamic_int8,
)


@pytest.fixture(scope="module")
def mock():
    torch.manual_seed(0)
    return load_model("mock", device="cpu")


def sample_inputs(batch=2, seq=32):
    return torch.randn(batch, 3, 224, 224), torch.randint(0, 1000, (batch, seq))


def test_mock_model_output_shapes(mock):
    pixels, ids = sample_inputs()
    with torch.no_grad():
        out = mock.model(pixels, ids)
    assert out["actions"].shape == (2, 7)  # 7-DoF action
    assert out["logits"].shape == (2, 32, 1000)


def test_processor_tokens_are_stable_across_processes():
    code = "from models.load_model import MockProcessor; print(MockProcessor()(text='pick up the red block')['input_ids'].tolist())"
    outs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1


def test_processor_pads_to_max_length():
    ids = MockProcessor(max_length=8)(text="a b c")["input_ids"]
    assert ids.shape == (1, 8)
    assert (ids[0, 3:] == 0).all()


def test_benchmark_percentiles_are_ordered(mock):
    r = run_benchmark(mock, num_runs=15, warmup_runs=2)
    assert isinstance(r, BenchmarkResult)
    assert 0 < r.p50_latency_ms <= r.p95_latency_ms <= r.p99_latency_ms
    assert r.throughput_qps > 0
    assert len(r.summary_row()) == len(BenchmarkResult.table_headers())


def test_onnx_export_matches_pytorch_and_int8_shrinks(mock, tmp_path):
    path = export_to_onnx(mock, str(tmp_path / "mock.onnx"))
    pixels, ids = sample_inputs(batch=3, seq=32)  # batch is dynamic; length is fixed at 32
    with torch.no_grad():
        ref = mock.model.cpu()(pixels, ids)

    sess = ort.InferenceSession(path)
    actions, logits = sess.run(None, {"pixel_values": pixels.numpy(), "input_ids": ids.numpy()})
    np.testing.assert_allclose(actions, ref["actions"].numpy(), atol=1e-4)
    assert logits.shape == (3, 32, 1000)
    with pytest.raises(Exception, match="(?i)dimension|shape"):  # wrong length fails clearly
        sess.run(None, {"pixel_values": pixels.numpy(), "input_ids": ids[:, :16].numpy()})

    q_path = quantize_dynamic_int8(path)
    assert os.path.getsize(q_path) < 0.5 * os.path.getsize(path)
    q_actions, _ = ort.InferenceSession(q_path).run(None, {"pixel_values": pixels.numpy(), "input_ids": ids.numpy()})
    assert np.abs(q_actions - actions).max() < 0.1


def test_structured_pruning_zeroes_output_channels(mock):
    pruned = apply_structured_pruning(mock, amount=0.3)
    head = pruned.model.lm_head.weight.detach()
    zero_rows = (head.abs().sum(dim=1) == 0).float().mean().item()
    assert zero_rows == pytest.approx(0.3, abs=0.01)
    # the original model is untouched
    assert (mock.model.lm_head.weight.abs().sum(dim=1) == 0).sum() == 0


def test_pytorch_int8_quantization_keeps_outputs_close(mock):
    q = apply_pytorch_dynamic_quantization(mock)
    pixels, ids = sample_inputs()
    with torch.no_grad():
        ref = mock.model.cpu()(pixels, ids)["actions"]
        got = q.model(pixels, ids)["actions"]
    assert torch.allclose(ref, got, atol=0.05)
    assert q.size_mb < mock.size_mb
