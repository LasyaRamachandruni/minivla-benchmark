import numpy as np
import pytest
import torch

from models.frames import synthetic_frames
from models.load_model import load_model
from pipeline.optimize import (
    VariantUnavailable,
    build_variant,
    cast_backbone,
    int8_target_linears,
    parse_variants,
    prune_mlp_channels,
)


@pytest.fixture(scope="module")
def mock():
    return load_model("mock", device="cpu")


@pytest.fixture(scope="module")
def frames():
    return synthetic_frames(6, seed=1)


def test_mock_predicts_action_chunks(mock, frames):
    a = mock.predict(frames, seeds=range(6), batch_size=3)
    assert a.shape == (6, mock.chunk_size, 7)
    assert np.isfinite(a).all()


def test_noise_is_per_frame_so_batching_does_not_change_outputs(mock, frames):
    a = mock.predict(frames, seeds=range(6), batch_size=1)
    b = mock.predict(frames, seeds=range(6), batch_size=6)
    np.testing.assert_allclose(a, b, atol=1e-5)
    c = mock.predict(frames, seeds=range(100, 106), batch_size=6)
    assert np.abs(a - c).max() > 1e-3  # different noise -> different flow-matching sample


def test_quantizing_attention_projections_breaks_forward(mock, frames):
    """The reason int8 keeps q/k/v/o in FP32: the forward reads q_proj.weight.dtype."""
    policy = __import__("copy").deepcopy(mock.policy)
    all_linears = {n for n, m in policy.named_modules() if isinstance(m, torch.nn.Linear)}
    q = torch.ao.quantization.quantize_dynamic(policy, all_linears, dtype=torch.qint8)
    with pytest.raises(AttributeError):
        mock.with_policy(q, "cpu", "int8").predict(frames[:1])


def test_int8_dynamic_variant(mock, frames):
    q, info = build_variant(mock, "int8_dynamic")
    targets, skipped = int8_target_linears(mock.policy, mock.int8_skip_suffixes)
    assert info["quantization"]["int8_linear_layers"] == len(targets) > 0
    assert all(n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj") for n in skipped)
    assert q.size_mb() < mock.size_mb()
    assert q.num_parameters() == mock.num_parameters()  # packed int8 weights are still counted
    ref, out = mock.predict(frames, range(6)), q.predict(frames, range(6))
    assert np.abs(out - ref).max() < 0.1
    assert mock.policy.backbone.layers[0].mlp.up_proj.weight.dtype == torch.float32  # baseline untouched


def test_structured_pruning_really_shrinks_the_model(mock, frames):
    p, info = build_variant(mock, "pruned", prune_amount=0.25)
    pi = info["pruning"]
    assert pi["mlps_pruned"] == 3  # 2 gated MLPs + 1 vision MLP
    assert pi["params_after"] < pi["params_before"]
    assert p.size_mb() < mock.size_mb()
    for layer in pi["layers"]:
        assert layer["width_after"] % 8 == 0 and layer["width_after"] < layer["width_before"]
    mlp = p.policy.backbone.layers[0].mlp
    assert mlp.up_proj.out_features == mlp.gate_proj.out_features == mlp.down_proj.in_features == 192
    assert p.predict(frames, range(6)).shape == (6, mock.chunk_size, 7)


def test_pruning_zero_amount_is_exact(mock, frames):
    p = mock.with_policy(__import__("copy").deepcopy(mock.policy), "cpu", "fp32")
    prune_mlp_channels(p.policy, 0.0)
    np.testing.assert_allclose(p.predict(frames, range(6)), mock.predict(frames, range(6)), atol=1e-6)


def test_pruning_keeps_the_most_important_channels():
    from models.mock_vla import GatedMLP

    m = torch.nn.Module()
    m.mlp = GatedMLP(8, 16)
    with torch.no_grad():
        m.mlp.down_proj.weight[:, :8] *= 0.01  # channels 0-7 contribute almost nothing
    up_before = m.mlp.up_proj.weight.detach().clone()
    prune_mlp_channels(m, 0.5)
    assert m.mlp.up_proj.out_features == 8
    torch.testing.assert_close(m.mlp.up_proj.weight, up_before[8:])


def test_half_precision_casts_backbone_only(mock, frames):
    h = cast_backbone(mock, torch.bfloat16, "bf16")
    assert h.policy.backbone.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
    assert h.policy.action_out_proj.weight.dtype == torch.float32
    out = h.predict(frames, range(6))
    assert np.abs(out - mock.predict(frames, range(6))).max() < 0.2


def test_gpu_only_variants_are_skipped_on_cpu(mock):
    for v in ("fp16", "bf16", "bnb_int8", "pruned+fp16"):
        with pytest.raises(VariantUnavailable):
            build_variant(mock, v)


def test_parse_variants_always_includes_fp32_baseline():
    assert parse_variants("int8_dynamic", "cpu") == ["fp32", "int8_dynamic"]
    assert parse_variants(None, "cuda")[0] == "fp32"
    assert "int8_dynamic" in parse_variants(None, "cpu")
    with pytest.raises(ValueError):
        build_variant(load_model("mock", device="cpu"), "fp8")
