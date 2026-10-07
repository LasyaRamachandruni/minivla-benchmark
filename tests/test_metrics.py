import numpy as np
import pytest

from pipeline.benchmark import MIN_TIMED_RUNS, environment_info, latency_stats, time_runs
from pipeline.evaluate import action_errors, agreement, degradation


def test_action_errors_exact_values_and_padding():
    pred = np.zeros((2, 4, 2))
    gt = np.ones((2, 3, 2))
    gt[:, :, 1] = 2.0
    pad = np.array([[False, False, True], [False, False, False]])
    e = action_errors(pred, gt, pad)
    assert e["horizon"] == 3
    assert e["next_action"]["mse"] == pytest.approx((1 + 4) / 2)
    assert e["next_action"]["per_dim_l1"] == pytest.approx([1.0, 2.0])
    assert e["chunk"]["count"] == 5  # one padded step dropped
    assert e["chunk"]["per_dim_mse"] == pytest.approx([1.0, 4.0])


def test_action_errors_refuses_mismatched_action_spaces():
    with pytest.raises(ValueError, match="action dims"):
        action_errors(np.zeros((1, 2, 6)), np.zeros((1, 2, 7)))


def test_agreement_is_zero_for_identical_and_scale_free():
    rng = np.random.RandomState(0)
    ref = rng.randn(5, 4, 3)
    assert agreement(ref, ref)["rel_l2_mean"] == 0.0
    assert agreement(ref * 1.1, ref)["rel_l2_mean"] == pytest.approx(0.1)
    assert agreement(ref * 1.1 * 1000, ref * 1000)["rel_l2_mean"] == pytest.approx(0.1)


def test_degradation_sign():
    base = {"next_action": {"mse": 1.0, "l1": 1.0}, "chunk": {"mse": 2.0, "l1": 1.0}}
    worse = {"next_action": {"mse": 1.5, "l1": 1.0}, "chunk": {"mse": 1.0, "l1": 1.0}}
    d = degradation(worse, base)
    assert d["next_action_mse_delta_pct"] == pytest.approx(50.0)
    assert d["chunk_mse_delta"] == pytest.approx(-1.0)


def test_time_runs_requires_enough_runs_and_does_warmup():
    calls = []
    with pytest.raises(ValueError):
        time_runs(lambda: calls.append(1), num_runs=5, warmup=1)
    lat = time_runs(lambda: calls.append(1), num_runs=MIN_TIMED_RUNS, warmup=3)
    assert len(lat) == MIN_TIMED_RUNS and len(calls) == MIN_TIMED_RUNS + 3
    assert (lat >= 0).all()


def test_latency_stats_ci_brackets_estimate():
    rng = np.random.RandomState(0)
    lat = rng.lognormal(3, 0.2, size=200)
    s = latency_stats(lat)
    assert s["n"] == 200
    assert s["p50_ci95_ms"][0] <= s["p50_ms"] <= s["p50_ci95_ms"][1]
    assert s["mean_ci95_ms"][0] < s["mean_ms"] < s["mean_ci95_ms"][1]
    assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"]
    assert s == latency_stats(lat)  # bootstrap is seeded


def test_environment_info_records_versions():
    env = environment_info("cpu")
    for k in ("torch", "python", "cpu", "torch_num_threads", "device", "timestamp_utc"):
        assert env[k] is not None
