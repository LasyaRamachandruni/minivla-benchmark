import json
import os

from click.testing import CliRunner

from models.load_model import load_model
from models.vla import ModelSpec
from pipeline.report import load_results, markdown_for, plot_result, write_markdown
from pipeline.suite import run_suite


def _mock_with_gt():
    m = load_model("mock", device="cpu")
    # pretend the mock shares the dataset's action space so the ground-truth path runs
    m.spec = ModelSpec(key="mock", repo_id="mock", gt_comparable=True)
    return m


def test_suite_writes_complete_results(tmp_path, frames_with_gt):
    out = str(tmp_path / "r.json")
    res = run_suite(_mock_with_gt(), frames_with_gt, {"dataset": "unit-test", "num_frames": 12},
                    ["int8_dynamic", "pruned", "fp16"], num_runs=5, warmup=1, eval_batch_size=4,
                    out_path=out, allow_small_n=True, log=lambda s: None)
    with open(out) as f:
        on_disk = json.load(f)
    assert on_disk["protocol"]["variants"] == ["fp32", "int8_dynamic", "pruned", "fp16"]
    v = {e["variant"]: e for e in on_disk["variants"]}
    assert v["fp32"]["agreement_with_fp32"]["rel_l2_mean"] == 0.0
    assert v["fp32"]["noise_floor"]["rel_l2_mean"] > 0
    assert "degradation_vs_fp32" not in v["fp32"]
    for name in ("int8_dynamic", "pruned"):
        assert v[name]["device"] == "cpu"  # same device as the baseline
        assert v[name]["latency"]["inference"]["n"] == 5
        assert "chunk_mse_delta_pct" in v[name]["degradation_vs_fp32"]
        assert v[name]["vs_ground_truth"]["horizon"] == 10
    assert "skipped" in v["fp16"]
    assert on_disk["environment"]["torch"]
    assert res["protocol"]["ground_truth_metrics"] is True


def test_suite_disables_ground_truth_for_unrelated_checkpoint(tmp_path, frames_with_gt):
    res = run_suite(load_model("mock", device="cpu"), frames_with_gt, {"dataset": "unit-test"}, ["pruned"],
                    num_runs=3, warmup=1, allow_small_n=True, log=lambda s: None)
    assert res["protocol"]["ground_truth_metrics"] is False
    assert all("vs_ground_truth" not in v for v in res["variants"])


def test_report_markdown_and_plot(tmp_path, frames_with_gt):
    out = str(tmp_path / "mock_cpu.json")
    run_suite(_mock_with_gt(), frames_with_gt, {"dataset": "unit-test", "num_frames": 12, "seed": 0},
              ["int8_dynamic", "fp16"], num_runs=3, warmup=1, out_path=out, allow_small_n=True, log=lambda s: None)
    results = load_results([out])
    md = markdown_for(results[0])
    assert "| fp32 |" in md and "| int8_dynamic |" in md and "skipped" in md and "Noise floor" in md
    text = write_markdown(results, str(tmp_path / "RESULTS.md"))
    assert text.startswith("# Results")
    assert os.path.exists(plot_result(results[0], str(tmp_path / "mock_cpu.png")))


def test_cli_optimize_and_compare(tmp_path):
    from cli import cli

    runner = CliRunner()
    out = str(tmp_path / "mock_cpu.json")
    r = runner.invoke(cli, ["optimize", "--model", "mock", "--device", "cpu", "--synthetic", "--num-frames", "4",
                            "--N", "3", "--warmup", "1", "--allow-small-n", "--out", out])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli, ["compare", "--results", str(tmp_path / "*.json"), "--markdown", str(tmp_path / "R.md")])
    assert r.exit_code == 0, r.output
    assert "Ground-truth columns omitted" in r.output


def test_cli_refuses_small_n_and_synthetic_real_models(tmp_path):
    from cli import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--model", "mock", "--device", "cpu", "--synthetic", "--num-frames", "2",
                            "--N", "5", "--out", str(tmp_path / "x.json")])
    assert r.exit_code != 0 and "at least 50" in str(r.exception)
    r = runner.invoke(cli, ["benchmark", "--model", "smolvla_libero", "--synthetic"])
    assert r.exit_code != 0 and "only allowed with --model mock" in r.output
