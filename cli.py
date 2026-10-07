"""MiniVLA Benchmark - command line entry point.

    python cli.py sample-libero --num-frames 500 --cache data_cache/libero_frames.npz
    python cli.py optimize --model smolvla_libero --device cuda
    python cli.py compare --results "results/*.json" --markdown results/RESULTS.md
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click

DEFAULT_CACHE = "data_cache/libero_frames.npz"


@click.group()
@click.version_option(version="0.3.0", prog_name="minivla")
def cli():
    """Benchmark SmolVLA inference optimizations on real LIBERO frames."""


def _frames_for(model_name, frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic):
    if synthetic:
        from models.frames import synthetic_frames

        if model_name != "mock":
            raise click.UsageError("--synthetic inputs are only allowed with --model mock")
        frames = synthetic_frames(num_frames, seed=seed)
        return frames, {"dataset": "synthetic (random images, no ground truth)", "num_frames": len(frames),
                        "seed": seed, "num_tasks": len({f.task for f in frames})}
    from pipeline.data import DEFAULT_DATASET, get_frames

    return get_frames(frames_cache, DEFAULT_DATASET, num_frames, episodes_per_task, seed, horizon, max_tasks)


def _data_options(f):
    f = click.option("--synthetic", is_flag=True, help="Random frames without ground truth (mock model only).")(f)
    f = click.option("--seed", default=0, show_default=True, type=int, help="Frame sampling seed.")(f)
    f = click.option("--horizon", default=10, show_default=True, type=int,
                     help="Ground-truth action steps scored per frame.")(f)
    f = click.option("--max-tasks", default=None, type=int, help="Only sample this many of the 40 tasks.")(f)
    f = click.option("--episodes-per-task", default=1, show_default=True, type=int)(f)
    f = click.option("--num-frames", default=500, show_default=True, type=int, help="Evaluation frames.")(f)
    f = click.option("--frames-cache", default=DEFAULT_CACHE, show_default=True,
                     help="npz of sampled LIBERO frames; created on first use.")(f)
    return f


def _suite_options(f):
    f = click.option("--allow-small-n", is_flag=True, hidden=True, help="Permit N < 50 (smoke tests only).")(f)
    f = click.option("--out", default=None, help="Results JSON (default results/<model>_<device>.json).")(f)
    f = click.option("--eval-batch-size", default=1, show_default=True, type=int)(f)
    f = click.option("--warmup", default=10, show_default=True, type=int)(f)
    f = click.option("--N", "num_runs", default=100, show_default=True, type=int,
                     help="Timed runs per variant (>= 50).")(f)
    f = click.option("--device", default=None, help="cpu or cuda (default: cuda if available).")(f)
    f = click.option("--model", default="smolvla_libero", show_default=True,
                     help="smolvla_libero, smolvla_base, mock, or a LeRobot policy repo id / path.")(f)
    return f


@cli.command("sample-libero")
@click.option("--dataset", default="HuggingFaceVLA/libero", show_default=True)
@click.option("--num-frames", default=500, show_default=True, type=int)
@click.option("--episodes-per-task", default=1, show_default=True, type=int)
@click.option("--max-tasks", default=None, type=int)
@click.option("--horizon", default=10, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--cache", default=DEFAULT_CACHE, show_default=True)
def sample_libero(dataset, num_frames, episodes_per_task, max_tasks, horizon, seed, cache):
    """Download a fixed, seeded sample of LIBERO frames and cache it."""
    from pipeline.data import load_libero_frames, save_frames

    frames, info = load_libero_frames(dataset, num_frames, episodes_per_task, seed, horizon, max_tasks=max_tasks)
    save_frames(cache, frames, info)
    click.echo(f"Saved {len(frames)} frames from {info['num_tasks']} tasks / {len(info['episodes'])} episodes to {cache}")


def _run(model, device, variants, prune_amount, num_runs, warmup, eval_batch_size, out, frames_args, allow_small_n):
    from models.load_model import load_model
    from pipeline.optimize import parse_variants
    from pipeline.suite import default_output_path, run_suite

    frames, info = _frames_for(model, *frames_args)
    click.echo(f"Loaded {len(frames)} eval frames ({info.get('dataset')})")
    base = load_model(model, device=device)
    click.echo(f"Model {base.name} ({base.spec.repo_id}) on {base.device}: {base.size_mb():.1f} MB fp32")
    out = out or default_output_path(base.name, base.device)
    run_suite(base, frames, info, parse_variants(variants, base.device), num_runs=num_runs, warmup=warmup,
              eval_batch_size=eval_batch_size, prune_amount=prune_amount, out_path=out,
              allow_small_n=allow_small_n, log=click.echo)


@cli.command()
@_suite_options
@_data_options
def benchmark(model, device, num_runs, warmup, eval_batch_size, out, allow_small_n,
              frames_cache, num_frames, episodes_per_task, max_tasks, horizon, seed, synthetic):
    """FP32 baseline only: action error on LIBERO frames + latency."""
    _run(model, device, "fp32", 0.0, num_runs, warmup, eval_batch_size, out,
         (frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic), allow_small_n)


@cli.command()
@_suite_options
@_data_options
@click.option("--variants", default=None,
              help="Comma list, e.g. fp32,fp16,bf16,pruned,pruned+fp16,bnb_int8 (GPU) or "
                   "fp32,int8_dynamic,pruned,pruned+int8_dynamic (CPU). Default depends on device.")
@click.option("--prune-amount", default=0.2, show_default=True, type=float,
              help="Fraction of MLP hidden channels removed by structured pruning.")
def optimize(model, device, num_runs, warmup, eval_batch_size, out, allow_small_n,
             frames_cache, num_frames, episodes_per_task, max_tasks, horizon, seed, synthetic,
             variants, prune_amount):
    """FP32 baseline + optimized variants, all on the same device."""
    _run(model, device, variants, prune_amount, num_runs, warmup, eval_batch_size, out,
         (frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic), allow_small_n)


@cli.command()
@click.option("--results", "pattern", default="results/*.json", show_default=True, help="Glob of results JSON files.")
@click.option("--markdown", default="results/RESULTS.md", show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True, help="Write <results>.png next to each JSON.")
def compare(pattern, markdown, plot):
    """Build a markdown table (and plots) from results JSON files."""
    from pipeline.report import load_results, plot_result, write_markdown

    paths = [p for p in sorted(glob.glob(pattern)) if not os.path.basename(p).startswith("ray_")]
    if not paths:
        click.echo(f"No results match {pattern}. Run `python cli.py optimize` first.")
        sys.exit(1)
    results = load_results(paths)
    click.echo(write_markdown(results, markdown))
    if plot:
        for r in results:
            png = plot_result(r, os.path.splitext(r["_path"])[0] + ".png")
            if png:
                click.echo(f"Plot: {png}")


@cli.command("ray-bench")
@click.option("--model", default="mock", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--variant", default="fp32", show_default=True, help="Variant each worker runs (e.g. int8_dynamic).")
@click.option("--workers", default=2, show_default=True, type=int)
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--scaling-test", is_flag=True, help="Run with 1, 2, ... up to --workers workers.")
@click.option("--out", default=None, help="Optional results JSON (name it ray_*.json).")
@_data_options
def ray_bench(model, device, variant, workers, batch_size, scaling_test, out,
              frames_cache, num_frames, episodes_per_task, max_tasks, horizon, seed, synthetic):
    """Data-parallel batched inference over the eval frames with Ray actors."""
    from ray_workers.distributed_bench import compare_scaling

    frames, info = _frames_for(model, frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic)
    counts = sorted({c for c in (1, 2, 4, workers) if c <= workers}) if scaling_test else [workers]
    res = compare_scaling(model, frames, counts, batch_size, variant=variant, device=device)
    for r in res:
        click.echo(f"{r['num_workers']} worker(s): {r['throughput_frames_per_s']:.2f} frames/s, "
                   f"batch p50 {r['batch_latency']['p50_ms']:.1f} ms over {r['num_frames']} frames")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"dataset": info, "runs": res}, f, indent=2)
    if len(res) > 1:
        from pipeline.visualize import plot_scaling_curve

        plot_scaling_curve(res, [r["num_workers"] for r in res], output_path="results/ray_scaling.png")


@cli.command("pipeline-bench")
@click.option("--model", default="mock", show_default=True)
@click.option("--device", default="cpu", show_default=True, help="Device of the policy stage.")
@click.option("--batch-size", default=1, show_default=True, type=int)
@_data_options
def pipeline_bench(model, device, batch_size, frames_cache, num_frames, episodes_per_task, max_tasks, horizon,
                   seed, synthetic):
    """Pipelined preprocess -> policy -> postprocess Ray actors over the eval frames."""
    from pipeline.visualize import plot_pipeline_breakdown
    from ray_workers.pipeline_actors import run_pipeline_benchmark

    frames, _ = _frames_for(model, frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic)
    r = run_pipeline_benchmark(model, frames, batch_size=batch_size, device=device)
    click.echo(f"{r['num_frames']} frames: {r['throughput_frames_per_s']:.2f} frames/s, "
               f"e2e p50 {r['e2e_p50_ms']:.1f} ms")
    for stage, s in r["stage_stats"].items():
        click.echo(f"  {stage:12s} p50 {s['p50_ms']:.2f} ms  ({s['pct_of_stage_total']:.1f}% of stage time)")
    plot_pipeline_breakdown(r)


@cli.command("serve")
@click.option("--model", default="mock", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--replicas", default=1, type=int, show_default=True)
@click.option("--max-batch-size", default=8, type=int, show_default=True)
def serve(model, device, port, replicas, max_batch_size):
    """Ray Serve HTTP endpoint with request batching (requires real images + state + instruction)."""
    from ray_workers.serve_endpoint import run_serve

    run_serve(model_name=model, device=device, port=port, num_replicas=replicas, max_batch_size=max_batch_size)


@cli.command("batch")
@click.option("--model", default="mock", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--batch-size", default=8, type=int, show_default=True)
@click.option("--concurrency", default=1, type=int, show_default=True)
@click.option("--out", default=None, help="Optional npz with predicted action chunks.")
@_data_options
def batch(model, device, batch_size, concurrency, out, frames_cache, num_frames, episodes_per_task, max_tasks,
          horizon, seed, synthetic):
    """Offline batch inference over the eval frames with Ray Data."""
    from ray_workers.batch_inference import run_batch_inference

    frames, _ = _frames_for(model, frames_cache, num_frames, episodes_per_task, seed, horizon, max_tasks, synthetic)
    run_batch_inference(model, frames, batch_size=batch_size, concurrency=concurrency, device=device, out_path=out)


if __name__ == "__main__":
    cli()
