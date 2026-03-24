"""MiniVLA Benchmark — Unified CLI entry point.

Usage:
    python cli.py benchmark --model mock --N 100
    python cli.py optimize --model mock --quant-type int8 --prune-amount 0.3
    python cli.py ray-bench --model mock --workers 4 --N 200
    python cli.py compare --results results/results_table.csv
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="minivla")
def cli():
    """MiniVLA Benchmark — VLA inference benchmarking and optimization toolkit."""
    pass


@cli.command()
@click.option("--model", default="mock", help="Model name: mock, mobilevlm, llava, or HF model ID.")
@click.option("--input", "input_image", default=None, help="Path to input image (optional).")
@click.option("--prompt", default="pick up the red block", help="Text prompt for the model.")
@click.option("--N", "num_runs", default=100, type=int, help="Number of benchmark runs.")
@click.option("--warmup", default=10, type=int, help="Number of warmup runs.")
@click.option("--output", default="results/results_table.csv", help="Output CSV path.")
@click.option("--device", default=None, help="Device: cpu, cuda, mps. Auto-detect if omitted.")
@click.option("--eval-samples", default=None, type=int, help="Number of eval samples (default: 50 for mock, 10 for real models).")
def benchmark(model, input_image, prompt, num_runs, warmup, output, device, eval_samples):
    """Stage 1: Benchmark single-device VLA inference."""
    from PIL import Image
    from models.load_model import load_model
    from pipeline.benchmark import run_benchmark, print_results_table, save_results_csv
    from pipeline.evaluate import generate_eval_dataset, evaluate_model

    image = None
    if input_image:
        image = Image.open(input_image)

    click.echo(f"Loading model: {model}")
    model_info = load_model(model, device=device)
    click.echo(f"Model loaded: {model_info.name} ({model_info.size_mb:.1f} MB) on {model_info.device}")

    # Evaluate accuracy (baseline = 100%)
    n_eval = eval_samples or (50 if model == "mock" else 10)
    click.echo(f"Evaluating accuracy ({n_eval} samples)...")
    dataset = generate_eval_dataset(model_info.processor, model_info.device, num_samples=n_eval)
    accuracy, _ = evaluate_model(model_info, dataset)

    # Run benchmark
    click.echo("Running benchmark...")
    result = run_benchmark(
        model_info,
        config_name="Baseline PyTorch",
        num_runs=num_runs,
        warmup_runs=warmup,
        image=image,
        prompt=prompt,
        accuracy_pct=accuracy,
    )

    print_results_table([result])
    save_results_csv([result], output)


@cli.command()
@click.option("--model", default="mock", help="Model name to optimize.")
@click.option("--quant-type", default="int8", type=click.Choice(["int8"]), help="Quantization type.")
@click.option("--prune-amount", default=0.3, type=float, help="Pruning fraction (0.0-1.0).")
@click.option("--output-dir", default="optimized", help="Directory for optimized models.")
@click.option("--N", "num_runs", default=50, type=int, help="Benchmark runs per config.")
@click.option("--results-output", default="results/results_table.csv", help="Output CSV path.")
@click.option("--device", default=None, help="Device override.")
@click.option("--eval-samples", default=None, type=int, help="Number of eval samples (default: 50 for mock, 10 for real models).")
def optimize(model, quant_type, prune_amount, output_dir, num_runs, results_output, device, eval_samples):
    """Stage 2: Optimize model (ONNX export, quantization, pruning) and benchmark."""
    from models.load_model import load_model
    from pipeline.optimize import optimize_full_pipeline
    from pipeline.benchmark import run_benchmark, print_results_table, save_results_csv
    from pipeline.evaluate import generate_eval_dataset, evaluate_model

    click.echo(f"Loading baseline model: {model}")
    model_info = load_model(model, device=device)

    # Generate eval dataset for accuracy comparison
    n_eval = eval_samples or (50 if model == "mock" else 10)
    click.echo(f"Using {n_eval} eval samples for accuracy comparison")
    dataset = generate_eval_dataset(model_info.processor, model_info.device, num_samples=n_eval)

    # Baseline evaluation
    click.echo("Evaluating baseline...")
    baseline_acc, baseline_outputs = evaluate_model(model_info, dataset)

    # Baseline benchmark
    click.echo("Benchmarking baseline...")
    baseline_result = run_benchmark(
        model_info, config_name="Baseline PyTorch",
        num_runs=num_runs, accuracy_pct=baseline_acc,
    )
    all_results = [baseline_result]

    # Run optimization pipeline
    click.echo("\nRunning optimization pipeline...")
    optimized = optimize_full_pipeline(
        model_info, output_dir=output_dir,
        quant_type=quant_type, prune_amount=prune_amount,
    )

    # Benchmark each optimized variant
    for config_name, opt_info in optimized.items():
        click.echo(f"\nBenchmarking: {config_name}")

        # Evaluate accuracy relative to baseline
        acc, _ = evaluate_model(opt_info, dataset, baseline_outputs=baseline_outputs)

        result = run_benchmark(
            opt_info, config_name=config_name,
            num_runs=num_runs, accuracy_pct=acc,
        )
        all_results.append(result)

    # Print and save comparison
    print_results_table(all_results)
    save_results_csv(all_results, results_output)


@cli.command("ray-bench")
@click.option("--model", default="mock", help="Model name for workers.")
@click.option("--model-path", default=None, help="Path to ONNX model for workers.")
@click.option("--workers", default=4, type=int, help="Number of Ray workers.")
@click.option("--N", "num_requests", default=200, type=int, help="Total inference requests.")
@click.option("--prompt", default="pick up the red block", help="Text prompt.")
@click.option("--scaling-test", is_flag=True, help="Test scaling with 1, 2, 4 workers.")
@click.option("--results-output", default="results/results_table.csv", help="Output CSV path.")
def ray_bench(model, model_path, workers, num_requests, prompt, scaling_test, results_output):
    """Stage 3: Distributed inference benchmark with Ray workers."""
    from ray_workers.distributed_bench import run_distributed_benchmark, compare_scaling
    from pipeline.benchmark import BenchmarkResult, print_results_table, save_results_csv

    if scaling_test:
        click.echo("Running scaling test: 1, 2, 4 workers")
        worker_counts = [1, 2, 4]
        if workers not in worker_counts:
            worker_counts.append(workers)
            worker_counts.sort()

        scaling_results = compare_scaling(
            model_name=model,
            model_path=model_path,
            worker_counts=worker_counts,
            num_requests=num_requests,
        )

        # Convert to BenchmarkResults for table output
        bench_results = []
        for res, n in zip(scaling_results, worker_counts):
            br = BenchmarkResult(
                config_name=f"Ray ({n} workers)",
                model_name=model,
                backend="ray",
                device="distributed",
                size_mb=0,
                num_runs=num_requests,
                p50_latency_ms=res["p50_latency_ms"],
                p95_latency_ms=res["p95_latency_ms"],
                p99_latency_ms=res["p99_latency_ms"],
                mean_latency_ms=res["mean_latency_ms"],
                throughput_qps=res["throughput_qps"],
                peak_memory_mb=0,
            )
            bench_results.append(br)

        print_results_table(bench_results)
        save_results_csv(bench_results, results_output)

        # Generate scaling curve
        from pipeline.visualize import plot_scaling_curve
        plot_scaling_curve(scaling_results, worker_counts,
                          output_path="results/scaling_curve.png")

    else:
        click.echo(f"Running distributed benchmark: {workers} workers, {num_requests} requests")
        result = run_distributed_benchmark(
            model_name=model,
            model_path=model_path,
            num_workers=workers,
            num_requests=num_requests,
            prompt=prompt,
        )

        br = BenchmarkResult(
            config_name=f"Ray ({workers} workers)",
            model_name=model,
            backend="ray",
            device="distributed",
            size_mb=0,
            num_runs=num_requests,
            p50_latency_ms=result["p50_latency_ms"],
            p95_latency_ms=result["p95_latency_ms"],
            p99_latency_ms=result["p99_latency_ms"],
            mean_latency_ms=result["mean_latency_ms"],
            throughput_qps=result["throughput_qps"],
            peak_memory_mb=0,
        )

        print_results_table([br])
        save_results_csv([br], results_output)

        # Print worker utilization
        click.echo("\nWorker Utilization:")
        for ws in result["worker_stats"]:
            click.echo(f"  Worker {ws['worker_id']}: "
                       f"{ws['request_count']} requests, "
                       f"avg {ws['avg_latency_ms']:.2f} ms/request")


@cli.command()
@click.option("--results", default="results/results_table.csv", help="Path to results CSV.")
@click.option("--plot", is_flag=True, help="Generate comparison plots.")
@click.option("--plot-output", default="results/comparison.png", help="Plot output path.")
def compare(results, plot, plot_output):
    """Stage 4: Display and compare benchmark results."""
    import csv
    from tabulate import tabulate

    if not os.path.exists(results):
        click.echo(f"Results file not found: {results}")
        click.echo("Run 'minivla benchmark' or 'minivla optimize' first.")
        sys.exit(1)

    with open(results) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    click.echo("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    if plot:
        _generate_plots(headers, rows, plot_output)
        click.echo(f"\nPlot saved to {plot_output}")


def _generate_plots(headers, rows, output_path):
    """Generate comparison bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [r[0] for r in rows]

    # Find column indices
    col_map = {h: i for i, h in enumerate(headers)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MiniVLA Benchmark Results", fontsize=14, fontweight="bold")

    # Latency comparison
    if "p50 (ms)" in col_map:
        p50_vals = [float(r[col_map["p50 (ms)"]]) for r in rows]
        p95_vals = [float(r[col_map["p95 (ms)"]]) for r in rows]
        x = range(len(configs))
        axes[0].bar([i - 0.2 for i in x], p50_vals, 0.4, label="p50", color="#2196F3")
        axes[0].bar([i + 0.2 for i in x], p95_vals, 0.4, label="p95", color="#FF9800")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title("Latency Comparison")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        axes[0].legend()

    # Throughput comparison
    if "Throughput (QPS)" in col_map:
        qps_vals = [float(r[col_map["Throughput (QPS)"]]) for r in rows]
        x = range(len(configs))
        axes[1].bar(list(x), qps_vals, color="#4CAF50")
        axes[1].set_ylabel("Queries/sec")
        axes[1].set_title("Throughput")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)

    # Size comparison
    if "Size (MB)" in col_map:
        size_vals = [float(r[col_map["Size (MB)"]]) for r in rows]
        x = range(len(configs))
        axes[2].bar(list(x), size_vals, color="#9C27B0")
        axes[2].set_ylabel("Size (MB)")
        axes[2].set_title("Model Size")
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@cli.command("pipeline-bench")
@click.option("--model", default="mock", help="Model name.")
@click.option("--N", "num_requests", default=100, type=int, help="Number of requests.")
@click.option("--prompt", default="pick up the red block", help="Text prompt.")
def pipeline_bench(model, num_requests, prompt):
    """Benchmark pipeline-parallel VLA inference (vision -> language -> action actors)."""
    from ray_workers.pipeline_actors import run_pipeline_benchmark
    from pipeline.visualize import plot_pipeline_breakdown

    result = run_pipeline_benchmark(
        model_name=model,
        num_requests=num_requests,
        prompt=prompt,
    )

    click.echo(f"\nPipeline Benchmark Results:")
    click.echo(f"  End-to-end p50: {result['e2e_p50_ms']:.2f} ms")
    click.echo(f"  End-to-end p95: {result['e2e_p95_ms']:.2f} ms")
    click.echo(f"  Throughput: {result['throughput_qps']:.2f} QPS")

    click.echo(f"\n  Stage Breakdown:")
    for stage, stats in result["stage_stats"].items():
        click.echo(f"    {stage:20s}  avg={stats['avg_ms']:.2f}ms  "
                    f"p50={stats['p50_ms']:.2f}ms  "
                    f"({stats['pct_of_e2e']:.1f}% of e2e)")

    plot_pipeline_breakdown(result)


@cli.command("serve")
@click.option("--model", default="mock", help="Model name.")
@click.option("--port", default=8000, type=int, help="HTTP port.")
@click.option("--replicas", default=1, type=int, help="Number of serve replicas.")
def serve(model, port, replicas):
    """Start a Ray Serve HTTP endpoint for VLA inference."""
    from ray_workers.serve_endpoint import run_serve
    run_serve(model_name=model, port=port, num_replicas=replicas)


@cli.command("batch")
@click.option("--model", default="mock", help="Model name.")
@click.option("--image-dir", default=None, help="Directory of images (or synthetic if omitted).")
@click.option("--N", "num_images", default=50, type=int, help="Number of synthetic images.")
@click.option("--prompt", default="pick up the red block", help="Text prompt.")
@click.option("--batch-size", default=1, type=int, help="Batch size for Ray Data.")
def batch(model, image_dir, num_images, prompt, batch_size):
    """Run batch inference over images using Ray Data."""
    from ray_workers.batch_inference import run_batch_inference
    run_batch_inference(
        model_name=model,
        image_dir=image_dir,
        num_synthetic=num_images,
        prompt=prompt,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
