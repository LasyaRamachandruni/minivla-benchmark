"""Plots for the Ray benchmarks (scaling curve, pipeline stage breakdown)."""

import os
from typing import List


def plot_scaling_curve(scaling_results: List[dict], worker_counts: List[int],
                       output_path: str = "results/ray_scaling.png"):
    """Measured throughput vs workers, with ideal linear scaling and the machine's core count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qps = [r["throughput_frames_per_s"] for r in scaling_results]
    base = qps[0]
    speedup = [q / base for q in qps]
    ideal = [n / worker_counts[0] for n in worker_counts]
    cpus = scaling_results[0].get("cpu_count")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Ray data-parallel inference ({scaling_results[0]['model']}, {cpus} CPU cores)", fontsize=12)
    ax1.plot(worker_counts, qps, "o-", label="measured")
    ax1.plot(worker_counts, [base * i for i in ideal], "--", color="gray", label="ideal linear")
    ax1.set_xlabel("workers")
    ax1.set_ylabel("frames / s")
    ax1.set_xticks(worker_counts)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(worker_counts, speedup, "o-", color="#4CAF50", label="measured")
    ax2.plot(worker_counts, ideal, "--", color="gray", label="ideal linear")
    for n, s, i in zip(worker_counts, speedup, ideal):
        ax2.annotate(f"{s / i * 100:.0f}%", (n, s), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax2.set_xlabel("workers")
    ax2.set_ylabel("speedup vs 1 worker")
    ax2.set_xticks(worker_counts)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Scaling curve saved to {output_path}")


def plot_pipeline_breakdown(pipeline_result: dict, output_path: str = "results/ray_pipeline_breakdown.png"):
    """Average time per pipeline stage (preprocess / policy / postprocess)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = pipeline_result["stage_stats"]
    names = list(stages)
    avg = [stages[s]["avg_ms"] for s in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, avg, color=["#2196F3", "#FF9800", "#4CAF50"][:len(names)])
    for b, v in zip(bars, avg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("avg ms per batch")
    ax.set_title(f"Pipeline stages ({pipeline_result['model']}, batch {pipeline_result['batch_size']}, "
                 f"{pipeline_result['device']})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Pipeline breakdown saved to {output_path}")
