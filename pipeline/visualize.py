"""Visualization utilities for benchmark results.

Generates scaling curves, pipeline stage breakdowns, and comparison charts.
"""

import os
from typing import List

import numpy as np


def plot_scaling_curve(scaling_results: List[dict], worker_counts: List[int],
                       output_path: str = "results/scaling_curve.png"):
    """Generate a throughput scaling curve with ideal linear scaling reference.

    Args:
        scaling_results: List of benchmark result dicts from compare_scaling().
        worker_counts: Corresponding worker counts.
        output_path: Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qps_values = [r["throughput_qps"] for r in scaling_results]
    baseline_qps = qps_values[0]
    speedups = [q / baseline_qps for q in qps_values]
    ideal = [n / worker_counts[0] for n in worker_counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ray Scaling Analysis", fontsize=14, fontweight="bold")

    # Throughput curve
    ax1.plot(worker_counts, qps_values, "o-", color="#2196F3", linewidth=2,
             markersize=8, label="Measured QPS")
    ax1.plot(worker_counts, [baseline_qps * i / worker_counts[0] for i in worker_counts],
             "--", color="#9E9E9E", linewidth=1.5, label="Ideal linear scaling")
    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Throughput (QPS)")
    ax1.set_title("Throughput vs Workers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)

    # Speedup curve
    ax2.plot(worker_counts, speedups, "o-", color="#4CAF50", linewidth=2,
             markersize=8, label="Measured speedup")
    ax2.plot(worker_counts, ideal, "--", color="#9E9E9E", linewidth=1.5,
             label="Ideal linear")
    ax2.set_xlabel("Number of Workers")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Scaling Efficiency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)

    # Add efficiency annotations
    for i, (n, s) in enumerate(zip(worker_counts, speedups)):
        efficiency = s / (n / worker_counts[0]) * 100
        ax2.annotate(f"{efficiency:.0f}%", (n, s),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scaling curve saved to {output_path}")


def plot_pipeline_breakdown(pipeline_result: dict,
                            output_path: str = "results/pipeline_breakdown.png"):
    """Generate a stacked bar chart showing per-stage latency breakdown.

    Args:
        pipeline_result: Result dict from run_pipeline_benchmark().
        output_path: Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = pipeline_result["stage_stats"]
    stage_names = list(stages.keys())
    avg_latencies = [stages[s]["avg_ms"] for s in stage_names]
    pct_of_e2e = [stages[s]["pct_of_e2e"] for s in stage_names]

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    display_names = ["Vision Encoder", "Language Decoder", "Action Decoder"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("VLA Pipeline Stage Breakdown", fontsize=14, fontweight="bold")

    # Bar chart of average latencies
    bars = ax1.bar(display_names, avg_latencies, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Average Latency (ms)")
    ax1.set_title("Per-Stage Latency")
    for bar, val in zip(bars, avg_latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}ms", ha="center", fontsize=10)

    # Pie chart of time distribution
    ax2.pie(pct_of_e2e, labels=display_names, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Time Distribution")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pipeline breakdown saved to {output_path}")
