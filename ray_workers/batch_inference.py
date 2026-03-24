"""Batch inference with Ray Data.

Processes a folder of images (or synthetic dataset) in parallel using
Ray Data, demonstrating scalable batch processing for VLA inference.
"""

import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_batch_inference(
    model_name: str = "mock",
    image_dir: Optional[str] = None,
    num_synthetic: int = 50,
    prompt: str = "pick up the red block",
    batch_size: int = 1,
    num_cpus_per_task: int = 1,
) -> dict:
    """Run batch inference over a dataset using Ray Data.

    Args:
        model_name: Model to use for inference.
        image_dir: Directory of images. If None, generates synthetic images.
        num_synthetic: Number of synthetic images if image_dir is None.
        prompt: Text prompt for all images.
        batch_size: Batch size for map_batches.
        num_cpus_per_task: CPUs per inference task.

    Returns:
        Dict with throughput, latency stats, and sample outputs.
    """
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Build dataset
    if image_dir and os.path.isdir(image_dir):
        image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        print(f"Found {len(image_paths)} images in {image_dir}")
        ds = ray.data.from_items([
            {"image_path": p, "prompt": prompt, "index": i}
            for i, p in enumerate(image_paths)
        ])
    else:
        print(f"Generating {num_synthetic} synthetic images")
        rng = np.random.RandomState(42)
        items = []
        for i in range(num_synthetic):
            img_bytes = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8).tobytes()
            items.append({
                "image_bytes": img_bytes,
                "prompt": prompt,
                "index": i,
            })
        ds = ray.data.from_items(items)

    class VLAInferenceMapper:
        """Stateful mapper that loads the model once per worker."""

        def __init__(self, model_name: str):
            from models.load_model import load_model
            self.model_info = load_model(model_name, device="cpu")

        def __call__(self, batch: dict) -> dict:
            from models.load_model import create_sample_input
            from pipeline.infer import run_inference
            from PIL import Image

            results = {
                "index": [],
                "latency_ms": [],
                "output_text": [],
            }

            indices = batch["index"]
            prompts = batch["prompt"]

            for i in range(len(indices)):
                # Reconstruct image
                if "image_bytes" in batch:
                    img_arr = np.frombuffer(batch["image_bytes"][i], dtype=np.uint8).reshape(224, 224, 3)
                    image = Image.fromarray(img_arr)
                elif "image_path" in batch:
                    image = Image.open(batch["image_path"][i])
                else:
                    image = None

                p = prompts[i]
                inputs = create_sample_input(
                    self.model_info.processor, self.model_info.device, image, p
                )
                result = run_inference(self.model_info, inputs)

                results["index"].append(indices[i])
                results["latency_ms"].append(result["latency_ms"])

                if "text" in result:
                    results["output_text"].append(result["text"][:100])
                elif "actions" in result:
                    results["output_text"].append(str(result["actions"].tolist()[:3]) + "...")
                else:
                    results["output_text"].append("")

            return results

    print(f"Running batch inference with Ray Data (batch_size={batch_size})...")
    wall_start = time.perf_counter()

    result_ds = ds.map_batches(
        VLAInferenceMapper,
        fn_constructor_args=(model_name,),
        batch_size=batch_size,
        num_cpus=num_cpus_per_task,
        concurrency=2,
    )

    # Collect results
    collected = result_ds.take_all()
    wall_elapsed = time.perf_counter() - wall_start

    latencies = np.array([r["latency_ms"] for r in collected])
    total_images = len(collected)

    # Sample outputs
    samples = collected[:5]

    summary = {
        "total_images": total_images,
        "wall_time_sec": wall_elapsed,
        "throughput_img_per_sec": total_images / wall_elapsed,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
        "sample_outputs": [
            {"index": s["index"], "text": s["output_text"], "latency_ms": s["latency_ms"]}
            for s in samples
        ],
    }

    # Print results
    print(f"\nBatch Inference Results:")
    print(f"  Total images: {total_images}")
    print(f"  Wall time: {wall_elapsed:.2f}s")
    print(f"  Throughput: {summary['throughput_img_per_sec']:.2f} images/sec")
    print(f"  p50 latency: {summary['p50_latency_ms']:.2f} ms")
    print(f"  p95 latency: {summary['p95_latency_ms']:.2f} ms")

    if samples:
        print(f"\n  Sample outputs:")
        for s in samples:
            print(f"    [{s['index']}] ({s['latency_ms']:.1f}ms) {s['output_text']}")

    return summary
