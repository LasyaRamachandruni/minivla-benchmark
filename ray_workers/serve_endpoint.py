"""Ray Serve deployment for VLA inference.

Exposes the VLA model as an HTTP endpoint, enabling production-style
serving with autoscaling, batching, and health checks.

Usage:
    python ray_workers/serve_endpoint.py --model mock --port 8000

    # Query the endpoint:
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "pick up the red block"}'
"""

import os
import sys
import time
import json
import io
import base64
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_serve_deployment(model_name: str = "mock", num_replicas: int = 1):
    """Create a Ray Serve deployment for VLA inference."""
    import ray
    from ray import serve

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": 1},
    )
    class VLAServeDeployment:
        def __init__(self, model_name: str):
            from models.load_model import load_model
            self.model_info = load_model(model_name, device="cpu")
            self.request_count = 0
            self.total_latency_ms = 0.0
            print(f"VLA Serve replica ready: {model_name} ({self.model_info.size_mb:.1f} MB)")

        async def __call__(self, request) -> dict:
            from models.load_model import create_sample_input
            from pipeline.infer import run_inference
            from PIL import Image

            body = await request.json()
            prompt = body.get("prompt", "pick up the red block")

            # Decode base64 image if provided
            image = None
            if "image_base64" in body:
                img_bytes = base64.b64decode(body["image_base64"])
                image = Image.open(io.BytesIO(img_bytes))

            inputs = create_sample_input(
                self.model_info.processor, self.model_info.device, image, prompt
            )
            result = run_inference(self.model_info, inputs)

            self.request_count += 1
            self.total_latency_ms += result["latency_ms"]

            response = {
                "latency_ms": result["latency_ms"],
                "model": self.model_info.name,
                "prompt": prompt,
            }

            if "actions" in result:
                response["actions"] = result["actions"].tolist()
            if "text" in result:
                response["text"] = result["text"]
            if "generated_ids" in result:
                response["generated_ids"] = result["generated_ids"].tolist()

            return response

        async def health(self, request) -> dict:
            return {
                "status": "healthy",
                "model": self.model_info.name,
                "requests_served": self.request_count,
                "avg_latency_ms": (
                    self.total_latency_ms / self.request_count
                    if self.request_count > 0 else 0
                ),
            }

    app = VLAServeDeployment.bind(model_name)
    return app


def run_serve(model_name: str = "mock", port: int = 8000, num_replicas: int = 1):
    """Start the Ray Serve endpoint."""
    from ray import serve

    app = create_serve_deployment(model_name, num_replicas)
    serve.run(app, host="0.0.0.0", port=port)

    print(f"\nVLA Serve endpoint running at http://localhost:{port}")
    print(f"  POST /         - Run inference (body: {{\"prompt\": \"...\", \"image_base64\": \"...\"}})")
    print(f"  Model: {model_name}, Replicas: {num_replicas}")
    print("\nPress Ctrl+C to stop.")

    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()


def benchmark_serve_endpoint(
    url: str = "http://localhost:8000",
    num_requests: int = 50,
    concurrency: int = 4,
    prompt: str = "pick up the red block",
) -> dict:
    """Benchmark a running Ray Serve endpoint with concurrent requests."""
    import concurrent.futures
    import urllib.request

    def send_request(i):
        payload = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"request_id": i, "total_ms": elapsed_ms, **result}

    print(f"Benchmarking {url} with {num_requests} requests, concurrency={concurrency}")
    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    wall_elapsed = time.perf_counter() - wall_start
    latencies = np.array([r["total_ms"] for r in results])

    return {
        "num_requests": num_requests,
        "concurrency": concurrency,
        "wall_time_sec": wall_elapsed,
        "throughput_qps": num_requests / wall_elapsed,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLA Ray Serve Endpoint")
    parser.add_argument("--model", default="mock", help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Serve port")
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    args = parser.parse_args()

    run_serve(args.model, args.port, args.replicas)
