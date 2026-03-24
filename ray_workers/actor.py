"""Ray actor wrapping the VLA inference pipeline."""

import time
from typing import Optional

import numpy as np
import ray
import torch
from PIL import Image


@ray.remote
class VLAInferenceActor:
    """Ray actor that encapsulates a VLA model for distributed inference.

    Each actor loads its own copy of the model, enabling horizontal
    scaling across multiple workers (CPUs/GPUs).
    """

    def __init__(self, model_name: str = "mock", model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """Initialize the actor with a loaded model.

        Args:
            model_name: Model name to load ('mock', 'mobilevlm', etc.)
            model_path: Path to ONNX model file (overrides model_name).
            device: Target device. Auto-detected if None.
        """
        # Import here to avoid serialization issues
        from models.load_model import load_model, load_onnx_model

        if model_path and model_path.endswith(".onnx"):
            self.model_info = load_onnx_model(model_path)
        else:
            self.model_info = load_model(model_name, device=device or "cpu")

        self.request_count = 0
        self.total_latency_ms = 0.0

    def infer(self, image_bytes: Optional[bytes] = None,
              prompt: str = "pick up the red block") -> dict:
        """Run inference on a single image+prompt pair.

        Args:
            image_bytes: JPEG/PNG bytes, or None for random input.
            prompt: Text instruction.

        Returns:
            Dict with 'latency_ms', 'actions', 'worker_id'.
        """
        from models.load_model import create_sample_input
        from pipeline.infer import run_inference

        image = None
        if image_bytes:
            import io
            image = Image.open(io.BytesIO(image_bytes))

        inputs = create_sample_input(
            self.model_info.processor, self.model_info.device, image, prompt
        )
        result = run_inference(self.model_info, inputs)

        self.request_count += 1
        self.total_latency_ms += result["latency_ms"]

        # Serialize numpy arrays for Ray transport
        if "actions" in result:
            result["actions"] = result["actions"].tolist()
        if "generated_ids" in result:
            result["generated_ids"] = result["generated_ids"].tolist()
        if "logits" in result:
            del result["logits"]  # Too large to send back

        result["worker_id"] = ray.get_runtime_context().get_actor_id()
        return result

    def get_stats(self) -> dict:
        """Return worker utilization statistics."""
        return {
            "worker_id": ray.get_runtime_context().get_actor_id(),
            "request_count": self.request_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": (
                self.total_latency_ms / self.request_count
                if self.request_count > 0 else 0
            ),
        }

    def health_check(self) -> bool:
        return True
