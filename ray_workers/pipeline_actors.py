"""Pipeline-parallel Ray actors for VLA inference.

Splits the VLA model into three pipeline stages, each running as a
separate Ray actor:
  Stage 1: VisionEncoderActor  — image -> visual embeddings
  Stage 2: LanguageDecoderActor — visual embeddings + text -> token logits
  Stage 3: ActionDecoderActor   — decoded features -> 7-DoF actions

This mirrors how production VLA systems are deployed: each stage can
scale independently, run on different hardware (e.g., GPU for vision,
CPU for action decoding), and be profiled in isolation.
"""

import time
from typing import Optional

import numpy as np
import ray
import torch
from PIL import Image


@ray.remote
class VisionEncoderActor:
    """Pipeline stage 1: Encodes images into visual feature embeddings."""

    def __init__(self, model_name: str = "mock", device: str = "cpu"):
        from models.load_model import load_model, MockVLMModel
        self.model_info = load_model(model_name, device=device)
        self.model = self.model_info.model
        self.is_mock = isinstance(self.model, MockVLMModel)
        self.call_count = 0
        self.total_ms = 0.0

    @torch.no_grad()
    def encode(self, pixel_values_np: np.ndarray) -> dict:
        """Encode image pixels into visual features.

        Args:
            pixel_values_np: Image tensor as numpy array (B, C, H, W).

        Returns:
            Dict with 'vision_features' (numpy) and 'stage_latency_ms'.
        """
        pixel_values = torch.tensor(pixel_values_np).to(self.model_info.device)

        start = time.perf_counter()
        if self.is_mock:
            features = self.model.vision_encoder(pixel_values)  # (B, D)
        else:
            # Real HF VLM: run the vision encoder
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                vision_out = self.model.model.vision_model(pixel_values)
                features = vision_out.last_hidden_state  # (B, N, D)
            else:
                features = pixel_values  # Fallback
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.call_count += 1
        self.total_ms += elapsed_ms

        return {
            "vision_features": features.cpu().numpy(),
            "stage_latency_ms": elapsed_ms,
        }

    def get_stats(self) -> dict:
        return {
            "stage": "VisionEncoder",
            "call_count": self.call_count,
            "total_ms": self.total_ms,
            "avg_ms": self.total_ms / self.call_count if self.call_count > 0 else 0,
        }


@ray.remote
class LanguageDecoderActor:
    """Pipeline stage 2: Processes vision features + text into decoded representations."""

    def __init__(self, model_name: str = "mock", device: str = "cpu"):
        from models.load_model import load_model, MockVLMModel
        self.model_info = load_model(model_name, device=device)
        self.model = self.model_info.model
        self.is_mock = isinstance(self.model, MockVLMModel)
        self.call_count = 0
        self.total_ms = 0.0

    @torch.no_grad()
    def decode(self, vision_features_np: np.ndarray, input_ids_np: np.ndarray) -> dict:
        """Decode vision features with text into language representations.

        Args:
            vision_features_np: Vision encoder output as numpy.
            input_ids_np: Tokenized text input as numpy.

        Returns:
            Dict with 'decoded_features' (numpy) and 'stage_latency_ms'.
        """
        device = self.model_info.device
        vision_features = torch.tensor(vision_features_np).to(device)
        input_ids = torch.tensor(input_ids_np).to(device)

        start = time.perf_counter()
        if self.is_mock:
            if vision_features.dim() == 2:
                vision_features = vision_features.unsqueeze(1)  # (B, 1, D)
            text_features = self.model.text_embedding(input_ids)  # (B, T, D)
            decoded = self.model.decoder(text_features, vision_features)  # (B, T, D)
        else:
            # For real models, simulate decoder processing
            decoded = vision_features
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.call_count += 1
        self.total_ms += elapsed_ms

        return {
            "decoded_features": decoded.cpu().numpy(),
            "stage_latency_ms": elapsed_ms,
        }

    def get_stats(self) -> dict:
        return {
            "stage": "LanguageDecoder",
            "call_count": self.call_count,
            "total_ms": self.total_ms,
            "avg_ms": self.total_ms / self.call_count if self.call_count > 0 else 0,
        }


@ray.remote
class ActionDecoderActor:
    """Pipeline stage 3: Converts decoded features into 7-DoF robot actions."""

    def __init__(self, model_name: str = "mock", device: str = "cpu"):
        from models.load_model import load_model, MockVLMModel
        self.model_info = load_model(model_name, device=device)
        self.model = self.model_info.model
        self.is_mock = isinstance(self.model, MockVLMModel)
        self.call_count = 0
        self.total_ms = 0.0

    @torch.no_grad()
    def predict_action(self, decoded_features_np: np.ndarray) -> dict:
        """Predict robot actions from decoded features.

        Args:
            decoded_features_np: Language decoder output as numpy.

        Returns:
            Dict with 'actions' (list), 'stage_latency_ms'.
        """
        device = self.model_info.device
        decoded = torch.tensor(decoded_features_np).to(device)

        start = time.perf_counter()
        if self.is_mock:
            pooled = decoded.mean(dim=1)  # (B, D)
            actions = self.model.action_head(pooled)  # (B, 7)
        else:
            # For real models, produce synthetic actions from features
            if decoded.dim() == 3:
                pooled = decoded.mean(dim=1)
            else:
                pooled = decoded
            actions = torch.randn(pooled.shape[0], 7)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.call_count += 1
        self.total_ms += elapsed_ms

        return {
            "actions": actions.cpu().numpy().tolist(),
            "stage_latency_ms": elapsed_ms,
        }

    def get_stats(self) -> dict:
        return {
            "stage": "ActionDecoder",
            "call_count": self.call_count,
            "total_ms": self.total_ms,
            "avg_ms": self.total_ms / self.call_count if self.call_count > 0 else 0,
        }


def run_pipeline_benchmark(
    model_name: str = "mock",
    num_requests: int = 100,
    prompt: str = "pick up the red block",
) -> dict:
    """Benchmark the pipeline-parallel VLA inference.

    Splits inference across three Ray actors and measures per-stage
    latency and end-to-end throughput.
    """
    import ray
    from models.load_model import load_model, create_sample_input, MockProcessor

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Load model locally just to get processor and sample inputs
    model_info = load_model(model_name, device="cpu")
    processor = model_info.processor

    print("Spawning pipeline actors: VisionEncoder -> LanguageDecoder -> ActionDecoder")
    vision_actor = VisionEncoderActor.remote(model_name=model_name)
    language_actor = LanguageDecoderActor.remote(model_name=model_name)
    action_actor = ActionDecoderActor.remote(model_name=model_name)

    # Prepare inputs
    sample = create_sample_input(processor, "cpu", prompt=prompt)
    pixel_values_np = sample["pixel_values"].cpu().numpy()
    input_ids_np = sample["input_ids"].cpu().numpy()

    # Warmup
    print("Warming up pipeline...")
    for _ in range(5):
        v_out = ray.get(vision_actor.encode.remote(pixel_values_np))
        l_out = ray.get(language_actor.decode.remote(v_out["vision_features"], input_ids_np))
        ray.get(action_actor.predict_action.remote(l_out["decoded_features"]))

    # Benchmark: sequential pipeline (measures per-stage latency)
    print(f"Running {num_requests} requests through pipeline...")
    stage_latencies = {"vision": [], "language": [], "action": []}
    e2e_latencies = []
    wall_start = time.perf_counter()

    for i in range(num_requests):
        e2e_start = time.perf_counter()

        v_out = ray.get(vision_actor.encode.remote(pixel_values_np))
        stage_latencies["vision"].append(v_out["stage_latency_ms"])

        l_out = ray.get(language_actor.decode.remote(v_out["vision_features"], input_ids_np))
        stage_latencies["language"].append(l_out["stage_latency_ms"])

        a_out = ray.get(action_actor.predict_action.remote(l_out["decoded_features"]))
        stage_latencies["action"].append(a_out["stage_latency_ms"])

        e2e_latencies.append((time.perf_counter() - e2e_start) * 1000)

    wall_elapsed = time.perf_counter() - wall_start

    # Collect stats
    actor_stats = ray.get([
        vision_actor.get_stats.remote(),
        language_actor.get_stats.remote(),
        action_actor.get_stats.remote(),
    ])

    # Cleanup
    for a in [vision_actor, language_actor, action_actor]:
        ray.kill(a)

    e2e = np.array(e2e_latencies)
    result = {
        "num_requests": num_requests,
        "wall_time_sec": wall_elapsed,
        "throughput_qps": num_requests / wall_elapsed,
        "e2e_p50_ms": float(np.percentile(e2e, 50)),
        "e2e_p95_ms": float(np.percentile(e2e, 95)),
        "e2e_p99_ms": float(np.percentile(e2e, 99)),
        "stage_stats": {},
        "actor_stats": actor_stats,
    }

    for stage_name, lats in stage_latencies.items():
        arr = np.array(lats)
        result["stage_stats"][stage_name] = {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "avg_ms": float(np.mean(arr)),
            "pct_of_e2e": float(np.mean(arr) / np.mean(e2e) * 100),
        }

    return result
