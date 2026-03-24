"""Single inference run for VLA models."""

import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from models.load_model import ModelInfo, MockVLMModel, create_sample_input

# Maximum tokens to generate for real VLMs (kept short for benchmarking)
MAX_NEW_TOKENS = 20


def _is_generative_model(model_info: ModelInfo) -> bool:
    """Check if this is a real HF generative VLM (vs mock or ONNX)."""
    return (
        model_info.backend == "pytorch"
        and hasattr(model_info.model, "generate")
        and not isinstance(model_info.model, MockVLMModel)
    )


@torch.no_grad()
def run_inference_pytorch(model_info: ModelInfo, inputs: dict) -> dict:
    """Run a single PyTorch inference pass.

    For real VLMs (SmolVLM, LLaVA, etc.), uses model.generate().
    For the mock model, uses a direct forward pass.
    """
    model = model_info.model

    if model_info.device == "cuda":
        torch.cuda.synchronize()

    if _is_generative_model(model_info):
        # Real VLM: use generate() to produce text tokens
        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if model_info.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract only the newly generated tokens (strip the input prompt)
        input_len = inputs.get("input_ids", torch.empty(1, 0)).shape[-1]
        new_token_ids = generated_ids[0, input_len:].cpu().numpy()

        result = {
            "latency_ms": elapsed_ms,
            "generated_ids": new_token_ids,
        }

        # Decode if processor available
        if model_info.processor is not None and hasattr(model_info.processor, "decode"):
            result["text"] = model_info.processor.decode(new_token_ids, skip_special_tokens=True)

    else:
        # Mock model: direct forward pass
        start = time.perf_counter()
        outputs = model(**inputs)
        if model_info.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = {"latency_ms": elapsed_ms}

        if isinstance(outputs, dict):
            if "actions" in outputs:
                result["actions"] = outputs["actions"].cpu().numpy()
            if "logits" in outputs:
                result["logits"] = outputs["logits"].cpu().numpy()
        else:
            if hasattr(outputs, "logits"):
                result["logits"] = outputs.logits.cpu().numpy()

    return result


def run_inference_onnx(model_info: ModelInfo, inputs: dict) -> dict:
    """Run a single ONNX Runtime inference pass."""
    session = model_info.model

    # Convert torch tensors to numpy for ONNX Runtime
    ort_inputs = {}
    for inp in session.get_inputs():
        name = inp.name
        if name in inputs:
            val = inputs[name]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            ort_inputs[name] = val

    start = time.perf_counter()
    outputs = session.run(None, ort_inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {"latency_ms": elapsed_ms}
    output_names = [o.name for o in session.get_outputs()]
    for name, val in zip(output_names, outputs):
        result[name] = val

    return result


def run_inference(model_info: ModelInfo, inputs: dict) -> dict:
    """Run inference using the appropriate backend."""
    if model_info.backend == "onnx":
        return run_inference_onnx(model_info, inputs)
    return run_inference_pytorch(model_info, inputs)


def single_inference(model_info: ModelInfo, image: Optional[Image.Image] = None,
                     prompt: str = "pick up the red block") -> dict:
    """Complete single inference: preprocess -> infer -> return results."""
    inputs = create_sample_input(model_info.processor, model_info.device, image, prompt)
    return run_inference(model_info, inputs)
