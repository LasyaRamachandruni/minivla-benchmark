"""Optimization pipeline: ONNX export, quantization, pruning."""

import os
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.utils.prune as prune

from models.load_model import (
    ModelInfo, MockVLMModel, MockProcessor,
    get_model_size_mb, get_onnx_model_size_mb, load_onnx_model,
)


def _is_mock_model(model_info: ModelInfo) -> bool:
    return isinstance(model_info.model, MockVLMModel)


def export_to_onnx(model_info: ModelInfo, output_path: str,
                   image_size: int = 224, max_seq_len: int = 32) -> str:
    """Export a PyTorch model to ONNX format.

    Only supported for the mock model. Real VLMs (encoder-decoder)
    have complex architectures not easily exportable to ONNX.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = model_info.model
    model.eval()

    # Create dummy inputs on CPU for export
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)
    dummy_input_ids = torch.randint(0, 100, (1, max_seq_len))

    # Move model to CPU for export
    model_cpu = model.cpu()

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model_cpu,
        (dummy_pixel_values, dummy_input_ids),
        output_path,
        input_names=["pixel_values", "input_ids"],
        output_names=["actions", "logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "actions": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Move model back to original device
    model.to(model_info.device)

    size_mb = get_onnx_model_size_mb(output_path)
    print(f"ONNX model exported: {size_mb:.1f} MB")
    return output_path


def quantize_dynamic_int8(onnx_path: str, output_path: Optional[str] = None) -> str:
    """Apply dynamic INT8 quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_int8{ext}"

    print(f"Quantizing (INT8): {onnx_path} -> {output_path}")
    # Use QUInt8 to avoid ConvInteger issues (same fix as EdgeOpt)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    original_size = get_onnx_model_size_mb(onnx_path)
    quantized_size = get_onnx_model_size_mb(output_path)
    print(f"Quantization complete: {original_size:.1f} MB -> {quantized_size:.1f} MB "
          f"({(1 - quantized_size / original_size) * 100:.1f}% reduction)")
    return output_path


def apply_pytorch_dynamic_quantization(model_info: ModelInfo) -> ModelInfo:
    """Apply PyTorch dynamic quantization (INT8) to a real VLM.

    Quantizes Linear layers to int8 using PyTorch's built-in dynamic
    quantization. Works on CPU only.
    """
    print("Applying PyTorch dynamic INT8 quantization...")

    # Ensure a quantization engine is available
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"

    model = copy.deepcopy(model_info.model)
    model = model.cpu().float()  # Quantization needs float32 on CPU
    model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    size_mb = get_model_size_mb(quantized_model)
    print(f"Quantized model size: {size_mb:.1f} MB (was {model_info.size_mb:.1f} MB)")

    return ModelInfo(
        name=f"{model_info.name}_int8",
        backend="pytorch",
        size_mb=size_mb,
        device="cpu",  # Quantized models run on CPU
        model=quantized_model,
        processor=model_info.processor,
    )


def apply_structured_pruning(model_info: ModelInfo, amount: float = 0.3) -> ModelInfo:
    """Apply structured (channel) pruning to linear layers.

    Args:
        model_info: Loaded PyTorch model.
        amount: Fraction of channels to prune (0.0 to 1.0).

    Returns:
        New ModelInfo with pruned model.
    """
    print(f"Applying structured pruning (amount={amount})...")

    model = copy.deepcopy(model_info.model)
    model.eval()

    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip very small layers (embeddings, heads with few features)
            if module.weight.shape[0] < 4 or module.weight.shape[1] < 4:
                continue
            try:
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                prune.remove(module, "weight")
                pruned_count += 1
            except Exception:
                continue  # Skip layers that can't be pruned
        elif isinstance(module, torch.nn.Conv2d):
            if module.weight.shape[0] < 4:
                continue
            try:
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                prune.remove(module, "weight")
                pruned_count += 1
            except Exception:
                continue

    size_mb = get_model_size_mb(model)
    print(f"Pruned {pruned_count} layers. Model size: {size_mb:.1f} MB")

    return ModelInfo(
        name=f"{model_info.name}_pruned_{amount}",
        backend="pytorch",
        size_mb=size_mb,
        device=model_info.device,
        model=model,
        processor=model_info.processor,
    )


def optimize_full_pipeline(
    model_info: ModelInfo,
    output_dir: str = "optimized",
    quant_type: str = "int8",
    prune_amount: float = 0.3,
) -> dict:
    """Run the full optimization pipeline.

    For mock models: ONNX export -> ONNX quantize -> prune -> pruned ONNX + quantize
    For real VLMs: PyTorch dynamic quantize -> prune -> pruned quantize
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    if _is_mock_model(model_info):
        # Mock model: full ONNX pipeline
        onnx_path = os.path.join(output_dir, f"{model_info.name}_baseline.onnx")
        export_to_onnx(model_info, onnx_path)
        results["ONNX Export"] = load_onnx_model(onnx_path)

        if quant_type == "int8":
            quant_path = quantize_dynamic_int8(onnx_path)
            results["INT8 Quantized"] = load_onnx_model(quant_path)

        pruned_info = apply_structured_pruning(model_info, amount=prune_amount)
        results["Pruned PyTorch"] = pruned_info

        pruned_onnx_path = os.path.join(output_dir, f"{model_info.name}_pruned.onnx")
        export_to_onnx(pruned_info, pruned_onnx_path)
        pruned_quant_path = quantize_dynamic_int8(pruned_onnx_path)
        results["Pruned + Quantized"] = load_onnx_model(pruned_quant_path)

    else:
        # Real VLM: PyTorch-native optimization
        # Step 1: Dynamic INT8 quantization
        if quant_type == "int8":
            quant_info = apply_pytorch_dynamic_quantization(model_info)
            results["INT8 Quantized"] = quant_info

        # Step 2: Structured pruning
        pruned_info = apply_structured_pruning(model_info, amount=prune_amount)
        results["Pruned"] = pruned_info

        # Step 3: Pruned + quantized
        if quant_type == "int8":
            pruned_quant_info = apply_pytorch_dynamic_quantization(pruned_info)
            results["Pruned + Quantized"] = pruned_quant_info

    return results
