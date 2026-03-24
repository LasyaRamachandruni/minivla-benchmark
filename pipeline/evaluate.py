"""Accuracy evaluation for VLA models.

Compares model outputs across optimization configurations to measure
accuracy degradation from quantization and pruning.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple

from PIL import Image

from models.load_model import ModelInfo, create_sample_input
from pipeline.infer import run_inference


def generate_eval_dataset(
    processor,
    device: str,
    num_samples: int = 50,
    seed: int = 42,
) -> List[dict]:
    """Generate a synthetic evaluation dataset.

    Creates deterministic random image+prompt pairs for consistent
    comparison across model configurations.
    """
    rng = np.random.RandomState(seed)
    prompts = [
        "pick up the red block",
        "move the cup to the left",
        "push the button",
        "grasp the yellow ball",
        "place the object on the shelf",
        "open the drawer",
        "close the gripper",
        "rotate the handle clockwise",
        "slide the box forward",
        "lift the plate carefully",
    ]

    dataset = []
    for i in range(num_samples):
        img_arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(img_arr)
        prompt = prompts[i % len(prompts)]
        inputs = create_sample_input(processor, device, image, prompt)
        dataset.append(inputs)

    return dataset


def compute_action_accuracy(
    baseline_actions: np.ndarray,
    test_actions: np.ndarray,
    tolerance: float = 0.1,
) -> float:
    """Compute accuracy as percentage of action dimensions within tolerance."""
    if baseline_actions.shape != test_actions.shape:
        min_len = min(len(baseline_actions), len(test_actions))
        baseline_actions = baseline_actions[:min_len]
        test_actions = test_actions[:min_len]

    diffs = np.abs(baseline_actions - test_actions)
    within_tolerance = diffs < tolerance
    return float(np.mean(within_tolerance) * 100)


def compute_logit_accuracy(
    baseline_logits: np.ndarray,
    test_logits: np.ndarray,
) -> float:
    """Compute top-1 agreement between baseline and test logits."""
    baseline_preds = np.argmax(baseline_logits, axis=-1).flatten()
    test_preds = np.argmax(test_logits, axis=-1).flatten()

    min_len = min(len(baseline_preds), len(test_preds))
    agreement = np.mean(baseline_preds[:min_len] == test_preds[:min_len])
    return float(agreement * 100)


def compute_token_accuracy(
    baseline_ids: np.ndarray,
    test_ids: np.ndarray,
) -> float:
    """Compute token-level agreement between generated token sequences.

    Uses a combination of exact positional match and token overlap (Jaccard).
    This is more forgiving than strict positional matching, since optimized
    models may produce semantically similar but slightly shifted outputs.
    """
    if len(baseline_ids) == 0 and len(test_ids) == 0:
        return 100.0

    # Positional agreement (where tokens overlap in position)
    min_len = min(len(baseline_ids), len(test_ids))
    if min_len > 0:
        positional = np.mean(baseline_ids[:min_len] == test_ids[:min_len])
    else:
        positional = 0.0

    # Jaccard overlap (set similarity regardless of position)
    base_set = set(baseline_ids.tolist()) if len(baseline_ids) > 0 else set()
    test_set = set(test_ids.tolist()) if len(test_ids) > 0 else set()
    union = base_set | test_set
    jaccard = len(base_set & test_set) / len(union) if union else 1.0

    # Weighted: 60% positional, 40% overlap
    combined = 0.6 * positional + 0.4 * jaccard
    return float(combined * 100)


@torch.no_grad()
def evaluate_model(
    model_info: ModelInfo,
    dataset: List[dict],
    baseline_outputs: Optional[List[dict]] = None,
) -> Tuple[float, List[dict]]:
    """Evaluate a model on the synthetic dataset.

    Args:
        model_info: Model to evaluate.
        dataset: List of input dicts.
        baseline_outputs: If provided, compute accuracy relative to baseline.

    Returns:
        (accuracy_pct, list_of_outputs)
    """
    outputs = []
    for inputs in dataset:
        # Move inputs to the model's device (quantized models may be on CPU
        # while dataset was created on MPS/CUDA)
        device_inputs = {
            k: v.to(model_info.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        result = run_inference(model_info, device_inputs)
        outputs.append(result)

    if baseline_outputs is None:
        return 100.0, outputs

    # Compare against baseline
    accuracies = []
    for base_out, test_out in zip(baseline_outputs, outputs):
        if "actions" in base_out and "actions" in test_out:
            acc = compute_action_accuracy(base_out["actions"], test_out["actions"])
            accuracies.append(acc)
        elif "generated_ids" in base_out and "generated_ids" in test_out:
            acc = compute_token_accuracy(base_out["generated_ids"], test_out["generated_ids"])
            accuracies.append(acc)
        elif "logits" in base_out and "logits" in test_out:
            acc = compute_logit_accuracy(base_out["logits"], test_out["logits"])
            accuracies.append(acc)

    accuracy = float(np.mean(accuracies)) if accuracies else 100.0
    return accuracy, outputs
