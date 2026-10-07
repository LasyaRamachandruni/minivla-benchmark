"""Model registry and loading."""

import os
from typing import Optional

import torch

from models.vla import LeRobotVLA, ModelSpec, VLAModel

MODEL_REGISTRY = {
    "smolvla_libero": ModelSpec(
        key="smolvla_libero",
        repo_id="HuggingFaceVLA/smolvla_libero",
        gt_comparable=True,
        notes=(
            "SmolVLA fine-tuned on LIBERO with the same features as HuggingFaceVLA/libero "
            "(image, image2, 8-D state, 7-D action). LIBERO has a single train split, so these "
            "frames were likely seen in fine-tuning: errors measure fit, not generalisation."
        ),
    ),
    "smolvla_base": ModelSpec(
        key="smolvla_base",
        repo_id="lerobot/smolvla_base",
        rename_map={
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        },
        gt_comparable=False,
        allow_state_truncation=True,
        notes=(
            "Pretrained SmolVLA (SO-100 community data: 6-D joint state/action, 3 cameras). On LIBERO "
            "frames it gets the first 6 state dims and 2 of 3 cameras, and predicts 6-D actions in a "
            "different action space, so it is NOT compared with LIBERO ground truth. Only latency and "
            "agreement with its own FP32 outputs are reported."
        ),
    ),
    "mock": ModelSpec(key="mock", repo_id="mock", notes="Random-weight mock; tests and plumbing only."),
}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: Optional[str] = None, revision: Optional[str] = None,
               gt_comparable: bool = False, load_policy: bool = True) -> VLAModel:
    """Load a VLA by registry key, Hub repo id, or local LeRobot policy directory.

    The returned model is FP32 on `device`. Unknown repo ids / paths are treated as not
    comparable with ground truth unless `gt_comparable=True` is passed explicitly.
    `load_policy=False` skips the weights (enough for prepare/postprocess).
    """
    device = device or get_device()
    if model_name == "mock":
        from models.mock_vla import MockVLA

        return MockVLA(device=device)

    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        spec = ModelSpec(
            key=os.path.basename(model_name.rstrip("/")),
            repo_id=model_name,
            gt_comparable=gt_comparable,
            notes="User-supplied LeRobot policy.",
        )
    return LeRobotVLA(spec, device=device, revision=revision, precision="fp32", load_policy=load_policy)
