"""Vision-Language-Action model wrappers.

Every model exposes the same three-step interface so evaluation, latency benchmarks and the
Ray workers all run *real* inference through the same code path:

    batch   = model.prepare(frames)          # tokenize / normalize / move to device
    actions = model.forward(batch, noise)    # policy forward pass (normalized action space)
    actions = model.postprocess(actions)     # un-normalize, move to CPU -> (B, chunk, action_dim)

`model.predict(frames)` chains the three. SmolVLA is a flow-matching policy, so its output
depends on the initial noise; `make_noise` draws it from a per-frame seed so that the FP32
baseline and every optimized variant see identical noise for the same frame.
"""

import copy
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from models.frames import Frame


@dataclass
class ModelSpec:
    key: str
    repo_id: str
    # dataset camera key -> policy camera key, applied by LeRobot's rename step
    rename_map: Dict[str, str] = field(default_factory=dict)
    # True only when the checkpoint was trained on the same action space as the eval dataset,
    # so comparing its actions with dataset ground truth is meaningful.
    gt_comparable: bool = False
    # If the dataset state is longer than the policy expects, feed only the first dims.
    allow_state_truncation: bool = False
    notes: str = ""


class VLAModel:
    """Common interface; see module docstring."""

    name: str
    device: str
    precision: str
    policy: torch.nn.Module
    action_dim: int
    chunk_size: int
    noise_dim: int
    spec: ModelSpec

    # Linear layers whose `.weight.dtype` is read inside the policy's forward pass. Swapping
    # them for quantized modules breaks that (the weight becomes a method), so int8 schemes
    # must leave them in floating point.
    int8_skip_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # -- the three inference steps -------------------------------------------------
    def prepare(self, frames: List[Frame]) -> dict:
        raise NotImplementedError

    def forward(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- variants --------------------------------------------------------------------
    def backbone(self) -> torch.nn.Module:
        """The large sub-module that half-precision variants cast (heads stay FP32)."""
        raise NotImplementedError

    def with_policy(self, policy: torch.nn.Module, device: str, precision: str) -> "VLAModel":
        """A shallow copy of this wrapper around another policy (an optimized variant)."""
        new = copy.copy(self)
        new.policy = policy
        new.device = device
        new.precision = precision
        return new

    # -- helpers -----------------------------------------------------------------------
    def make_noise(self, seeds: Sequence[int]) -> torch.Tensor:
        """(B, chunk, noise_dim) float32 standard-normal noise, one generator per frame seed."""
        out = []
        for s in seeds:
            g = torch.Generator(device="cpu").manual_seed(int(s))
            out.append(torch.randn(self.chunk_size, self.noise_dim, generator=g))
        return torch.stack(out).to(self.device)

    def synchronize(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def run(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        return self.postprocess(self.forward(batch, noise))

    @torch.inference_mode()
    def predict(self, frames: List[Frame], seeds: Optional[Sequence[int]] = None,
                batch_size: int = 1) -> np.ndarray:
        """Actions for each frame, (N, chunk, action_dim) float32 numpy."""
        if seeds is None:
            seeds = range(len(frames))
        seeds = list(seeds)
        outs = []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            batch = self.prepare(chunk)
            noise = self.make_noise(seeds[i:i + batch_size])
            outs.append(self.run(batch, noise).float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def size_mb(self) -> float:
        return state_dict_size_mb(self.policy)

    def num_parameters(self) -> int:
        return count_parameters(self.policy)

    def describe(self) -> dict:
        return {
            "model_key": self.spec.key,
            "repo_id": self.spec.repo_id,
            "revision": getattr(self, "revision", None),
            "device": self.device,
            "precision": self.precision,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "gt_comparable": self.spec.gt_comparable,
            "notes": self.spec.notes,
        }


def state_dict_size_mb(module: torch.nn.Module) -> float:
    """Serialized size of the state dict.

    Counting `parameters()` misses the packed weights of dynamically quantized layers (they are
    not Parameters), which under-reports int8 models; serializing counts what would be saved.
    """
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def count_parameters(module: torch.nn.Module) -> int:
    """Parameter count, including the packed weights of dynamically quantized Linear layers."""
    n = sum(p.numel() for p in module.parameters())
    for m in module.modules():
        if type(m).__module__.startswith("torch.ao.nn.quantized") and hasattr(m, "weight") and callable(m.weight):
            w = m.weight()
            n += w.numel()
            b = m.bias() if callable(getattr(m, "bias", None)) else None
            n += b.numel() if b is not None else 0
    return n


def frames_to_tensors(frames: List[Frame], camera_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """uint8 HWC images -> float32 (B, 3, H, W) in [0, 1], the format LeRobot datasets produce."""
    out = {}
    for key in camera_keys:
        arr = np.stack([f.images[key] for f in frames])  # (B, H, W, 3)
        out[key] = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    return out


class LeRobotVLA(VLAModel):
    """A LeRobot policy (SmolVLA) plus its saved pre/post-processor pipelines."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", revision: Optional[str] = None,
                 precision: str = "fp32", load_policy: bool = True):
        """`load_policy=False` loads only the config and processors (for pre/post-processing workers)."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        self.spec = spec
        self.name = spec.key
        self.revision = revision or _resolve_revision(spec.repo_id)
        self.device = device

        cfg = PreTrainedConfig.from_pretrained(spec.repo_id, revision=self.revision)
        cfg.device = device
        policy = None
        if load_policy:
            policy_cls = get_policy_class(cfg.type)
            policy = policy_cls.from_pretrained(spec.repo_id, config=cfg, revision=self.revision)
            policy.eval()
        self.config = cfg

        self.policy_image_keys = list(self.config.image_features)
        inverse = {v: k for k, v in spec.rename_map.items()}
        # dataset keys we need to supply, in policy order
        self.camera_keys = [inverse.get(k, k) for k in self.policy_image_keys]
        self.state_dim = self.config.input_features["observation.state"].shape[0]
        self.action_dim = self.config.output_features["action"].shape[0]
        self.chunk_size = self.config.chunk_size
        self.noise_dim = self.config.max_action_dim
        self._processors = {}

        self.policy = policy
        # LeRobot loads SmolVLA's VLM in bf16 and the rest in fp32; "fp32" makes the baseline uniform.
        self.precision = "lerobot-default"
        if precision == "fp32" and policy is not None:
            self.policy = policy.float()
            self.precision = "fp32"

    def _get_processors(self, device: str):
        if device not in self._processors:
            from lerobot.policies.factory import make_pre_post_processors

            pre_overrides = {"device_processor": {"device": device}}
            if self.spec.rename_map:
                pre_overrides["rename_observations_processor"] = {"rename_map": dict(self.spec.rename_map)}
            self._processors[device] = make_pre_post_processors(
                self.config,
                pretrained_path=self.spec.repo_id,
                pretrained_revision=self.revision,
                preprocessor_overrides=pre_overrides,
            )
        return self._processors[device]

    def backbone(self) -> torch.nn.Module:
        # VLM + action expert. The small state/action/time projections stay FP32, which is the
        # same mixed layout LeRobot itself uses when it loads the VLM in bf16.
        return self.policy.model.vlm_with_expert

    def adapt_state(self, state: np.ndarray) -> np.ndarray:
        if state.shape[-1] == self.state_dim:
            return state
        if state.shape[-1] > self.state_dim and self.spec.allow_state_truncation:
            return state[..., :self.state_dim]
        raise ValueError(
            f"{self.name} expects a {self.state_dim}-D state, got {state.shape[-1]}-D. "
            "This checkpoint was not trained on this dataset's state space."
        )

    def prepare(self, frames: List[Frame]) -> dict:
        missing = [k for k in self.camera_keys if k not in frames[0].images]
        if missing:
            raise KeyError(f"frames are missing camera(s) {missing} required by {self.name}")
        raw = frames_to_tensors(frames, self.camera_keys)
        raw["observation.state"] = torch.from_numpy(np.stack([self.adapt_state(f.state) for f in frames]))
        raw["task"] = [f.task for f in frames]
        pre, _ = self._get_processors(self.device)
        return pre(raw)

    def forward(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        # predict_action_chunk mutates its input dict; hand it a shallow copy
        return self.policy.predict_action_chunk(dict(batch), noise=noise)

    def postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        _, post = self._get_processors(self.device)
        return post(actions.float())


def _resolve_revision(repo_id: str) -> Optional[str]:
    """Commit sha of a Hub repo so results record exactly which weights were used."""
    import os

    if os.path.isdir(repo_id):
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception:
        return None
