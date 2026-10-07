"""A small, randomly initialised stand-in for SmolVLA used by the tests and for quick plumbing checks.

It has the same *structure* the optimizations care about — a backbone of attention blocks with
`q/k/v/o_proj` layers (whose `.weight.dtype` the forward pass reads, exactly like LeRobot's
SmolVLA), gated `gate/up/down_proj` MLPs and a plain `fc1/fc2` vision MLP — plus FP32 state and
action projections and a noise input. Its weights are random, so its actions mean nothing and
are never compared with ground truth.
"""

import math
import zlib
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.frames import Frame
from models.vla import ModelSpec, VLAModel, frames_to_tensors

MOCK_SPEC = ModelSpec(
    key="mock",
    repo_id="mock",
    gt_comparable=False,
    notes="Random-weight mock with SmolVLA-like structure; for tests and plumbing only.",
)


class GatedMLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PlainMLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class Block(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = Attention(d)
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = GatedMLP(d, ff)

    def forward(self, x):
        # Like LeRobot's SmolVLA forward: cast activations to the attention weights' dtype.
        h = self.input_layernorm(x).to(dtype=self.self_attn.q_proj.weight.dtype)
        q, k, v = self.self_attn.q_proj(h), self.self_attn.k_proj(h), self.self_attn.v_proj(h)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1) @ v
        x = x + self.self_attn.o_proj(att)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, d, ff, num_layers, vocab):
        super().__init__()
        self.patch = nn.Conv2d(3, d, kernel_size=8, stride=8)
        self.vision_mlp = PlainMLP(d, ff)
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d, ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d)


class MockVLAPolicy(nn.Module):
    def __init__(self, num_cameras=2, state_dim=8, action_dim=7, chunk_size=10, noise_dim=8,
                 d=64, ff=256, num_layers=2, vocab=1000):
        super().__init__()
        self.num_cameras, self.action_dim, self.chunk_size, self.noise_dim = num_cameras, action_dim, chunk_size, noise_dim
        self.backbone = Backbone(d, ff, num_layers, vocab)
        # FP32 projections, as in SmolVLA
        self.state_proj = nn.Linear(state_dim, d)
        self.action_in_proj = nn.Linear(chunk_size * noise_dim, d)
        self.action_out_proj = nn.Linear(d, chunk_size * noise_dim)

    def predict_action_chunk(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        dtype = bb.norm.weight.dtype
        pixels = batch["pixels"]  # (B, cams, 3, H, W)
        B = pixels.shape[0]
        img = bb.patch(pixels.flatten(0, 1).to(dtype))  # (B*cams, d, h, w)
        img = img.flatten(2).mean(-1).view(B, self.num_cameras, -1)
        img = img + bb.vision_mlp(img)
        txt = bb.embed_tokens(batch["tokens"])  # (B, T, d)
        st = self.state_proj(batch["state"].float())[:, None].to(dtype)
        nz = self.action_in_proj(noise.reshape(B, -1).float())[:, None].to(dtype)
        x = torch.cat([img, txt, st, nz], dim=1)
        for layer in bb.layers:
            x = layer(x)
        pooled = bb.norm(x).mean(1).float()
        out = self.action_out_proj(pooled).view(B, self.chunk_size, self.noise_dim)
        return out + noise.float()  # residual on the noise, so different noise -> different actions


class MockVLA(VLAModel):
    def __init__(self, device: str = "cpu", seed: int = 0, camera_keys=("observation.images.image", "observation.images.image2"),
                 state_dim: int = 8, image_size: int = 64, max_tokens: int = 16, vocab: int = 1000):
        torch.manual_seed(seed)
        self.spec = MOCK_SPEC
        self.name = "mock"
        self.revision = None
        self.camera_keys = list(camera_keys)
        self.state_dim = state_dim
        self.image_size = image_size
        self.max_tokens = max_tokens
        self.vocab = vocab
        self.policy = MockVLAPolicy(num_cameras=len(camera_keys), state_dim=state_dim, vocab=vocab).eval().to(device)
        self.device = device
        self.precision = "fp32"
        self.action_dim = self.policy.action_dim
        self.chunk_size = self.policy.chunk_size
        self.noise_dim = self.policy.noise_dim

    def tokenize(self, tasks: List[str]) -> torch.Tensor:
        ids = []
        for t in tasks:
            # crc32 is stable across processes, unlike hash()
            tok = [zlib.crc32(w.encode()) % self.vocab for w in t.split()][:self.max_tokens]
            ids.append(tok + [0] * (self.max_tokens - len(tok)))
        return torch.tensor(ids, dtype=torch.long)

    def prepare(self, frames: List[Frame]) -> dict:
        imgs = frames_to_tensors(frames, self.camera_keys)
        pixels = torch.stack([imgs[k] for k in self.camera_keys], dim=1)
        if pixels.shape[-1] != self.image_size:
            pixels = F.interpolate(pixels.flatten(0, 1), size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False).view(*pixels.shape[:3], self.image_size, self.image_size)
        state = torch.from_numpy(np.stack([f.state[:self.state_dim] for f in frames]))
        return {
            "pixels": pixels.to(self.device),
            "tokens": self.tokenize([f.task for f in frames]).to(self.device),
            "state": state.to(self.device),
        }

    def forward(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        return self.policy.predict_action_chunk(batch, noise)

    def postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., :self.action_dim].float().cpu()

    def backbone(self) -> torch.nn.Module:
        return self.policy.backbone
