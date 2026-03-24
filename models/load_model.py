"""Model loading utilities for VLA benchmark pipeline.

Supports loading vision-language models from HuggingFace Transformers,
ONNX Runtime models, and a lightweight mock model for testing.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

SUPPORTED_MODELS = {
    "smolvlm": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "mobilevlm": "mtgv/MobileVLM_V2-1.7B",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "mock": "mock",
}

DEFAULT_IMAGE_SIZE = 224


@dataclass
class ModelInfo:
    name: str
    backend: str  # "pytorch", "onnx", "mock"
    size_mb: float
    device: str
    model: object
    processor: object = None
    tokenizer: object = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_size_mb(model) -> float:
    """Calculate model size in MB from parameters."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def get_onnx_model_size_mb(path: str) -> float:
    """Get ONNX model file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


class MockVLMModel(torch.nn.Module):
    """Lightweight mock VLA model for pipeline testing without large downloads.

    Architecture: vision encoder (CNN) -> projection -> language decoder (Transformer)
    This mimics the structure of real VLM models at a fraction of the size.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, image_size: int = 224):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Vision encoder (simple CNN)
        self.vision_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(7),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, embed_dim),
        )

        # Text embedding
        self.text_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # Transformer decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action head: predicts 7-DoF action (x, y, z, roll, pitch, yaw, gripper)
        self.action_head = torch.nn.Linear(embed_dim, 7)

        # Language head
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> dict:
        vision_features = self.vision_encoder(pixel_values).unsqueeze(1)  # (B, 1, D)

        text_features = self.text_embedding(input_ids)  # (B, T, D)

        # Cross-attend text to vision
        decoded = self.decoder(text_features, vision_features)  # (B, T, D)

        # Pool and predict action
        pooled = decoded.mean(dim=1)  # (B, D)
        actions = self.action_head(pooled)  # (B, 7)

        logits = self.lm_head(decoded)  # (B, T, V)

        return {"actions": actions, "logits": logits}


class MockProcessor:
    """Mock processor that mimics HuggingFace processor interface."""

    def __init__(self, image_size: int = 224, vocab_size: int = 1000, max_length: int = 32):
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        result = {}
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            batch = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = img.resize((self.image_size, self.image_size))
                    arr = np.array(img).astype(np.float32) / 255.0
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.shape[-1] == 4:
                        arr = arr[:, :, :3]
                    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
                else:
                    arr = np.random.randn(3, self.image_size, self.image_size).astype(np.float32)
                batch.append(arr)
            result["pixel_values"] = torch.tensor(np.stack(batch))

        if text is not None:
            if isinstance(text, str):
                text = [text]
            ids = []
            for t in text:
                # Simple hash-based tokenization
                tokens = [hash(w) % self.vocab_size for w in t.split()]
                tokens = tokens[:self.max_length]
                tokens += [0] * (self.max_length - len(tokens))
                ids.append(tokens)
            result["input_ids"] = torch.tensor(ids, dtype=torch.long)

        return result

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token_{t}" for t in token_ids if t != 0)


def load_model(model_name: str, device: Optional[str] = None) -> ModelInfo:
    """Load a VLA model by name.

    Args:
        model_name: One of 'smolvlm', 'mobilevlm', 'llava', 'mock', or a HuggingFace model ID.
        device: Target device. Auto-detected if None.

    Returns:
        ModelInfo with loaded model and processor.
    """
    if device is None:
        device = get_device()

    resolved = SUPPORTED_MODELS.get(model_name, model_name)

    if resolved == "mock":
        return _load_mock_model(device)

    return _load_hf_model(resolved, model_name, device)


def _load_mock_model(device: str) -> ModelInfo:
    model = MockVLMModel()
    model = model.to(device)
    model.eval()
    processor = MockProcessor()
    size_mb = get_model_size_mb(model)
    return ModelInfo(
        name="mock",
        backend="pytorch",
        size_mb=size_mb,
        device=device,
        model=model,
        processor=processor,
    )


def _load_hf_model(model_id: str, display_name: str, device: str) -> ModelInfo:
    from transformers import AutoProcessor

    # Use the non-deprecated class if available
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print(f"Loading {model_id} from HuggingFace...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Use bfloat16 for efficiency on supported devices, float32 on CPU
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoVLM.from_pretrained(
        model_id,
        dtype=dtype,
        _attn_implementation="eager",
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    size_mb = get_model_size_mb(model)
    print(f"Loaded {display_name}: {size_mb:.1f} MB on {device}")

    return ModelInfo(
        name=display_name,
        backend="pytorch",
        size_mb=size_mb,
        device=device,
        model=model,
        processor=processor,
    )


def load_onnx_model(onnx_path: str) -> ModelInfo:
    """Load an ONNX model for inference."""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    active_provider = session.get_providers()[0]
    device = "cuda" if "CUDA" in active_provider else "cpu"
    size_mb = get_onnx_model_size_mb(onnx_path)

    return ModelInfo(
        name=os.path.basename(onnx_path),
        backend="onnx",
        size_mb=size_mb,
        device=device,
        model=session,
        processor=MockProcessor(),
    )


def create_sample_input(processor, device: str = "cpu",
                        image: Optional[Image.Image] = None,
                        prompt: str = "pick up the red block") -> dict:
    """Create a sample input dict suitable for model forward pass."""
    if image is None:
        image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

    # Check if processor supports chat templates (real HF VLMs like SmolVLM)
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")
    else:
        # Mock processor or simple processors
        inputs = processor(images=image, text=prompt)

    return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
