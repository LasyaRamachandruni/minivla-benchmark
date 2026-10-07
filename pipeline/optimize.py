"""Optimized variants of a VLA policy.

Every variant is a deep copy of the FP32 baseline policy wrapped by the same `VLAModel`, so it
goes through the same preprocessing, the same noise and the same timing code.

* `fp16` / `bf16` (GPU): the VLM + action expert in half precision; the small state / action /
  time projections stay FP32 (the mixed layout LeRobot itself uses when it loads the VLM in bf16).
* `int8_dynamic` (CPU): `torch.ao.quantization.quantize_dynamic` on `nn.Linear` layers - int8
  weights, activations quantized on the fly. The attention q/k/v/o projections are left in FP32
  because SmolVLA's forward pass reads `layer.self_attn.q_proj.weight.dtype`, which a dynamically
  quantized Linear does not have.
* `bnb_int8` (GPU, optional): bitsandbytes `Linear8bitLt` for the same set of Linear layers, with
  the rest of the backbone in fp16.
* `pruned` (any device): *structured* pruning of MLP hidden channels. Channels are physically
  removed (the Linear layers get smaller), so parameter count, model size and FLOPs drop. No
  fine-tuning is done afterwards, so accuracy loss is expected and is what the suite measures.
"""

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from models.vla import VLAModel, count_parameters

CPU_DEFAULT_VARIANTS = ["fp32", "int8_dynamic", "pruned", "pruned+int8_dynamic"]
GPU_DEFAULT_VARIANTS = ["fp32", "fp16", "bf16", "pruned", "pruned+fp16"]
ALL_VARIANTS = ["fp32", "fp16", "bf16", "int8_dynamic", "bnb_int8", "pruned",
                "pruned+int8_dynamic", "pruned+fp16", "pruned+bf16"]


class VariantUnavailable(RuntimeError):
    """The variant cannot run on this device / install (recorded as skipped, not as a failure)."""


def _copy_model(model: VLAModel) -> VLAModel:
    return model.with_policy(copy.deepcopy(model.policy), model.device, model.precision)


# -- precision -------------------------------------------------------------------------

def cast_backbone(model: VLAModel, dtype: torch.dtype, label: str) -> VLAModel:
    if dtype == torch.bfloat16 and model.device.startswith("cuda") and not _cuda_bf16_native():
        raise VariantUnavailable("GPU has no native bf16 support (e.g. T4); bf16 would be emulated")
    new = _copy_model(model)
    new.backbone().to(dtype)
    new.precision = label
    return new


def _cuda_bf16_native() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch has no including_emulation argument
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


# -- int8 ----------------------------------------------------------------------------

def int8_target_linears(module: nn.Module, skip_suffixes: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Names of nn.Linear layers to quantize, and of the ones skipped."""
    targets, skipped = [], []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            (skipped if name.rsplit(".", 1)[-1] in skip_suffixes else targets).append(name)
    return targets, skipped


def _ensure_quant_engine():
    engines = torch.backends.quantized.supported_engines
    if torch.backends.quantized.engine in (None, "none"):
        for e in ("x86", "fbgemm", "qnnpack"):
            if e in engines:
                torch.backends.quantized.engine = e
                break


def quantize_int8_dynamic(model: VLAModel) -> Tuple[VLAModel, dict]:
    if model.device != "cpu":
        raise VariantUnavailable("PyTorch dynamic int8 quantization runs on CPU only")
    _ensure_quant_engine()
    new = _copy_model(model)
    new.policy.float()
    targets, skipped = int8_target_linears(new.policy, model.int8_skip_suffixes)
    frac = _weight_fraction(new.policy, targets)
    new.policy = torch.ao.quantization.quantize_dynamic(new.policy, set(targets), dtype=torch.qint8)
    new.precision = f"{model.precision}+int8_dynamic" if model.precision != "fp32" else "int8_dynamic"
    return new, {"int8_linear_layers": len(targets), "fp32_linear_layers_kept": len(skipped),
                 "int8_fraction_of_all_params": frac,
                 "kept_suffixes": list(model.int8_skip_suffixes), "engine": torch.backends.quantized.engine}


def _weight_fraction(module: nn.Module, names: Iterable[str]) -> float:
    total = sum(p.numel() for p in module.parameters())
    part = sum(module.get_submodule(n).weight.numel() for n in names)
    return part / total if total else 0.0


def quantize_bnb_int8(model: VLAModel, threshold: float = 6.0) -> Tuple[VLAModel, dict]:
    if not model.device.startswith("cuda"):
        raise VariantUnavailable("bitsandbytes int8 needs a CUDA GPU")
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise VariantUnavailable("bitsandbytes is not installed") from e

    new = cast_backbone(model, torch.float16, "fp16")
    backbone = new.backbone()
    targets, skipped = int8_target_linears(backbone, model.int8_skip_suffixes)
    frac = _weight_fraction(new.policy, [_prefix(new.policy, backbone) + t for t in targets])
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        lin = getattr(parent, child)
        q = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                has_fp16_weights=False, threshold=threshold)
        q.weight = bnb.nn.Int8Params(lin.weight.data.half().cpu(), requires_grad=False, has_fp16_weights=False)
        if lin.bias is not None:
            q.bias = nn.Parameter(lin.bias.data.half().cpu(), requires_grad=False)
        setattr(parent, child, q.to(model.device))  # moving to CUDA performs the int8 quantization
    new.precision = "fp16+bnb_int8"
    return new, {"int8_linear_layers": len(targets), "fp16_linear_layers_kept": len(skipped),
                 "int8_fraction_of_all_params": frac,
                 "kept_suffixes": list(model.int8_skip_suffixes), "threshold": threshold}


def _prefix(root: nn.Module, sub: nn.Module) -> str:
    for name, m in root.named_modules():
        if m is sub:
            return name + "." if name else ""
    raise ValueError("submodule not found")


# -- structured pruning ---------------------------------------------------------------

def _slice_linear(lin: nn.Linear, keep: torch.Tensor, dim: int) -> nn.Linear:
    """New Linear keeping `keep` output rows (dim=0) or input columns (dim=1)."""
    keep = keep.to(lin.weight.device)
    w = lin.weight.data.index_select(dim, keep)
    out_f, in_f = w.shape
    new = nn.Linear(in_f, out_f, bias=lin.bias is not None, device=w.device, dtype=w.dtype)
    new.weight.data.copy_(w)
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data.index_select(0, keep) if dim == 0 else lin.bias.data)
    new.weight.requires_grad_(lin.weight.requires_grad)
    return new


def _n_keep(width: int, amount: float, multiple: int) -> int:
    keep = int(width * (1.0 - amount))
    keep = max(multiple, (keep // multiple) * multiple)
    return min(width, keep)


def prune_mlp_channels(module: nn.Module, amount: float, multiple: int = 8) -> dict:
    """Remove the lowest-importance `amount` of hidden channels from every MLP, in place.

    Gated MLPs (`gate_proj`, `up_proj`, `down_proj`, as in SmolVLM's Llama text model and the
    action expert) and plain MLPs (`fc1`, `fc2`, as in the SigLIP vision encoder) are supported.
    Channel importance is the product of the L2 norms of the weights entering and leaving the
    channel. Kept widths are rounded down to a multiple of `multiple` (tensor-core friendly).
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be in [0, 1)")
    before = count_parameters(module)
    pruned = []
    for name, m in list(module.named_modules()):
        lins = {a: getattr(m, a, None) for a in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")}
        if all(isinstance(lins[a], nn.Linear) for a in ("gate_proj", "up_proj", "down_proj")):
            width = m.up_proj.out_features
            with torch.no_grad():
                score = (m.gate_proj.weight.float().norm(dim=1) * m.up_proj.weight.float().norm(dim=1)
                         * m.down_proj.weight.float().norm(dim=0))
            keep = torch.topk(score, _n_keep(width, amount, multiple)).indices.sort().values
            m.gate_proj = _slice_linear(m.gate_proj, keep, 0)
            m.up_proj = _slice_linear(m.up_proj, keep, 0)
            m.down_proj = _slice_linear(m.down_proj, keep, 1)
        elif isinstance(lins["fc1"], nn.Linear) and isinstance(lins["fc2"], nn.Linear):
            width = m.fc1.out_features
            with torch.no_grad():
                score = m.fc1.weight.float().norm(dim=1) * m.fc2.weight.float().norm(dim=0)
            keep = torch.topk(score, _n_keep(width, amount, multiple)).indices.sort().values
            m.fc1 = _slice_linear(m.fc1, keep, 0)
            m.fc2 = _slice_linear(m.fc2, keep, 1)
        else:
            continue
        if hasattr(m, "intermediate_size"):
            m.intermediate_size = len(keep)
        pruned.append({"module": name, "width_before": width, "width_after": int(len(keep))})
    after = count_parameters(module)
    return {"amount": amount, "mlps_pruned": len(pruned), "params_before": before, "params_after": after,
            "param_reduction_pct": (1 - after / before) * 100 if before else 0.0, "layers": pruned}


def prune_structured(model: VLAModel, amount: float) -> Tuple[VLAModel, dict]:
    new = _copy_model(model)
    info = prune_mlp_channels(new.policy, amount)
    new.precision = f"{model.precision}+pruned{int(round(amount * 100))}"
    return new, info


# -- variant factory ------------------------------------------------------------------

def build_variant(base: VLAModel, variant: str, prune_amount: float = 0.2) -> Tuple[VLAModel, dict]:
    """Build a named variant from the FP32 baseline. Raises VariantUnavailable when not applicable."""
    if variant == "fp32":
        return base, {}
    info: Dict[str, dict] = {}
    model = base
    steps = variant.split("+")
    if steps[0] == "pruned":
        model, info["pruning"] = prune_structured(model, prune_amount)
        steps = steps[1:]
    for step in steps:
        if step == "fp16":
            if not model.device.startswith("cuda"):
                raise VariantUnavailable("fp16 variants are only benchmarked on GPU")
            model = cast_backbone(model, torch.float16, model.precision.replace("fp32", "fp16"))
        elif step == "bf16":
            if not model.device.startswith("cuda"):
                raise VariantUnavailable("bf16 variants are only benchmarked on GPU")
            model = cast_backbone(model, torch.bfloat16, model.precision.replace("fp32", "bf16"))
        elif step == "int8_dynamic":
            model, info["quantization"] = quantize_int8_dynamic(model)
        elif step == "bnb_int8":
            model, info["quantization"] = quantize_bnb_int8(model)
        else:
            raise ValueError(f"unknown variant step {step!r} in {variant!r}; choose from {ALL_VARIANTS}")
    return model, info


def default_variants(device: str) -> List[str]:
    return list(GPU_DEFAULT_VARIANTS if device.startswith("cuda") else CPU_DEFAULT_VARIANTS)


def parse_variants(spec: Optional[str], device: str) -> List[str]:
    if not spec:
        return default_variants(device)
    variants = [v.strip() for v in spec.split(",") if v.strip()]
    if "fp32" not in variants:
        variants.insert(0, "fp32")  # every comparison needs the same-device FP32 baseline
    return variants
