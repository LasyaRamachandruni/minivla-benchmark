"""Integration tests against LeRobot's real SmolVLA code, using a tiny randomly initialised model
built offline in a temp dir (no Hub downloads). Skipped when lerobot is not installed.

This checks the wrapper's use of the LeRobot API (config / policy / processor loading, the
predict_action_chunk noise argument, un-normalization) and that every optimization runs on the
actual SmolVLA module structure. It says nothing about the real checkpoint's accuracy or speed.
"""

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from models.frames import synthetic_frames  # noqa: E402

pytestmark = pytest.mark.integration


def _build_tiny_vlm(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, SmolVLMConfig, SmolVLMImageProcessor, SmolVLMProcessor

    words = ["[PAD]", "[UNK]"] + "pick up the black bowl open top drawer and put it on plate".split()
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(dict.fromkeys(words))}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    special = {"fake_image_token": "<fake_token_around_image>", "image_token": "<image>",
               "global_image_token": "<global-img>", "end_of_utterance_token": "<end_of_utterance>"}
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                     extra_special_tokens=special)
    hf_tok.add_special_tokens({"additional_special_tokens": list(special.values())})
    img_proc = SmolVLMImageProcessor(do_image_splitting=False, max_image_size={"longest_edge": 64},
                                     size={"longest_edge": 64})
    try:
        from transformers import SmolVLMVideoProcessor

        proc = SmolVLMProcessor(image_processor=img_proc, tokenizer=hf_tok, video_processor=SmolVLMVideoProcessor(),
                                image_seq_len=4)
    except (ImportError, TypeError):
        proc = SmolVLMProcessor(image_processor=img_proc, tokenizer=hf_tok, image_seq_len=4)
    proc.save_pretrained(path)
    SmolVLMConfig(
        text_config=dict(model_type="llama", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=len(hf_tok),
                         max_position_embeddings=512),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=64, patch_size=16),
        scale_factor=2,
        image_token_id=hf_tok.convert_tokens_to_ids("<image>"),
    ).save_pretrained(path)


@pytest.fixture(scope="module")
def tiny_policy_dir(tmp_path_factory):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    root = tmp_path_factory.mktemp("tiny_smolvla")
    vlm, out = root / "vlm", root / "policy"
    _build_tiny_vlm(str(vlm))
    torch.manual_seed(0)
    cfg = SmolVLAConfig(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        vlm_model_name=str(vlm), load_vlm_weights=False, resize_imgs_with_padding=(64, 64),
        chunk_size=10, n_action_steps=10, num_steps=3, device="cpu", pad_language_to="longest",
    )
    policy = SmolVLAPolicy(cfg)
    stats = {"observation.state": {"mean": torch.zeros(8), "std": torch.ones(8)},
             "action": {"mean": torch.full((7,), 0.5), "std": torch.full((7,), 2.0)}}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    for obj in (policy, pre, post):
        obj.save_pretrained(str(out))
    return str(out)


@pytest.fixture(scope="module")
def tiny(tiny_policy_dir):
    from models.load_model import load_model

    return load_model(tiny_policy_dir, device="cpu", gt_comparable=True)


@pytest.fixture(scope="module")
def frames():
    return synthetic_frames(4, image_size=96)


def test_wrapper_reads_features_from_checkpoint(tiny):
    assert tiny.camera_keys == ["observation.images.image", "observation.images.image2"]
    assert (tiny.state_dim, tiny.action_dim, tiny.chunk_size, tiny.noise_dim) == (8, 7, 10, 32)
    assert all(p.dtype == torch.float32 for p in tiny.policy.parameters())


def test_predict_is_deterministic_given_noise_and_unnormalized(tiny, frames):
    a = tiny.predict(frames, seeds=range(4), batch_size=4)
    b = tiny.predict(frames, seeds=range(4), batch_size=1)
    assert a.shape == (4, 10, 7)
    np.testing.assert_allclose(a, b, atol=1e-4)
    # manual path: preprocess -> predict_action_chunk(noise) -> un-normalize (x * std + mean)
    batch = tiny.prepare(frames[:1])
    raw = tiny.policy.predict_action_chunk(dict(batch), noise=tiny.make_noise([0]))
    np.testing.assert_allclose(a[:1], (raw * 2.0 + 0.5).numpy(), atol=1e-4)


def test_all_cpu_variants_run_on_real_smolvla(tiny, frames):
    from pipeline.optimize import build_variant

    ref = tiny.predict(frames, range(4), batch_size=4)
    for v in ("int8_dynamic", "pruned", "pruned+int8_dynamic"):
        model, info = build_variant(tiny, v, prune_amount=0.25)
        out = model.predict(frames, range(4), batch_size=4)
        assert out.shape == ref.shape and np.isfinite(out).all(), v
        assert model.size_mb() < tiny.size_mb(), v
    assert info["pruning"]["mlps_pruned"] > 0 and info["quantization"]["int8_linear_layers"] > 0


def test_half_precision_backbone_runs(tiny, frames):
    from pipeline.optimize import cast_backbone

    ref = tiny.predict(frames, range(4), batch_size=4)
    out = cast_backbone(tiny, torch.bfloat16, "bf16").predict(frames, range(4), batch_size=4)
    assert np.isfinite(out).all() and np.abs(out - ref).max() < 0.5


def test_policy_free_wrapper_can_preprocess(tiny_policy_dir, frames):
    from models.load_model import load_model

    light = load_model(tiny_policy_dir, device="cpu", load_policy=False)
    assert light.policy is None
    batch = light.prepare(frames)
    assert batch["observation.state"].shape == (4, 8)
