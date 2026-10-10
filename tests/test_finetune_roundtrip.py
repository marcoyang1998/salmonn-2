"""End-to-end check that a fine-tuning run round-trips through scripts/merge_lora.py.

Uses a stub audio encoder and a tiny Qwen3 so the whole thing runs on CPU in a few seconds
without a downloaded checkpoint.
"""
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import salmonn.modeling_salmonn as modeling
from salmonn.configuration_salmonn import SalmonnConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
QWEN = dict(
    model_type="qwen3",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=256,
    tie_word_embeddings=False,
    vision_start_token_id=7,
)
LORA = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


class StubSpear(nn.Module):
    def __init__(self, dim=16, layers=3):
        super().__init__()
        self.encoder_dim, self.num_encoder_layers = dim, layers
        self.encoder_embed = nn.Linear(128, dim)
        self.encoder = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))

    def forward_encoder(self, features, feature_lengths):
        hidden, middle = self.encoder_embed(features), []
        for layer in self.encoder:
            hidden = layer(hidden)
            middle.append(hidden.permute(1, 0, 2))
        return hidden, feature_lengths, middle


@pytest.fixture
def released_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(modeling, "build_spear", StubSpear)
    torch.manual_seed(0)
    config = SalmonnConfig(qwen_config=QWEN, connector_hidden_size=64, connector_segment_size=5)
    model = modeling.SalmonnForConditionalGeneration(config)
    path = tmp_path / "released"
    model.save_pretrained(path)
    return path, set(model.state_dict())


def _batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (2, 12))
    input_ids[:, 3] = QWEN["vision_start_token_id"]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
        "audio_features": torch.randn(2, 40, 128),
        "audio_lengths": torch.tensor([40, 30]),
        "audio_counts": torch.tensor([1, 1]),
    }


def _finetune_config(tmp_path, released, **overrides):
    """The config file scripts/train.py was run with; merge_lora.py reads it back."""
    path = tmp_path / "finetune.json"
    path.write_text(json.dumps({"model_name_or_path": str(released), "lora": {**LORA, **overrides}}))
    return path


def _train_briefly(released):
    """Stand in for scripts/train.py: wrap in PEFT and take a few steps."""
    from peft import LoraConfig, TaskType, get_peft_model

    model = modeling.SalmonnForConditionalGeneration.from_pretrained(released)
    model.base_llm = get_peft_model(model.base_llm, LoraConfig(task_type=TaskType.CAUSAL_LM, **LORA))

    batch = _batch()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        model(**batch, labels=batch["input_ids"]).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, batch


def _run_merge(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["merge_lora.py", *map(str, argv)])
    runpy.run_path(str(SCRIPTS / "merge_lora.py"), run_name="__main__")


def test_merge_lora_restores_released_layout(released_checkpoint, tmp_path, monkeypatch, capsys):
    released, released_keys = released_checkpoint
    checkpoint, merged = tmp_path / "checkpoint-3", tmp_path / "merged"

    model, batch = _train_briefly(released)
    lora_b = model.base_llm.base_model.model.model.layers[0].self_attn.q_proj.lora_B["default"].weight
    assert lora_b.norm() > 0, "adapter never trained, the test would pass vacuously"

    model.eval()
    with torch.no_grad():
        reference = model(**batch).logits.clone()
    model.save_pretrained(checkpoint)

    # A nested checkpoint must not be loadable as-is; that is what merging exists to fix.
    _, info = modeling.SalmonnForConditionalGeneration.from_pretrained(checkpoint, output_loading_info=True)
    assert info["missing_keys"] and info["unexpected_keys"]

    _run_merge(monkeypatch, "--config", _finetune_config(tmp_path, released),
               "--checkpoint", checkpoint, "--output", merged)
    assert "0 missing and 0 unexpected" in capsys.readouterr().out

    merged_model, info = modeling.SalmonnForConditionalGeneration.from_pretrained(
        merged, output_loading_info=True
    )
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert set(merged_model.state_dict()) == released_keys

    merged_model.eval()
    with torch.no_grad():
        after = merged_model(**batch).logits
    torch.testing.assert_close(after, reference, atol=1e-4, rtol=1e-4)


def test_frozen_connector_still_trains_the_adapter_under_gradient_checkpointing(released_checkpoint):
    """With SPEAR and the connector both frozen, nothing entering the checkpointed Qwen3 blocks
    requires grad, so no graph is built and the adapter silently gets no gradient. scripts/train.py
    calls enable_input_require_grads() for this case."""
    from peft import LoraConfig, TaskType, get_peft_model

    released, _ = released_checkpoint

    def build():
        model = modeling.SalmonnForConditionalGeneration.from_pretrained(released)
        model.config.freeze_audio_encoder = True
        model.audio_encoder.requires_grad_(False)
        for module in (model.ln_audio, model.concat_proj, model.connector):
            module.requires_grad_(False)
        model.base_llm = get_peft_model(model.base_llm, LoraConfig(task_type=TaskType.CAUSAL_LM, **LORA))
        model.gradient_checkpointing_enable()
        model.base_llm.config.use_cache = False
        # Qwen3 only checkpoints when self.training, which is what Trainer puts it in;
        # from_pretrained hands back a model in eval mode.
        model.train()
        return model

    batch = _batch()

    # Without the hook this is the failure the fix exists to prevent.
    with pytest.raises(RuntimeError, match="does not require grad"):
        build()(**batch, labels=batch["input_ids"]).loss.backward()

    model = build()
    model.enable_input_require_grads()
    model(**batch, labels=batch["input_ids"]).loss.backward()

    lora = [p for n, p in model.named_parameters() if ".lora_" in n]
    assert lora and all(p.grad is not None for p in lora)
    assert sum(p.grad.abs().sum() for p in lora) > 0, "adapter received an all-zero gradient"
    assert all(p.grad is None for p in model.connector.parameters()), "connector should stay frozen"


def test_merge_lora_needs_hyperparameters(released_checkpoint, tmp_path, monkeypatch):
    released, _ = released_checkpoint
    checkpoint = tmp_path / "checkpoint-3"
    model, _ = _train_briefly(released)
    model.save_pretrained(checkpoint)

    with pytest.raises(SystemExit, match="--r"):
        _run_merge(monkeypatch, "--checkpoint", checkpoint, "--output", tmp_path / "merged")


def test_merge_lora_rescues_a_checkpoint_with_no_config(released_checkpoint, tmp_path, monkeypatch, capsys):
    """Checkpoints written before this tooling existed record nothing, so the values are
    supplied on the command line. This is the recovery path for an existing run."""
    released, released_keys = released_checkpoint
    checkpoint, merged = tmp_path / "checkpoint-10080", tmp_path / "merged"

    model, batch = _train_briefly(released)
    model.eval()
    with torch.no_grad():
        reference = model(**batch).logits.clone()
    model.save_pretrained(checkpoint)
    assert not list(checkpoint.glob("*lora*.json")), "the checkpoint must record nothing"

    # --base_model_path only supplies the remote code, so the merge works without it.
    _run_merge(monkeypatch, "--checkpoint", checkpoint, "--output", merged,
               "--r", LORA["r"], "--lora_alpha", LORA["lora_alpha"])
    assert "WARNING: no base checkpoint given" in capsys.readouterr().out

    recovered, info = modeling.SalmonnForConditionalGeneration.from_pretrained(
        merged, output_loading_info=True
    )
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert set(recovered.state_dict()) == released_keys
    recovered.eval()
    with torch.no_grad():
        torch.testing.assert_close(recovered(**batch).logits, reference, atol=1e-4, rtol=1e-4)


def test_merge_lora_rejects_a_config_from_a_different_run(released_checkpoint, tmp_path, monkeypatch):
    """A stale config would silently apply the wrong scaling, so the rank is cross-checked."""
    released, _ = released_checkpoint
    checkpoint = tmp_path / "checkpoint-3"
    model, _ = _train_briefly(released)
    model.save_pretrained(checkpoint)

    wrong = _finetune_config(tmp_path, released, r=LORA["r"] * 2)
    with pytest.raises(SystemExit, match="does not match this checkpoint"):
        _run_merge(monkeypatch, "--config", wrong, "--checkpoint", checkpoint,
                   "--output", tmp_path / "merged")


def test_merge_lora_rejects_an_already_flat_checkpoint(released_checkpoint, tmp_path, monkeypatch):
    released, _ = released_checkpoint
    with pytest.raises(SystemExit, match="already in the released layout"):
        _run_merge(monkeypatch, "--config", _finetune_config(tmp_path, released),
                   "--checkpoint", released, "--output", tmp_path / "out")
