#!/usr/bin/env python3
"""Merge a LoRA fine-tuning checkpoint into the released SALMONN-2 checkpoint layout.

`scripts/train.py` wraps `base_llm` in a PEFT adapter, so the checkpoints it writes carry
PEFT-nested key names (``base_llm.base_model.model....`` with the original weights under
``.base_layer``). Those names are what training and `--resume_from_checkpoint` need, but
`SalmonnForConditionalGeneration` always rebuilds a plain Qwen3, so they cannot be loaded
back for inference. This script folds the adapter into the base weights and rewrites the
result with the released key names.

The rank and alpha are read from the fine-tuning config used for the run, since `lora_alpha`
cannot be recovered from the weights.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from salmonn import SalmonnForConditionalGeneration

PEFT_PREFIX = "base_llm.base_model.model."
# Everything the released checkpoint ships so that `trust_remote_code=True` keeps working.
SIDECAR_PATTERNS = ("*.py", "*.json", "*.txt", "chat_template.jinja")
SIDECAR_SKIP = {SAFE_WEIGHTS_INDEX_NAME, "trainer_state.json"}


def flatten_peft_state_dict(state_dict, r, lora_alpha, use_rslora=False):
    """PEFT-nested state dict -> released flat layout, folding in the LoRA delta."""
    scaling = lora_alpha / (r**0.5 if use_rslora else r)
    merged, factors = {}, {}
    for key, value in state_dict.items():
        if ".lora_A" in key or ".lora_B" in key:
            stem, which = key.split(".lora_")
            factors.setdefault(stem, {})[which[0]] = value
            continue
        if "lora_magnitude_vector" in key or "lora_embedding" in key:
            raise NotImplementedError(f"{key}: DoRA and embedding adapters are not supported")
        if ".modules_to_save." in key or ".original_module." in key:
            raise NotImplementedError(f"{key}: lora.modules_to_save is not supported")
        flat = key.replace(PEFT_PREFIX, "base_llm.", 1) if key.startswith(PEFT_PREFIX) else key
        merged[flat.replace(".base_layer.weight", ".weight")] = value
    for stem, factor in sorted(factors.items()):
        if set(factor) != {"A", "B"}:
            raise ValueError(f"{stem}: expected both lora_A and lora_B, found {sorted(factor)}")
        target = stem.replace(PEFT_PREFIX, "base_llm.", 1) + ".weight"
        if target not in merged:
            raise ValueError(f"{stem}: no base weight {target} to merge into")
        delta = (factor["B"] @ factor["A"]) * scaling
        merged[target] = (merged[target].float() + delta.float()).to(merged[target].dtype)
    return merged, len(factors)


def load_checkpoint_state(checkpoint):
    shards = sorted(glob.glob(str(Path(checkpoint) / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards in {checkpoint}")
    state = {}
    for shard in shards:
        state.update(load_file(shard))
    return state, len(shards)


def resolve_adapter(args):
    """Adapter hyper-parameters, from the fine-tuning config unless given on the command line."""
    saved, lora = {}, {}
    if args.config:
        saved = json.loads(Path(args.config).read_text())
        lora = saved.get("lora") or {}
    r = args.r or lora.get("r")
    lora_alpha = args.lora_alpha or lora.get("lora_alpha")
    missing = [n for n, v in (("--r", r), ("--lora_alpha", lora_alpha)) if not v]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)}. Pass --config with the fine-tuning config used for the run, "
            "or give the values on the command line."
        )
    base = args.base_model_path or saved.get("model_name_or_path")
    return int(r), float(lora_alpha), bool(lora.get("use_rslora", False)), base


def check_rank(state_dict, r):
    """The rank is visible in the lora_A shapes, so a mismatched config can be caught early."""
    for key, value in state_dict.items():
        if ".lora_A" in key:
            if value.shape[0] != r:
                raise SystemExit(
                    f"Configured r={r} but {key} has rank {value.shape[0]}. The fine-tuning config "
                    "does not match this checkpoint; merging with the wrong scaling would silently "
                    "produce a valid-looking but incorrect model."
                )
            return
    raise SystemExit("No lora_A tensors found; this checkpoint has no adapter to merge.")


def copy_sidecars(base_model_path, output_dir, checkpoint):
    """Model/processor metadata: from the base checkpoint first, then the run's own overrides.

    A checkpoint-N carries config.json and the tokenizer, but not the remote code or the
    processor config, which only exist in the released checkpoint.
    """
    sources = [checkpoint] if base_model_path is None else [base_model_path, checkpoint]
    copied = set()
    for source in sources:
        for pattern in SIDECAR_PATTERNS:
            for path in sorted(Path(source).glob(pattern)):
                if path.name in SIDECAR_SKIP or path.name.endswith(".safetensors"):
                    continue
                shutil.copy(path, Path(output_dir) / path.name)
                copied.add(path.name)
    return sorted(copied)


def write_sharded(state_dict, output_dir, max_shard_size):
    split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors"),
        max_shard_size=max_shard_size,
    )
    for filename, keys in split.filename_to_tensors.items():
        shard = {key: state_dict[key].contiguous() for key in keys}
        save_file(shard, Path(output_dir) / filename, metadata={"format": "pt"})
    index_path = Path(output_dir) / SAFE_WEIGHTS_INDEX_NAME
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        index_path.write_text(json.dumps(index, indent=2))
    elif index_path.exists():
        index_path.unlink()
    return sorted(split.filename_to_tensors)


def verify(output_dir):
    _, info = SalmonnForConditionalGeneration.from_pretrained(
        output_dir, torch_dtype="auto", output_loading_info=True
    )
    missing, unexpected = info["missing_keys"], info["unexpected_keys"]
    if missing or unexpected:
        raise SystemExit(
            f"Merged checkpoint does not round-trip: {len(missing)} missing, {len(unexpected)} unexpected keys.\n"
            f"  missing[:5]    = {sorted(missing)[:5]}\n"
            f"  unexpected[:5] = {sorted(unexpected)[:5]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="A checkpoint-N directory written by scripts/train.py")
    parser.add_argument("--output", required=True, help="Destination directory for the merged checkpoint")
    parser.add_argument("--config", help="The fine-tuning config used for the run, e.g. configs/finetune.json")
    parser.add_argument("--base_model_path",
                        help="Released checkpoint to copy the remote code and processor config from. "
                             "Only needed for trust_remote_code loading; a checkpoint-N already "
                             "carries config.json and the tokenizer.")
    parser.add_argument("--r", type=int, help="LoRA rank; overrides the fine-tuning config")
    parser.add_argument("--lora_alpha", type=float, help="LoRA alpha; overrides the fine-tuning config")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--skip_verify", action="store_true", help="Do not reload the result to check it")
    args = parser.parse_args()

    r, lora_alpha, use_rslora, base_model_path = resolve_adapter(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    state, shard_count = load_checkpoint_state(args.checkpoint)
    print(f"[merge] read {len(state)} tensors from {shard_count} shard(s) in {args.checkpoint}")
    if not any(key.startswith(PEFT_PREFIX) for key in state):
        raise SystemExit(f"{args.checkpoint} is already in the released layout; nothing to merge.")
    check_rank(state, r)

    merged, adapters = flatten_peft_state_dict(state, r, lora_alpha, use_rslora)
    print(f"[merge] folded {adapters} LoRA adapters at scaling={lora_alpha / (r**0.5 if use_rslora else r):g} "
          f"(r={r}, lora_alpha={lora_alpha:g}); {len(state)} -> {len(merged)} tensors")

    shards = write_sharded(merged, output_dir, args.max_shard_size)
    print(f"[merge] wrote {len(shards)} shard(s): {shards[0]}{' ...' if len(shards) > 1 else ''}")
    print(f"[merge] copied metadata: {copy_sidecars(base_model_path, output_dir, args.checkpoint)}")
    if base_model_path is None:
        print(
            "[merge] WARNING: no base checkpoint given, so the remote code and processor config "
            "were not copied. The result loads with the installed salmonn package but not with "
            "trust_remote_code=True, which scripts/infer.py uses. Re-run with --base_model_path "
            "to make it self-contained."
        )

    del state, merged
    if args.skip_verify:
        print(f"[merge] done (verification skipped) -> {output_dir}")
        return
    verify(output_dir)
    print(f"[merge] verified: reloads with 0 missing and 0 unexpected keys -> {output_dir}")


if __name__ == "__main__":
    main()
