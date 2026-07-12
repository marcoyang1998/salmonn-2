#!/usr/bin/env python3
"""Convert a legacy SALMONN-2 Trainer checkpoint into a Hugging Face checkpoint.

The converter operates one safetensors shard at a time, merges Qwen LoRA weights,
rewrites legacy PEFT parameter names, copies tokenizer and remote-code assets, and
deliberately excludes DeepSpeed, optimizer, scheduler, RNG, and Trainer state.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class ShardReader:
    def __init__(self, checkpoint, weight_map):
        self.checkpoint = checkpoint
        self.weight_map = weight_map

    def tensor(self, name):
        shard = self.weight_map[name]
        with safe_open(self.checkpoint / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def merge_and_rename(name, tensor, reader, lora_scale):
    if ".lora_A." in name or ".lora_B." in name:
        return None, None

    if name.endswith(".base_layer.weight"):
        prefix = name[: -len(".base_layer.weight")]
        a_name = prefix + ".lora_A.default.weight"
        b_name = prefix + ".lora_B.default.weight"
        present = (a_name in reader.weight_map, b_name in reader.weight_map)
        if any(present) and not all(present):
            raise ValueError(f"Incomplete LoRA pair for {prefix}: A={present[0]}, B={present[1]}")
        if all(present):
            a = reader.tensor(a_name).float()
            b = reader.tensor(b_name).float()
            tensor = (tensor.float() + (b @ a) * lora_scale).to(tensor.dtype)
        name = prefix + ".weight"

    legacy_prefix = "base_llm.base_model.model."
    if name.startswith(legacy_prefix):
        name = "base_llm." + name[len(legacy_prefix):]
    return name, tensor.contiguous()


def copy_remote_code(repo_root, output):
    for filename in ("configuration_salmonn.py", "audio.py"):
        shutil.copy2(repo_root / "salmonn" / filename, output / filename)
    modeling = (repo_root / "salmonn" / "modeling_salmonn.py").read_text(encoding="utf-8")
    modeling = modeling.replace("from .zipformer.model import", "from .zipformer_model import")
    modeling = modeling.replace("from .zipformer.scaling import", "from .zipformer_scaling import")
    modeling = modeling.replace("from .zipformer.subsampling import", "from .zipformer_subsampling import")
    modeling = modeling.replace("from .zipformer.zipformer_layerwise import", "from .zipformer_layerwise import")
    (output / "modeling_salmonn.py").write_text(modeling, encoding="utf-8")

    flattened = {
        "model.py": "zipformer_model.py",
        "scaling.py": "zipformer_scaling.py",
        "subsampling.py": "zipformer_subsampling.py",
        "zipformer_layerwise.py": "zipformer_layerwise.py",
    }
    for source_name, output_name in flattened.items():
        content = (repo_root / "salmonn" / "zipformer" / source_name).read_text(encoding="utf-8")
        content = content.replace("from .scaling import", "from .zipformer_scaling import")
        (output / output_name).write_text(content, encoding="utf-8")
    (output / "__init__.py").write_text(
        "from .configuration_salmonn import SalmonnConfig\n"
        "from .modeling_salmonn import SalmonnForConditionalGeneration\n",
        encoding="utf-8",
    )


def build_config(qwen_config, model_args):
    qwen_config = dict(qwen_config)
    qwen_config.pop("architectures", None)
    qwen_config.pop("_name_or_path", None)
    return {
        "model_type": "salmonn_2",
        "architectures": ["SalmonnForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_salmonn.SalmonnConfig",
            "AutoModel": "modeling_salmonn.SalmonnForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_salmonn.SalmonnForConditionalGeneration",
        },
        "qwen_config": qwen_config,
        "zipformer_checkpoint": None,
        "freeze_audio_encoder": True,
        "connector_hidden_size": model_args["connector_hid_size"],
        "connector_segment_size": model_args["connector_seg_size"],
        "concatenate_encoder_layers": model_args["concat_encoder_features"],
        "inject_temporal_embedding_nl": model_args.get("inject_temporal_embedding_nl", False),
        "temporal_granularity": model_args.get("temporal_granularity", 2.0),
        "encoder_frame_rate": model_args.get("encoder_frame_rate", 50),
        "dtype": qwen_config.get("dtype", qwen_config.get("torch_dtype", "bfloat16")),
        "transformers_version": qwen_config.get("transformers_version"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Legacy checkpoint-N directory")
    parser.add_argument("--output", required=True, type=Path, help="New, empty output directory")
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Experiment config.json containing model_args; defaults to INPUT/../config.json",
    )
    parser.add_argument("--lora-r", type=int, help="Override the saved LoRA rank")
    parser.add_argument("--lora-alpha", type=float, help="Override the saved LoRA alpha")
    args = parser.parse_args()

    checkpoint = args.input.resolve()
    output = args.output.resolve()
    training_config_path = (args.training_config or checkpoint.parent / "config.json").resolve()
    for required in (checkpoint / "config.json", checkpoint / "model.safetensors.index.json", training_config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training_config = read_json(training_config_path)
    model_args = training_config["model_args"]
    if model_args.get("encoder_type") != "zipformer2" or model_args.get("llm_type") != "Qwen":
        raise ValueError("This converter only supports the released Zipformer2 + Qwen model")
    if model_args.get("use_reasoning_network") or model_args.get("num_pause_steps", 0):
        raise ValueError("Refusing to discard an enabled reasoning network or pause embeddings")
    if model_args.get("encoder_lora"):
        raise ValueError("Encoder LoRA conversion is not implemented")

    lora_r = args.lora_r or model_args.get("lora_rank")
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else model_args.get("lora_alpha")
    if not model_args.get("lora") or not lora_r or lora_alpha is None:
        raise ValueError("The saved model_args do not describe a LoRA checkpoint")
    lora_scale = float(lora_alpha) / int(lora_r)

    old_index = read_json(checkpoint / "model.safetensors.index.json")
    weight_map = old_index["weight_map"]
    reader = ShardReader(checkpoint, weight_map)
    new_weight_map = {}
    total_size = 0
    merged_pairs = 0

    for shard_name in sorted(set(weight_map.values())):
        destination_tensors = {}
        with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            for old_name in source.keys():
                tensor = source.get_tensor(old_name)
                new_name, tensor = merge_and_rename(old_name, tensor, reader, lora_scale)
                if new_name is None:
                    continue
                if old_name.endswith(".base_layer.weight"):
                    prefix = old_name[: -len(".base_layer.weight")]
                    if prefix + ".lora_A.default.weight" in weight_map:
                        merged_pairs += 1
                if new_name in new_weight_map or new_name in destination_tensors:
                    raise ValueError(f"Parameter-name collision after conversion: {new_name}")
                destination_tensors[new_name] = tensor
                new_weight_map[new_name] = shard_name
                total_size += tensor.numel() * tensor.element_size()
        save_file(destination_tensors, output / shard_name, metadata=metadata or {"format": "pt"})
        print(f"Converted {shard_name}: {len(destination_tensors)} tensors")

    if merged_pairs == 0:
        raise RuntimeError("No LoRA modules were merged; refusing to produce a misleading export")
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, handle, indent=2)
        handle.write("\n")

    qwen_config = read_json(checkpoint / "config.json")
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(build_config(qwen_config, model_args), handle, indent=2)
        handle.write("\n")
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": qwen_config.get("bos_token_id"),
        "eos_token_id": qwen_config.get("eos_token_id"),
        "pad_token_id": qwen_config.get("pad_token_id"),
        "transformers_version": qwen_config.get("transformers_version"),
    }
    generation_config = {key: value for key, value in generation_config.items() if value is not None}
    with (output / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(generation_config, handle, indent=2)
        handle.write("\n")

    for filename in TOKENIZER_FILES:
        source = checkpoint / filename
        if source.is_file():
            shutil.copy2(source, output / filename)
    repo_root = Path(__file__).resolve().parents[1]
    copy_remote_code(repo_root, output)
    shutil.copy2(repo_root / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        "# SALMONN-2 inference checkpoint\n\n"
        "This checkpoint contains merged Qwen LoRA weights, Zipformer2, the audio connector, "
        "tokenizer assets, and pinned custom model code. Load it with `trust_remote_code=True`.\n",
        encoding="utf-8",
    )
    print(f"Merged {merged_pairs} LoRA modules with scale alpha/r={lora_scale:g}")
    print(f"Exported {len(new_weight_map)} tensors ({total_size / 1024**3:.2f} GiB) to {output}")


if __name__ == "__main__":
    main()
