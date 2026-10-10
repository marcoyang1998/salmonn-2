#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

from salmonn import AudioProcessor, SalmonnForConditionalGeneration
from salmonn.training import SalmonnCollator, SalmonnDataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a released SALMONN-2 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    config_data = json.loads(Path(args.config).read_text())
    model_path = config_data.pop("model_name_or_path", None)
    if not model_path:
        raise ValueError("model_name_or_path must point to a released SALMONN-2 checkpoint")

    attention = config_data.pop("attn_implementation", None)
    model = SalmonnForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
    if attention:
        # SalmonnForConditionalGeneration does not declare support for alternative attention
        # backends, so this cannot go through from_pretrained. Apply it to the Qwen3 stack,
        # which is where the attention cost is; SPEAR has its own attention either way.
        model.base_llm.set_attn_implementation(attention)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    freeze_audio_encoder = config_data.pop("freeze_audio_encoder", True)
    freeze_connector = config_data.pop("freeze_connector", False)
    model.config.freeze_audio_encoder = freeze_audio_encoder
    model.audio_encoder.requires_grad_(not freeze_audio_encoder)
    connector_modules = (model.ln_audio, model.concat_proj, model.connector)
    for module in connector_modules:
        if module is not None:
            module.requires_grad_(not freeze_connector)

    peft_values = config_data.pop("lora", None)
    if not peft_values:
        raise ValueError("The fine-tuning config must contain a non-empty lora block")
    model.base_llm = get_peft_model(
        model.base_llm,
        LoraConfig(task_type=TaskType.CAUSAL_LM, **peft_values),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.inject_temporal_embedding_nl:
        model.register_nl_timestamp_tokenizer(tokenizer)

    training_values = config_data.pop("training")
    if config_data:
        raise ValueError(f"Unknown fine-tuning config keys: {sorted(config_data)}")
    resume = training_values.pop("resume_from_checkpoint", None)
    training_values["output_dir"] = args.output_dir
    training_args = TrainingArguments(**training_values)
    # Without this, freezing both the encoder and the connector leaves nothing entering the
    # checkpointed Qwen3 blocks requiring grad, so no graph is built and training dies.
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SalmonnDataset(args.data_path),
        data_collator=SalmonnCollator(tokenizer, AudioProcessor()),
    )
    trainer.train(resume_from_checkpoint=resume)

    # Everything written here keeps the PEFT-nested key names, so `--resume_from_checkpoint`
    # works against any of these directories. Run `scripts/merge_lora.py` to convert one into
    # the released layout for inference.
    final_checkpoint = Path(args.output_dir) / "checkpoint-final"
    trainer.save_model(str(final_checkpoint))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_checkpoint))
        print(
            f"Saved {final_checkpoint}. It uses PEFT-nested key names; convert it for inference with:\n"
            f"  python scripts/merge_lora.py --config {args.config} "
            f"--checkpoint {final_checkpoint} --output {Path(args.output_dir) / 'merged'}"
        )


if __name__ == "__main__":
    main()
