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
    model_kwargs = {"torch_dtype": "auto"}
    if attention:
        model_kwargs["attn_implementation"] = attention
    model = SalmonnForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SalmonnDataset(args.data_path),
        data_collator=SalmonnCollator(tokenizer, AudioProcessor()),
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
