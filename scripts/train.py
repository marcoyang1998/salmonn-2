#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from salmonn import AudioProcessor, SalmonnConfig, SalmonnForConditionalGeneration
from salmonn.training import SalmonnCollator, SalmonnDataset


def main():
    parser = argparse.ArgumentParser(description="Train SALMONN-2 with Zipformer2 and Qwen3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args, overrides = parser.parse_known_args()

    config_data = json.loads(Path(args.config).read_text())
    base_llm = config_data.pop("base_llm_name_or_path")
    model_path = config_data.pop("model_name_or_path", None)
    attention = config_data.pop("attn_implementation", None)
    if model_path:
        model = SalmonnForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        qwen_config = AutoConfig.from_pretrained(base_llm)
        config = SalmonnConfig(qwen_config=qwen_config.to_dict(), **config_data.pop("model"))
        model = SalmonnForConditionalGeneration(config)
        model.base_llm = AutoModelForCausalLM.from_pretrained(
            base_llm, torch_dtype="auto", attn_implementation=attention
        )
        tokenizer = AutoTokenizer.from_pretrained(base_llm)
    peft_values = config_data.pop("lora", None)
    if peft_values:
        model.base_llm = get_peft_model(
            model.base_llm,
            LoraConfig(task_type=TaskType.CAUSAL_LM, **peft_values),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.inject_temporal_embedding_nl:
        model.register_nl_timestamp_tokenizer(tokenizer)

    training_values = config_data.pop("training")
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
