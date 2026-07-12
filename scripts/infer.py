#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from salmonn import AudioProcessor
from salmonn.audio import pad_audio_features
from salmonn.text import clean_decoded_response


def generate(model, tokenizer, processor, audio_paths, prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": "<audio>" * len(audio_paths) + prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text.replace("<audio>", "<|vision_start|><|vision_end|>")
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    features, lengths = pad_audio_features([processor(path) for path in audio_paths])
    output = model.generate(
        **tokens,
        audio_features=features.to(model.device),
        audio_lengths=lengths.to(model.device),
        audio_counts=torch.tensor([len(audio_paths)], device=model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return clean_decoded_response(tokenizer.decode(output[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio", action="append", required=True, help="Repeat for multiple audio files")
    parser.add_argument("--prompt", default="Please describe the audio.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    if model.config.inject_temporal_embedding_nl:
        model.register_nl_timestamp_tokenizer(tokenizer)
    print(generate(model, tokenizer, AudioProcessor(), args.audio, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
