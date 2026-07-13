#!/usr/bin/env python3
"""Run single- or multi-audio inference from a converted SALMONN-2 checkpoint."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from salmonn.audio import AudioProcessor, pad_audio_features
from salmonn.text import clean_decoded_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--audio", action="append", required=True, help="Repeat for multiple audio files")
    parser.add_argument("--prompt", default="Please describe the audio.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    if model.config.inject_temporal_embedding_nl:
        model.register_nl_timestamp_tokenizer(tokenizer)

    messages = [
        {
            "role": "user",
            "content": "<audio>" * len(args.audio) + args.prompt,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).replace("<audio>", "<|vision_start|><|vision_end|>")
    text_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    audio_features, audio_lengths = pad_audio_features([AudioProcessor()(path) for path in args.audio])
    device = next(model.parameters()).device

    with torch.inference_mode():
        output = model.generate(
            **text_inputs.to(device),
            audio_features=audio_features.to(device),
            audio_lengths=audio_lengths.to(device),
            audio_counts=torch.tensor([len(args.audio)], device=device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(clean_decoded_response(tokenizer.decode(output[0], skip_special_tokens=True)))


if __name__ == "__main__":
    main()
