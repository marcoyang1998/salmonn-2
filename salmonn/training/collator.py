from dataclasses import dataclass

import torch

from salmonn.audio import AudioProcessor, pad_audio_features


@dataclass
class SalmonnCollator:
    tokenizer: object
    audio_processor: AudioProcessor

    def _render(self, messages, generation_prompt=False):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        return text.replace("<audio>", "<|vision_start|><|vision_end|>")

    def __call__(self, samples):
        texts, prompt_lengths, audio_counts, features = [], [], [], []
        for sample in samples:
            texts.append(self._render(sample["messages"]))
            if sample["messages"][-1].get("role") != "assistant":
                raise ValueError("Training conversations must end with an assistant response")
            prompt = self._render(sample["messages"][:-1], generation_prompt=True)
            prompt_lengths.append(len(self.tokenizer(prompt, add_special_tokens=False).input_ids))
            audio_counts.append(len(sample["audios"]))
            features.extend(self.audio_processor(path) for path in sample["audios"])

        tokenized = self.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        labels = tokenized.input_ids.clone()
        labels[tokenized.attention_mask == 0] = -100
        for row, length in enumerate(prompt_lengths):
            labels[row, :length] = -100
        audio_features, audio_lengths = pad_audio_features(features)
        return {
            **tokenized,
            "labels": labels,
            "audio_features": audio_features,
            "audio_lengths": audio_lengths,
            "audio_counts": torch.tensor(audio_counts, dtype=torch.long),
        }
