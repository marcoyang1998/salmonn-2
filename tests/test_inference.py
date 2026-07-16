import array
import math
import wave

import torch

from salmonn import AudioProcessor
from scripts.infer import generate


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "<audio>Describe the audio."}]
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def __call__(self, text, return_tensors, add_special_tokens):
        assert text == "<|vision_start|><|vision_end|>Describe the audio."
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return TokenBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids.tolist() == [7]
        assert skip_special_tokens is True
        return "<think>\n\n</think>\n\nA synthetic tone."


class StubModel:
    device = torch.device("cpu")

    def generate(self, **inputs):
        assert inputs["input_ids"].device.type == "cpu"
        assert inputs["audio_features"].device.type == "cpu"
        assert inputs["audio_features"].shape[0] == 1
        assert inputs["audio_features"].shape[2] == 128
        assert inputs["audio_lengths"].tolist() == [inputs["audio_features"].shape[1]]
        assert inputs["audio_counts"].tolist() == [1]
        assert inputs["max_new_tokens"] == 4
        assert inputs["do_sample"] is False
        return torch.tensor([[7]])


def write_tone(path, sample_rate=16000, duration=0.25, frequency=440):
    samples = array.array(
        "h",
        (
            round(10000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(round(sample_rate * duration))
        ),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(samples.itemsize)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_single_audio_cpu_inference(tmp_path):
    audio_path = tmp_path / "tone.wav"
    write_tone(audio_path)

    response = generate(
        StubModel(),
        StubTokenizer(),
        AudioProcessor(),
        [audio_path],
        "Describe the audio.",
        max_new_tokens=4,
    )

    assert response == "A synthetic tone."
