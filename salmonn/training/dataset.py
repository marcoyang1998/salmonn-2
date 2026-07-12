import json
from pathlib import Path

from torch.utils.data import Dataset


class SalmonnDataset(Dataset):
    """Generic conversation manifest; contains no benchmark-specific behavior."""

    def __init__(self, manifest):
        with Path(manifest).open(encoding="utf-8") as handle:
            self.samples = json.load(handle)
        if not isinstance(self.samples, list):
            raise ValueError("The training manifest must contain a JSON list")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if not sample.get("audios") or not sample.get("messages"):
            raise ValueError("Each sample requires non-empty 'audios' and 'messages' fields")
        placeholders = sum(message.get("content", "").count("<audio>") for message in sample["messages"])
        if placeholders != len(sample["audios"]):
            raise ValueError("The number of <audio> placeholders must match the number of audio files")
        return sample
