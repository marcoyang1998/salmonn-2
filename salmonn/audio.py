from pathlib import Path

import torch
import torchaudio


class AudioProcessor:
    """Load audio and compute the 128-bin filterbanks expected by SPEAR."""

    def __init__(self, sample_rate=16000, num_mel_bins=128):
        from lhotse import Fbank, FbankConfig

        self.sample_rate = sample_rate
        self.fbank = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    def __call__(self, path):
        path = str(Path(path).expanduser())
        waveform, sample_rate = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        feature = self.fbank.extract(waveform.squeeze(0), sampling_rate=self.sample_rate)
        return feature.to(torch.float32)


def pad_audio_features(features):
    lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(features, batch_first=True), lengths
