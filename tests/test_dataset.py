import json

import pytest

from salmonn.training import SalmonnDataset


def test_manifest_validation(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{
        "audios": ["audio.wav"],
        "messages": [{"role": "user", "content": "<audio>Describe it."}]
    }]))
    assert len(SalmonnDataset(path)) == 1


def test_placeholder_count(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"audios": ["a.wav"], "messages": [{"role": "user", "content": "No audio"}]}]))
    with pytest.raises(ValueError):
        SalmonnDataset(path)[0]
