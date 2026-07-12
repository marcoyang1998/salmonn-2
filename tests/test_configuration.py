from salmonn import SalmonnConfig


def test_configuration_roundtrip():
    config = SalmonnConfig(qwen_config={"model_type": "qwen3"}, connector_segment_size=5)
    restored = SalmonnConfig.from_dict(config.to_dict())
    assert restored.qwen_config["model_type"] == "qwen3"
    assert restored.connector_segment_size == 5
