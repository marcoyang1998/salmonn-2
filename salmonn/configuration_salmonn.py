from transformers import PretrainedConfig


class SalmonnConfig(PretrainedConfig):
    model_type = "salmonn_2"

    def __init__(
        self,
        qwen_config=None,
        freeze_audio_encoder=True,
        connector_hidden_size=4096,
        connector_segment_size=5,
        concatenate_encoder_layers=True,
        inject_temporal_embedding_nl=False,
        temporal_granularity=2.0,
        encoder_frame_rate=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qwen_config = qwen_config or {}
        self.freeze_audio_encoder = freeze_audio_encoder
        self.connector_hidden_size = connector_hidden_size
        self.connector_segment_size = connector_segment_size
        self.concatenate_encoder_layers = concatenate_encoder_layers
        self.inject_temporal_embedding_nl = inject_temporal_embedding_nl
        self.temporal_granularity = temporal_granularity
        self.encoder_frame_rate = encoder_frame_rate
