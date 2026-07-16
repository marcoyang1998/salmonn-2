import pytest

from salmonn.text import clean_decoded_response, prepare_audio_prompt


def test_prepare_audio_prompt_prepends_missing_placeholders():
    assert prepare_audio_prompt("Describe the audio.", 2) == "<audio><audio>Describe the audio."


def test_prepare_audio_prompt_preserves_explicit_placement():
    prompt = "<audio>Transcribe this. References: <audio>SALMONN <audio>SPEAR"
    assert prepare_audio_prompt(prompt, 3) == prompt


def test_prepare_audio_prompt_rejects_mismatched_count():
    with pytest.raises(ValueError, match="1 <audio> placeholders but received 2 audio files"):
        prepare_audio_prompt("<audio>Transcribe this.", 2)


def test_clean_decoded_response_removes_thinking_tags():
    assert clean_decoded_response("<think>\n\n</think>\n\nAnswer") == "Answer"


def test_clean_decoded_response_preserves_inner_text():
    assert clean_decoded_response("<think>reasoning</think> answer") == "reasoning answer"
