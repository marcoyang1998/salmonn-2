def prepare_audio_prompt(prompt: str, audio_count: int) -> str:
    """Preserve explicit audio placement or prepend placeholders for simple prompts."""
    placeholder_count = prompt.count("<audio>")
    if placeholder_count == 0:
        return "<audio>" * audio_count + prompt
    if placeholder_count != audio_count:
        raise ValueError(
            f"Prompt contains {placeholder_count} <audio> placeholders but received {audio_count} audio files"
        )
    return prompt


def clean_decoded_response(text: str) -> str:
    """Remove Qwen thinking boundary tags from user-facing decoded text."""
    return text.replace("<think>", "").replace("</think>", "").strip()
