def clean_decoded_response(text: str) -> str:
    """Remove Qwen thinking boundary tags from user-facing decoded text."""
    return text.replace("<think>", "").replace("</think>", "").strip()
