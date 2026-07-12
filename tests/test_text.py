from salmonn.text import clean_decoded_response


def test_clean_decoded_response_removes_thinking_tags():
    assert clean_decoded_response("<think>\n\n</think>\n\nAnswer") == "Answer"


def test_clean_decoded_response_preserves_inner_text():
    assert clean_decoded_response("<think>reasoning</think> answer") == "reasoning answer"
