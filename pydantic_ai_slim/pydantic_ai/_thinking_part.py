from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []

    s, e = START_THINK_TAG, END_THINK_TAG
    pos = 0
    len_s = len(s)
    len_e = len(e)
    content_len = len(content)

    while True:
        start_index = content.find(s, pos)
        if start_index < 0:
            break
        # Text before <think>
        if start_index > pos:
            parts.append(TextPart(content=content[pos:start_index]))
        # Move past <think>
        t_start = start_index + len_s
        end_index = content.find(e, t_start)
        if end_index >= 0:
            # Content inside <think>
            parts.append(ThinkingPart(content=content[t_start:end_index]))
            pos = end_index + len_e
        else:
            # No closing tag found, take the rest as text
            parts.append(TextPart(content=content[t_start:]))
            return parts  # no more valid tags possible
    if pos < content_len:
        parts.append(TextPart(content=content[pos:]))
    return parts
