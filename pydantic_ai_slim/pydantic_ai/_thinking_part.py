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
    i = 0
    len_start = len(START_THINK_TAG)
    len_end = len(END_THINK_TAG)
    n = len(content)

    while i < n:
        start = content.find(START_THINK_TAG, i)
        if start == -1:
            if i < n:
                parts.append(TextPart(content=content[i:]))
            break
        if start > i:
            parts.append(TextPart(content=content[i:start]))
        think_start = start + len_start
        end = content.find(END_THINK_TAG, think_start)
        if end == -1:
            # No closing tag; treat the remainder as text (lose <think> tag)
            parts.append(TextPart(content=content[think_start:]))
            break
        parts.append(ThinkingPart(content=content[think_start:end]))
        i = end + len_end

    return parts
