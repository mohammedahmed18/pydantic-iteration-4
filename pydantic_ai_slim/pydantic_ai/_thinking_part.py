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
    pos = 0
    clen = len(content)
    stt_len = len(START_THINK_TAG)
    ett_len = len(END_THINK_TAG)

    while True:
        start_index = content.find(START_THINK_TAG, pos)
        if start_index == -1:
            if pos < clen:
                # Remaining text after last <think>
                parts.append(TextPart(content=content[pos:]))
            break

        if start_index > pos:
            # Text before <think>
            parts.append(TextPart(content=content[pos:start_index]))

        think_start = start_index + stt_len
        end_index = content.find(END_THINK_TAG, think_start)
        if end_index == -1:
            # No ending tag: treat the rest as plain text
            parts.append(TextPart(content=content[think_start:]))
            break
        else:
            # Between start and end tag is ThinkingPart
            parts.append(ThinkingPart(content=content[think_start:end_index]))
            pos = end_index + ett_len  # move past the end tag

    return parts
