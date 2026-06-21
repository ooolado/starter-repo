"""Shared utilities for graph nodes."""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def extract_text(msg: BaseMessage) -> str:
    """Pull the text content from a model response, handling list-of-blocks format.

    Bedrock models return content as a list of dicts like:
      [{"type": "reasoning_content", ...}, {"type": "text", "text": "..."}]
    This helper extracts and concatenates only the 'text' blocks.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def guardrail_blocked(msg: BaseMessage) -> bool:
    """True when Bedrock Guardrail intervened on this response."""
    meta = getattr(msg, "response_metadata", None) or {}
    return meta.get("stopReason") == "guardrail_intervened"
