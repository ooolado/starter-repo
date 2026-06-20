"""Summarize long text with a provider-agnostic chat model."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"


def _summary_prompt(text: str, focus: str) -> str:
    if focus.strip():
        return (
            f"Summarize the following text in exactly three sentences, "
            f"emphasising {focus.strip()}.\n\n{text}"
        )
    return f"Summarize the following text in exactly three sentences.\n\n{text}"


@tool
def summarize(text: str, focus: str = "") -> str:
    """Compress long text into a three-sentence summary. Use this when tool output is too long to fit in context."""
    model = init_chat_model(os.getenv("MONK_MODEL", DEFAULT_MODEL))
    response = model.invoke([HumanMessage(content=_summary_prompt(text, focus))])
    content = response.content
    return content if isinstance(content, str) else str(content)


if __name__ == "__main__":
    sample = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "It extends LangChain with cycles, persistence, and streaming. "
        "Teams use it for agents, workflows, and research assistants."
    )
    print(summarize.invoke({"text": sample, "focus": "LangGraph"}))
