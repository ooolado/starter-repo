"""Researcher node — gathers findings via tools for each sub-question."""

from __future__ import annotations

import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph import ResearchState
from app.nodes._utils import extract_text
from app.tools.fetch_url import fetch_url
from app.tools.search_local_docs import search_local_docs
from app.tools.summarize import summarize
from app.tools.web_search import web_search

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"
MAX_TOOL_CALLS_PER_SUB = 4

TOOLS = [web_search, fetch_url, search_local_docs, summarize]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = (
    "You are a focused researcher. Use tools to find 1-3 supporting facts with real "
    "source URLs for the given sub-question. When you have enough evidence, reply with "
    "ONLY a JSON list of findings in this format:\n"
    '[{"claim": "...", "evidence_url": "...", "evidence_text": "..."}]'
)


def _extract_urls_from_messages(messages: list[BaseMessage]) -> set[str]:
    """Collect every URL that actually appeared in ToolMessage content."""
    urls: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        for m in re.finditer(r"https?://[^\s\"'<>\]\)]+", text):
            urls.add(m.group().rstrip(".,;:"))
    return urls


def _parse_findings(text: str, sub_idx: int, allowed_urls: set[str]) -> list[dict]:
    """Try to parse the model's JSON findings and validate URLs."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    validated: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("evidence_url", ""))
        if url and url not in allowed_urls:
            continue
        validated.append(
            {
                "sub_question_index": sub_idx,
                "claim": str(item.get("claim", "")),
                "evidence_url": url,
                "evidence_text": str(item.get("evidence_text", "")),
            }
        )
    return validated


def researcher_node(state: ResearchState) -> dict:
    base_model = init_chat_model(os.getenv("MONK_MODEL", DEFAULT_MODEL))
    model = base_model.bind_tools(TOOLS)
    sub_questions = state["sub_questions"]
    total = len(sub_questions)
    all_findings: list[dict] = []
    step_log: list[str] = list(state["step_log"])

    for sub_idx, sub_q in enumerate(sub_questions):
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=sub_q["text"]),
        ]

        found = False
        for _ in range(MAX_TOOL_CALLS_PER_SUB):
            ai_msg = model.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                text = extract_text(ai_msg)
                allowed = _extract_urls_from_messages(messages)
                findings = _parse_findings(text, sub_idx, allowed)
                all_findings.extend(findings)
                found = True
                break

            for call in ai_msg.tool_calls:
                step_log.append(
                    f"[sub {sub_idx + 1}/{total}] {call['name']}({_summarize_args(call['args'])})"
                )
                try:
                    result = TOOL_MAP[call["name"]].invoke(call["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
                content = result if isinstance(result, str) else str(result)
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if not found:
            step_log.append(f"[sub {sub_idx + 1}/{total}] summarizing gathered data")
            messages.append(HumanMessage(content=(
                "You have gathered enough information. Now reply ONLY with a JSON list of findings. "
                'Each finding: {"claim": "...", "evidence_url": "...", "evidence_text": "..."}. '
                "Use only URLs that appeared in tool results above."
            )))
            ai_msg = base_model.invoke(messages)
            text = extract_text(ai_msg)
            allowed = _extract_urls_from_messages(messages)
            findings = _parse_findings(text, sub_idx, allowed)
            all_findings.extend(findings)
            if not findings:
                step_log.append(f"[sub {sub_idx + 1}/{total}] no parseable findings")

    return {"findings": all_findings, "step_log": step_log}


def _summarize_args(args: dict) -> str:
    """Compact representation of tool args for step_log."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
