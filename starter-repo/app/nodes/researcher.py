"""Researcher node — gathers findings via tools for each sub-question."""

from __future__ import annotations

import ast
import json
import re
from urllib.parse import urlparse, urlunparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph import ResearchState
from app.llm import get_chat_model
from app.nodes._utils import extract_text
from app.tools.fetch_url import fetch_url
from app.tools.read_pdf import read_pdf
from app.tools.search_local_docs import search_local_docs
from app.tools.summarize import summarize
from app.tools.web_search import web_search

MAX_TOOL_CALLS_PER_SUB = 6

TOOLS = [web_search, fetch_url, read_pdf, search_local_docs, summarize]
TOOL_MAP = {t.name: t for t in TOOLS}
_BEDROCK_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

SYSTEM_PROMPT = (
    "You are a focused researcher. Use tools to find 1-3 supporting facts with real "
    "source URLs for the given sub-question. When you have enough evidence, reply with "
    "ONLY a JSON list of findings in this format:\n"
    '[{"claim": "...", "evidence_url": "...", "evidence_text": "..."}]'
)


def _resolve_tool_name(name: str) -> str | None:
    """Map model-emitted tool names to registered tools (Bedrock requires [a-zA-Z0-9_-]+)."""
    if name in TOOL_MAP:
        return name
    tail = re.sub(r"[^a-zA-Z0-9_-]", "_", name.split(".")[-1])
    if tail in TOOL_MAP:
        return tail
    normalized = name.lower().replace("-", "_")
    for key in TOOL_MAP:
        if normalized == key.lower() or normalized.endswith(key):
            return key
    return None


def _ai_transcript_text(msg: AIMessage) -> str:
    """Text to include when flattening assistant turns (text + reasoning blocks)."""
    text = extract_text(msg)
    if text.strip():
        return text
    content = msg.content
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "reasoning_content":
                rc = block.get("reasoning_content") or {}
                if isinstance(rc, dict):
                    reasoning = str(rc.get("text", "")).strip()
                    if reasoning:
                        return reasoning
    return ""


def _messages_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Convert tool-call history to plain text so Bedrock never replays invalid toolUse blocks."""
    has_tool_history = any(
        isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
        for m in messages
    )
    if not has_tool_history:
        return _sanitize_messages(messages)

    parts: list[str] = []
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            text = str(msg.content).strip()
            if text:
                parts.append(text)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    name = _resolve_tool_name(call["name"]) or call["name"]
                    parts.append(
                        f"\n[Tool call: {name}({json.dumps(call['args'])})]"
                    )
            else:
                text = _ai_transcript_text(msg)
                if text.strip():
                    parts.append(f"\n[Assistant reply]\n{text}")
        elif isinstance(msg, ToolMessage):
            body = msg.content if isinstance(msg.content, str) else str(msg.content)
            if body.strip():
                parts.append(f"\n[Tool result]\n{body}")

    transcript = "\n".join(parts).strip() or "(continuing research)"
    return [*system_msgs, HumanMessage(content=transcript)]


def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Bedrock Converse rejects messages whose content field is empty."""
    sanitized: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            sanitized.append(msg)
        elif isinstance(msg, HumanMessage):
            text = str(msg.content).strip() or "(continuing research)"
            sanitized.append(HumanMessage(content=text))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                sanitized.append(msg)
                continue
            text = _ai_transcript_text(msg)
            if text.strip():
                sanitized.append(AIMessage(content=text))
            # Skip empty assistant turns — they cause ValidationException.
        elif isinstance(msg, ToolMessage):
            body = msg.content if isinstance(msg.content, str) else str(msg.content)
            sanitized.append(ToolMessage(content=body.strip() or "(empty tool result)", tool_call_id=msg.tool_call_id))
        else:
            sanitized.append(msg)
    return sanitized


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip(".,;:)")
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or ""
    return urlunparse((parsed.scheme.lower(), netloc, path, "", parsed.query, ""))


def _resolve_allowed_url(url: str, allowed_urls: set[str]) -> str | None:
    if not url:
        return None
    norm = _normalize_url(url)
    for candidate in allowed_urls:
        if _normalize_url(candidate) == norm:
            return candidate
    return None


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
    """Try to parse the model's JSON findings and validate URLs.

    If strict URL validation produces zero results but the model returned
    parseable findings with URLs, accept them anyway (the URL likely came from
    tool results but was slightly transformed by the model).
    """
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    strict: list[dict] = []
    relaxed: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("evidence_url", ""))
        claim = str(item.get("claim", "")).strip()
        evidence_text = str(item.get("evidence_text", "")).strip()
        if not claim and not evidence_text:
            continue

        resolved = _resolve_allowed_url(raw_url, allowed_urls) if raw_url else ""
        entry = {
            "sub_question_index": sub_idx,
            "claim": claim,
            "evidence_url": resolved or raw_url,
            "evidence_text": evidence_text,
        }
        if resolved:
            strict.append(entry)
        elif raw_url and raw_url.startswith("http"):
            relaxed.append(entry)

    return strict if strict else relaxed


def _try_parse_list(text: str) -> list[dict] | None:
    """Try json.loads then ast.literal_eval to parse a list of dicts."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    fragment = text[start : end + 1]
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(fragment)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return None


def _fallback_findings_from_tools(
    messages: list[BaseMessage], sub_idx: int, limit: int = 5
) -> list[dict]:
    """Build findings directly from tool outputs when the model returns no JSON."""
    findings: list[dict] = []
    seen: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        text = msg.content if isinstance(msg.content, str) else str(msg.content)

        for prefix in (r"\[Source: (https?://[^\]]+)\]", r"\[PDF Source: (https?://[^\]]+)\]"):
            match = re.match(rf"{prefix}\n(.*)", text, re.DOTALL)
            if not match:
                continue
            url = match.group(1).strip()
            body = match.group(2).strip()
            key = _normalize_url(url)
            if key in seen or not body:
                continue
            seen.add(key)
            claim = " ".join(body.split()[:24])
            findings.append(
                {
                    "sub_question_index": sub_idx,
                    "claim": claim,
                    "evidence_url": url,
                    "evidence_text": body[:500],
                }
            )
            if len(findings) >= limit:
                return findings
            break

        data = _try_parse_list(text)
        if not data:
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            url = str(
                item.get("url") or item.get("source_url") or item.get("evidence_url") or ""
            ).strip()
            content = str(
                item.get("content") or item.get("text") or item.get("snippet")
                or item.get("title") or item.get("evidence_text") or ""
            ).strip()
            if not url or not content:
                continue
            key = _normalize_url(url)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {
                    "sub_question_index": sub_idx,
                    "claim": " ".join(content.split()[:24]),
                    "evidence_url": url,
                    "evidence_text": content[:500],
                }
            )
            if len(findings) >= limit:
                return findings

    return findings[:limit]


def researcher_node(state: ResearchState) -> dict:
    gaps = state.get("reflect_gaps", "").strip()
    if gaps:
        return _research_gaps(state, gaps)

    base_model = get_chat_model()
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

        sub_findings: list[dict] = []
        for _ in range(MAX_TOOL_CALLS_PER_SUB):
            ai_msg = model.invoke(_messages_for_model(messages))

            if not ai_msg.tool_calls:
                text = extract_text(ai_msg)
                if not text.strip():
                    messages.append(
                        HumanMessage(
                            content=(
                                "Continue research: use a tool or reply ONLY with a JSON list of findings."
                            )
                        )
                    )
                    continue
                messages.append(ai_msg)
                allowed = _extract_urls_from_messages(messages)
                parsed = _parse_findings(text, sub_idx, allowed)
                if parsed:
                    sub_findings = parsed
                    break
                continue

            messages.append(ai_msg)
            for call in ai_msg.tool_calls:
                tool_name = _resolve_tool_name(call["name"])
                step_log.append(
                    f"[sub {sub_idx + 1}/{total}] {tool_name or call['name']}({_summarize_args(call['args'])})"
                )
                if not tool_name or not _BEDROCK_TOOL_NAME_RE.match(tool_name):
                    result = (
                        f"Error: unknown tool {call['name']!r}. "
                        f"Use one of: {', '.join(TOOL_MAP)}"
                    )
                else:
                    try:
                        result = TOOL_MAP[tool_name].invoke(call["args"])
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                content = result if isinstance(result, str) else str(result)
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if not sub_findings:
            step_log.append(f"[sub {sub_idx + 1}/{total}] summarizing gathered data")
            messages.append(HumanMessage(content=(
                "You have gathered enough information. Now reply ONLY with a JSON list of findings. "
                'Each finding: {"claim": "...", "evidence_url": "...", "evidence_text": "..."}. '
                "Use only URLs that appeared in tool results above."
            )))
            ai_msg = base_model.invoke(_messages_for_model(messages))
            text = extract_text(ai_msg)
            allowed = _extract_urls_from_messages(messages)
            sub_findings = _parse_findings(text, sub_idx, allowed)

        if not sub_findings:
            sub_findings = _fallback_findings_from_tools(messages, sub_idx)
            if sub_findings:
                step_log.append(
                    f"[sub {sub_idx + 1}/{total}] recovered {len(sub_findings)} finding(s) from tool output"
                )
            else:
                step_log.append(f"[sub {sub_idx + 1}/{total}] no parseable findings")

        all_findings.extend(sub_findings)

    return {"findings": all_findings, "step_log": step_log}


def _research_gaps(state: ResearchState, gaps: str) -> dict:
    """Second-pass research to fill gaps identified by reflect_node."""
    base_model = get_chat_model()
    model = base_model.bind_tools(TOOLS)
    step_log: list[str] = [*state["step_log"], f"Researcher: gap-fill — {gaps[:100]}"]

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original question: {state['question']}\n\n"
                f"Fill these specific gaps:\n{gaps}\n\n"
                "Use tools as needed, then reply ONLY with a JSON list of NEW findings:\n"
                '[{"claim": "...", "evidence_url": "...", "evidence_text": "..."}]'
            )
        ),
    ]

    existing = list(state.get("findings", []))
    sub_idx = max((f.get("sub_question_index", 0) for f in existing), default=0)

    for _ in range(MAX_TOOL_CALLS_PER_SUB):
        ai_msg = model.invoke(_messages_for_model(messages))

        if not ai_msg.tool_calls:
            text = extract_text(ai_msg)
            if not text.strip():
                messages.append(
                    HumanMessage(
                        content="Continue gap-fill: use a tool or reply with JSON findings."
                    )
                )
                continue
            messages.append(ai_msg)
            allowed = _extract_urls_from_messages(messages)
            new_findings = _parse_findings(text, sub_idx, allowed)
            if not new_findings:
                new_findings = _fallback_findings_from_tools(messages, sub_idx)
            return {
                "findings": existing + new_findings,
                "reflect_gaps": "",
                "reflect_loops": state.get("reflect_loops", 0) + 1,
                "step_log": step_log + [f"Researcher: added {len(new_findings)} gap-fill finding(s)"],
            }

        messages.append(ai_msg)
        for call in ai_msg.tool_calls:
            tool_name = _resolve_tool_name(call["name"])
            step_log.append(f"[gap-fill] {tool_name or call['name']}({_summarize_args(call['args'])})")
            if not tool_name or not _BEDROCK_TOOL_NAME_RE.match(tool_name):
                result = f"Error: unknown tool {call['name']!r}."
            else:
                try:
                    result = TOOL_MAP[tool_name].invoke(call["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            content = result if isinstance(result, str) else str(result)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    messages.append(HumanMessage(content=(
        "Reply ONLY with a JSON list of NEW findings for the gaps above."
    )))
    ai_msg = base_model.invoke(_messages_for_model(messages))
    allowed = _extract_urls_from_messages(messages)
    new_findings = _parse_findings(extract_text(ai_msg), sub_idx, allowed)
    if not new_findings:
        new_findings = _fallback_findings_from_tools(messages, sub_idx)
    return {
        "findings": existing + new_findings,
        "reflect_gaps": "",
        "reflect_loops": state.get("reflect_loops", 0) + 1,
        "step_log": step_log + [f"Researcher: added {len(new_findings)} gap-fill finding(s)"],
    }


def _summarize_args(args: dict) -> str:
    """Compact representation of tool args for step_log."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
