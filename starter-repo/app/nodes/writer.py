"""Writer node — drafts the markdown report from research findings."""

from __future__ import annotations

import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph import ResearchState

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"

SYSTEM_PROMPT = (
    "You are writing a research report. Produce a markdown report with: "
    "1) a 2-3 sentence executive summary, "
    "2) one H2 section per sub-question, "
    "3) inline [n] citations after each factual claim, "
    "4) a numbered Sources section at the end listing each unique URL once. "
    "Never invent a URL or fact - only use the supplied findings."
)

_MD_URL_RE = re.compile(r"\[.*?\]\((https?://[^\s\)]+)\)")
_BARE_URL_RE = re.compile(r"(https?://[^\s\)>]+)")


def _allowed_urls(findings: list[dict]) -> set[str]:
    return {f["evidence_url"] for f in findings if f.get("evidence_url")}


def _check_urls(report: str, allowed: set[str]) -> tuple[str, set[str]]:
    """Find any URLs in the report that didn't come from findings."""
    md_urls = set(_MD_URL_RE.findall(report))

    sources_idx = report.lower().rfind("# sources")
    if sources_idx == -1:
        sources_idx = report.lower().rfind("## sources")
    if sources_idx != -1:
        sources_section = report[sources_idx:]
        md_urls |= set(_BARE_URL_RE.findall(sources_section))

    bad = {url for url in md_urls if url not in allowed}
    if bad:
        warning = f"\n\n> WARNING: filtered hallucinated citations: {bad}"
        return report + warning, bad
    return report, set()


def writer_node(state: ResearchState) -> dict:
    model = init_chat_model(os.getenv("MONK_MODEL", DEFAULT_MODEL))
    payload = json.dumps(
        {
            "question": state["question"],
            "sub_questions": state["sub_questions"],
            "findings": state["findings"],
        },
        indent=2,
    )
    ai_msg = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
    )
    from app.nodes._utils import extract_text
    report = extract_text(ai_msg)

    allowed = _allowed_urls(state["findings"])
    report, bad_urls = _check_urls(report, allowed)

    step_log = [*state["step_log"], "Writer: report drafted"]
    if bad_urls:
        step_log.append(f"Writer: {len(bad_urls)} hallucinated URL(s) flagged")

    return {"report": report, "step_log": step_log}
