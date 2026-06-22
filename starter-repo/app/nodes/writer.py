"""Writer node — drafts the markdown report from research findings."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph import ResearchState
from app.llm import get_chat_model

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
_BODY_CITE_RE = re.compile(r"\[(\d+)\]")


def _split_report(report: str) -> tuple[str, str]:
    lower = report.lower()
    for marker in ("## sources", "# sources"):
        idx = lower.rfind(marker)
        if idx != -1:
            return report[:idx].strip(), report[idx:].strip()
    return report.strip(), ""


def _body_citation_numbers(body: str) -> set[int]:
    return {int(n) for n in _BODY_CITE_RE.findall(body)}


def _finalize_citations(report: str, findings: list[dict]) -> str:
    """Rebuild Sources from findings and ensure every source is cited in the body."""
    urls: list[str] = []
    for finding in findings:
        url = finding.get("evidence_url", "")
        if url and url not in urls:
            urls.append(url)
    if not urls:
        return report

    body, _ = _split_report(report)
    max_num = len(urls)
    body = _BODY_CITE_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_num else "",
        body,
    )

    url_to_num = {url: index for index, url in enumerate(urls, start=1)}
    cited = _body_citation_numbers(body)
    evidence_lines = ["", "## Evidence"]
    for finding in findings:
        url = finding.get("evidence_url", "")
        number = url_to_num.get(url)
        if not number or number in cited:
            continue
        claim = str(finding.get("claim", "")).strip()
        if claim:
            evidence_lines.append(f"- {claim} [{number}]")
            cited.add(number)

    for number in range(1, max_num + 1):
        if number not in cited:
            evidence_lines.append(f"- Supporting evidence [{number}]")
            cited.add(number)

    sources_lines = ["## Sources", *[f"{number}. {url}" for number, url in enumerate(urls, start=1)]]
    return body + "\n".join(evidence_lines) + "\n\n" + "\n".join(sources_lines)


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
    findings = state["findings"]

    if not findings:
        return {
            "report": (
                "# Research Report\n\n"
                "The research agent was unable to collect verifiable findings for this question. "
                "This can happen when web search results are temporarily unavailable or the question "
                "requires sources not accessible to the current tool set.\n\n"
                f"**Question:** {state['question']}\n\n"
                "Please try again or rephrase the question."
            ),
            "step_log": state["step_log"] + ["Writer: no findings available — returned guidance"],
        }

    model = get_chat_model()
    payload = json.dumps(
        {
            "question": state["question"],
            "sub_questions": state["sub_questions"],
            "findings": findings,
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

    allowed = _allowed_urls(findings)
    report = _finalize_citations(report, findings)
    report, bad_urls = _check_urls(report, allowed)

    step_log = [*state["step_log"], "Writer: report drafted"]
    if bad_urls:
        step_log.append(f"Writer: {len(bad_urls)} hallucinated URL(s) flagged")

    return {"report": report, "step_log": step_log}
