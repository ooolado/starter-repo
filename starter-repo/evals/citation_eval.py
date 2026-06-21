"""Citation eval — programmatic checks on full-graph research reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401 — load .env via app/__init__.py
from dotenv import load_dotenv

from app.graph import build_graph
from app.guardrails import extract_urls

load_dotenv()

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

_BODY_CITE_RE = re.compile(r"\[(\d+)\]")
_SOURCES_LINE_RE = re.compile(
    r"^\s*(?:\[(\d+)\]|(\d+)\.)\s*(?:\S.*)?$",
    re.MULTILINE,
)


def load_golden() -> list[dict]:
    rows: list[dict] = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_report(report: str) -> tuple[str, str]:
    """Split report into body and Sources section."""
    lower = report.lower()
    for marker in ("## sources", "# sources"):
        idx = lower.rfind(marker)
        if idx != -1:
            return report[:idx].strip(), report[idx:].strip()
    return report.strip(), ""


def body_citation_numbers(body: str) -> set[int]:
    return {int(n) for n in _BODY_CITE_RE.findall(body)}


def sources_citation_numbers(sources: str) -> set[int]:
    nums: set[int] = set()
    for match in _SOURCES_LINE_RE.finditer(sources):
        num = match.group(1) or match.group(2)
        if num:
            nums.add(int(num))
    return nums


def check_citations(report: str, min_citations: int) -> tuple[bool, list[str]]:
    """Return (ok, failure_reasons) for the three programmatic citation checks."""
    failures: list[str] = []
    body, sources = split_report(report)

    unique_urls = extract_urls(report)
    if len(unique_urls) < min_citations:
        failures.append(
            f"(a) unique URLs {len(unique_urls)} < min_citations {min_citations}"
        )

    body_nums = body_citation_numbers(body)
    source_nums = sources_citation_numbers(sources)
    missing_in_sources = sorted(body_nums - source_nums)
    if missing_in_sources:
        failures.append(
            f"(b) body [n] missing from Sources: {missing_in_sources}"
        )

    # Reports use inline [n] in the body and numbered URLs in Sources.
    # Check (c): every Sources entry must be cited in the body as [n].
    uncited_sources = sorted(source_nums - body_nums)
    if uncited_sources:
        failures.append(
            f"(c) Sources entries not cited in body: {uncited_sources}"
        )

    return len(failures) == 0, failures


def run_graph(question: str) -> dict:
    graph = build_graph()
    return graph.invoke(
        {
            "question": question,
            "sub_questions": [],
            "findings": [],
            "report": "",
            "step_log": [],
            "guardrail_blocked": False,
            "reflect_decision": "",
            "reflect_gaps": "",
            "reflect_loops": 0,
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run citation eval on golden rows")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N rows (0 = all)",
    )
    args = parser.parse_args()

    rows = load_golden()
    if args.limit > 0:
        rows = rows[: args.limit]

    passed = failed = 0

    print(f"Citation eval — {len(rows)} golden rows\n")

    for i, row in enumerate(rows, start=1):
        question = row["question"]
        min_citations = int(row["min_citations"])

        print(f"[{i}/{len(rows)}] {question[:70]}...")
        try:
            result = run_graph(question)
        except Exception as exc:
            print(f"  [FAIL] graph error: {exc}")
            failed += 1
            continue
        report = result.get("report", "")

        if result.get("guardrail_blocked"):
            print("  [FAIL] guardrail blocked — no report")
            failed += 1
            continue

        findings_count = len(result.get("findings", []))
        ok, reasons = check_citations(report, min_citations)
        if ok:
            print(
                f"  [PASS] {findings_count} finding(s), "
                f"{len(extract_urls(report))} unique URLs"
            )
            passed += 1
        else:
            print(f"  [FAIL] {findings_count} finding(s)")
            for reason in reasons:
                print(f"         {reason}")
            failed += 1

    total = passed + failed
    if total:
        print(f"\nAggregate: {passed}/{total} passed ({100 * passed / total:.0f}%)")
    else:
        print("\nNo rows evaluated.")


if __name__ == "__main__":
    main()
