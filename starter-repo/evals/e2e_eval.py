"""End-to-end eval — full graph + LLM-as-judge report quality (1-5)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401 — load .env via app/__init__.py
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from app.graph import build_graph
from app.llm import get_chat_model
from app.nodes._utils import extract_text

load_dotenv()

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
DATASET_NAME = os.getenv("E2E_GOLDEN_DATASET", "monk-e2e-golden")
PASS_THRESHOLD = 3.0

JUDGE_PROMPT = (
    "On a scale of 1-5, how well does this report answer the question?\n\n"
    "Question: {question}\n\n"
    "Report:\n{report}\n\n"
    "Return JSON with `score` (integer 1-5) and `feedback` (short string)."
)


def load_golden() -> list[dict]:
    rows: list[dict] = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def run_e2e(inputs: dict) -> dict:
    """Target function for LangSmith evaluate — run the full research graph."""
    result = run_graph(inputs["question"])
    return {
        "report": result.get("report", ""),
        "guardrail_blocked": bool(result.get("guardrail_blocked")),
        "findings_count": len(result.get("findings", [])),
    }


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract score (1-5) and feedback from judge JSON or free text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            score = float(data.get("score", 0))
            feedback = str(data.get("feedback", "")).strip()
            return max(1.0, min(5.0, score)), feedback
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"score[\"']?\s*[:=]\s*(\d)", text, re.IGNORECASE)
    score = float(match.group(1)) if match else 1.0
    return max(1.0, min(5.0, score)), text[:200]


def quality_judge(run, example) -> dict:
    """LLM-as-judge: score how well the report answers the question."""
    outputs = run.outputs or {}
    if outputs.get("guardrail_blocked"):
        return {
            "key": "quality",
            "score": 1.0,
            "comment": "Guardrail blocked — no report produced",
        }

    report = str(outputs.get("report", "")).strip()
    if not report:
        return {
            "key": "quality",
            "score": 1.0,
            "comment": "Empty report",
        }

    inputs = example.inputs if hasattr(example, "inputs") else (example.get("inputs") or {})
    question = str(inputs.get("question", ""))

    model = get_chat_model()
    prompt = JUDGE_PROMPT.format(question=question, report=report[:12000])
    resp = model.invoke([HumanMessage(content=prompt)])
    text = extract_text(resp)
    score, feedback = _parse_judge_response(text)

    return {"key": "quality", "score": score, "comment": feedback[:300]}


def _score_from_row(row: dict) -> float | None:
    eval_results = row.get("evaluation_results")
    if eval_results is None:
        return None

    if isinstance(eval_results, dict):
        results_list = eval_results.get("results", [])
    else:
        results_list = eval_results.results

    for result in results_list:
        if isinstance(result, dict):
            key, score = result.get("key"), result.get("score")
        else:
            key, score = result.key, result.score
        if key == "quality" and score is not None:
            return float(score)
    return None


def _comment_from_row(row: dict) -> str:
    eval_results = row.get("evaluation_results")
    if eval_results is None:
        return ""

    if isinstance(eval_results, dict):
        results_list = eval_results.get("results", [])
    else:
        results_list = eval_results.results

    for result in results_list:
        if isinstance(result, dict):
            key, comment = result.get("key"), result.get("comment", "")
        else:
            key, comment = result.key, getattr(result, "comment", "") or ""
        if key == "quality":
            return str(comment)
    return ""


def _question_from_row(row: dict) -> str:
    example = row.get("example")
    if example is None:
        return "?"
    inputs = example.get("inputs") if isinstance(example, dict) else example.inputs
    return (inputs or {}).get("question", "?")


def _rows_to_examples(rows: list[dict]) -> list[dict]:
    return [
        {
            "inputs": {"question": row["question"]},
            "outputs": {
                "expected_sections": row["expected_sections"],
                "min_citations": row["min_citations"],
            },
        }
        for row in rows
    ]


def ensure_golden_dataset(client: Client, rows: list[dict]) -> str:
    """Create or refresh the LangSmith dataset from golden.jsonl."""
    examples = _rows_to_examples(rows)
    if client.has_dataset(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_name=DATASET_NAME)
    client.create_dataset(
        DATASET_NAME,
        description="E2E eval golden set from evals/golden.jsonl",
    )
    client.create_examples(dataset_name=DATASET_NAME, examples=examples)
    return DATASET_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E eval with LLM-as-judge")
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

    client = Client()
    dataset_name = ensure_golden_dataset(client, rows)

    prefix = os.getenv("E2E_EVAL_PREFIX", "e2e-eval")
    print(f"E2E eval — {len(rows)} golden rows (pass threshold: score >= {PASS_THRESHOLD:.0f})\n")

    results = evaluate(
        run_e2e,
        data=dataset_name,
        evaluators=[quality_judge],
        experiment_prefix=prefix,
        description="Full-graph report quality vs golden questions (LLM judge 1-5)",
        max_concurrency=1,
        client=client,
    )

    print(f"\nExperiment: {results.experiment_name}")
    if results.url:
        print(f"LangSmith: {results.url}\n")

    passed = failed = 0
    for row in results:
        question = _question_from_row(row)
        score = _score_from_row(row)
        feedback = _comment_from_row(row)
        if score is None:
            print(f"[SKIP]  — {question[:70]}")
            continue
        ok = score >= PASS_THRESHOLD
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        line = f"[{status}] {score:.0f}/5 — {question[:70]}"
        if feedback:
            line += f"\n         {feedback[:120]}"
        print(line)

    total = passed + failed
    if total:
        print(f"\nAggregate: {passed}/{total} passed ({100 * passed / total:.0f}%)")
    else:
        print("\nNo scores returned — check LangSmith credentials and model access.")


if __name__ == "__main__":
    main()
