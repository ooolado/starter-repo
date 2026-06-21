"""Planner eval — scores sub-question coverage against golden expected_sections."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401 — load .env via app/__init__.py
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from app.llm import get_chat_model
from app.nodes._utils import extract_text
from app.nodes.planner import planner_node

load_dotenv()

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
DATASET_NAME = os.getenv("PLANNER_GOLDEN_DATASET", "monk-planner-golden")
PASS_THRESHOLD = 0.7
JUDGE_PROMPT = (
    "Given the sub-questions {sqs} and the expected coverage areas {expected_sections}, "
    "return a number 0.0-1.0 representing how well the sub-questions cover the expected areas. "
    "Return only a number."
)


def load_golden() -> list[dict]:
    rows: list[dict] = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_planner(inputs: dict) -> dict:
    """Target function for LangSmith evaluate — run planner_node only."""
    result = planner_node(
        {
            "question": inputs["question"],
            "sub_questions": [],
            "findings": [],
            "report": "",
            "step_log": [],
            "guardrail_blocked": False,
        }
    )
    return {"sub_questions": result.get("sub_questions", [])}


def _parse_score(text: str) -> float:
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(1))))


def coverage_judge(run, example) -> dict:
    """LLM-as-judge: score how well sub-questions cover expected_sections."""
    sqs = (run.outputs or {}).get("sub_questions", [])
    expected = (example.outputs or {}).get("expected_sections", [])
    model = get_chat_model()
    prompt = JUDGE_PROMPT.format(sqs=sqs, expected_sections=expected)
    resp = model.invoke([HumanMessage(content=prompt)])
    text = extract_text(resp)
    score = _parse_score(text)
    return {"key": "coverage", "score": score, "comment": text.strip()[:120]}


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
        if key == "coverage" and score is not None:
            return float(score)
    return None


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
        description="Planner eval golden set from evals/golden.jsonl",
    )
    client.create_examples(dataset_name=DATASET_NAME, examples=examples)
    return DATASET_NAME


def main() -> None:
    rows = load_golden()
    client = Client()
    dataset_name = ensure_golden_dataset(client, rows)

    prefix = os.getenv("PLANNER_EVAL_PREFIX", "planner-eval")
    results = evaluate(
        run_planner,
        data=dataset_name,
        evaluators=[coverage_judge],
        experiment_prefix=prefix,
        description="Planner coverage vs golden expected_sections",
        max_concurrency=2,
        client=client,
    )

    print(f"\nExperiment: {results.experiment_name}")
    if results.url:
        print(f"LangSmith: {results.url}\n")

    passed = failed = 0
    for row in results:
        question = _question_from_row(row)
        score = _score_from_row(row)
        if score is None:
            print(f"[SKIP]  — {question[:70]}")
            continue
        ok = score >= PASS_THRESHOLD
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {score:.2f} — {question[:70]}")

    total = passed + failed
    if total:
        print(f"\nAggregate: {passed}/{total} passed ({100 * passed / total:.0f}%)")
    else:
        print("\nNo scores returned — check LangSmith credentials and model access.")


if __name__ == "__main__":
    main()
