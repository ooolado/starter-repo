"""Planner node — decomposes the user question into sub-questions."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.graph import ResearchState
from app.guardrails import check_input_guardrail
from app.llm import get_chat_model
from app.nodes._utils import extract_text, guardrail_blocked
MAX_SUB_QUESTIONS = 5

SYSTEM_PROMPT = (
    f"You are a research planner. Decompose the user's question into 2-{MAX_SUB_QUESTIONS} sub-questions "
    "that, taken together, fully cover the question. For simple questions use fewer sub-questions (2-3). "
    "Tag each as 'web' (current/news/general), "
    "'local' (likely in our internal docs corpus), or 'both'.\n\n"
    "Reply ONLY with JSON in this exact format:\n"
    '{"sub_questions": [{"text": "...", "source": "web|local|both"}, ...]}'
)


class SubQuestion(BaseModel):
    text: str
    source: Literal["web", "local", "both"] = "web"


class PlannerOutput(BaseModel):
    sub_questions: list[SubQuestion]


def _parse_planner_json(text: str) -> PlannerOutput | None:
    """Extract and validate planner JSON from model text."""
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            continue
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                data = {"sub_questions": data}
            return PlannerOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def planner_node(state: ResearchState) -> dict:
    blocked, refusal = check_input_guardrail(state["question"])
    if blocked:
        return {
            "guardrail_blocked": True,
            "sub_questions": [],
            "findings": [],
            "report": refusal,
            "step_log": state["step_log"] + ["Planner: blocked by guardrail"],
        }

    model = get_chat_model()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]

    structured = model.with_structured_output(PlannerOutput)
    result = structured.invoke(messages)

    if result is None:
        ai_msg = model.invoke(messages)
        if guardrail_blocked(ai_msg):
            return {
                "guardrail_blocked": True,
                "sub_questions": [],
                "findings": [],
                "report": extract_text(ai_msg),
                "step_log": state["step_log"] + ["Planner: blocked by guardrail"],
            }
        text = extract_text(ai_msg)
        result = _parse_planner_json(text)

    if result is None:
        result = PlannerOutput(sub_questions=[
            SubQuestion(text=state["question"], source="web"),
        ])

    sqs = result.sub_questions[:MAX_SUB_QUESTIONS]
    return {
        "sub_questions": [sq.model_dump() for sq in sqs],
        "step_log": state["step_log"] + [f"Planner: {len(sqs)} sub-questions"],
    }
