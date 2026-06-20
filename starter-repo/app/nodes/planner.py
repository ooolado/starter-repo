"""Planner node — decomposes the user question into sub-questions."""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.graph import ResearchState
from app.nodes._utils import extract_text

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"
SYSTEM_PROMPT = (
    "You are a research planner. Decompose the user's question into 3-7 sub-questions "
    "that, taken together, fully cover the question. Tag each as 'web' (current/news/general), "
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
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]

    model = init_chat_model(os.getenv("MONK_MODEL", DEFAULT_MODEL))

    structured = model.with_structured_output(PlannerOutput)
    result = structured.invoke(messages)

    if result is None:
        ai_msg = model.invoke(messages)
        text = extract_text(ai_msg)
        result = _parse_planner_json(text)

    if result is None:
        result = PlannerOutput(sub_questions=[
            SubQuestion(text=state["question"], source="web"),
        ])

    return {
        "sub_questions": [sq.model_dump() for sq in result.sub_questions],
        "step_log": state["step_log"] + [f"Planner: {len(result.sub_questions)} sub-questions"],
    }
