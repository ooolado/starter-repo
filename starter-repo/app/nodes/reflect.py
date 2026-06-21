"""Reflect node — reviews findings and decides whether to loop back to researcher."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.graph import ResearchState
from app.llm import get_chat_model
from app.nodes._utils import extract_text

SYSTEM_PROMPT = (
    "You are a research quality reviewer. Given the user's question, the planned "
    "sub-questions, and the findings gathered so far, decide whether the evidence is "
    "sufficient to write a comprehensive cited report.\n\n"
    'Return decision "sufficient" if findings cover the sub-questions with real evidence. '
    'Return "need_more" if important sub-questions lack findings or evidence is thin.\n\n'
    "When decision is need_more, gaps must be a concrete instruction telling the "
    "researcher exactly what to look up next (missing topics, weak areas, missing URLs)."
)


class ReflectOutput(BaseModel):
    decision: Literal["sufficient", "need_more"]
    gaps: str = ""


def reflect_node(state: ResearchState) -> dict:
    model = get_chat_model()
    payload = json.dumps(
        {
            "question": state["question"],
            "sub_questions": state["sub_questions"],
            "findings": state["findings"],
        },
        indent=2,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=payload),
    ]

    structured = model.with_structured_output(ReflectOutput)
    result = structured.invoke(messages)

    if result is None:
        ai_msg = model.invoke(messages)
        text = extract_text(ai_msg)
        try:
            data = json.loads(text[text.find("{") : text.rfind("}") + 1])
            result = ReflectOutput.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            result = ReflectOutput(decision="sufficient", gaps="")

    decision = result.decision
    gaps = result.gaps.strip() if decision == "need_more" else ""

    step_log = [
        *state["step_log"],
        f"Reflect: {decision}" + (f" — {gaps[:80]}..." if len(gaps) > 80 else f" — {gaps}" if gaps else ""),
    ]

    updates: dict = {
        "reflect_decision": decision,
        "reflect_gaps": gaps,
        "step_log": step_log,
    }

    return updates
