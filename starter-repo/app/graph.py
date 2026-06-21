"""Research assistant LangGraph — planner -> researcher -> reflect -> writer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class ResearchState(TypedDict):
    question: str
    sub_questions: list[dict]
    findings: list[dict]
    report: str
    step_log: list[str]
    guardrail_blocked: bool
    reflect_decision: str
    reflect_gaps: str
    reflect_loops: int


def guard_node(state: ResearchState) -> dict:
    """Post-writer citation guardrail: flag any hallucinated URLs."""
    from app.guardrails import validate_citations

    allowed_urls = {f["evidence_url"] for f in state["findings"] if f.get("evidence_url")}
    ok, bad_urls = validate_citations(state["report"], allowed_urls)

    if not ok:
        warning = f"> WARNING: filtered hallucinated citations: {bad_urls}"
        return {
            "report": warning + "\n\n" + state["report"],
            "step_log": state["step_log"] + [f"Guard: {len(bad_urls)} hallucinated URL(s) flagged"],
        }

    return {"step_log": state["step_log"] + ["Guard: all citations valid"]}


def _route_after_reflect(state: ResearchState) -> str:
    if (
        state.get("reflect_decision") == "need_more"
        and state.get("reflect_loops", 0) < 1
    ):
        return "researcher"
    return "writer"


def _get_builder():
    from app.nodes.planner import planner_node
    from app.nodes.reflect import reflect_node
    from app.nodes.researcher import researcher_node
    from app.nodes.writer import writer_node

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("writer", writer_node)
    builder.add_node("guard", guard_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        lambda state: END if state.get("guardrail_blocked") else "researcher",
    )
    builder.add_edge("researcher", "reflect")
    builder.add_conditional_edges("reflect", _route_after_reflect)
    builder.add_edge("writer", "guard")
    builder.add_edge("guard", END)
    return builder


_saver = MemorySaver()


def build_graph():
    return _get_builder().compile(checkpointer=_saver)


async def stream_research(question: str, thread_id: str) -> AsyncIterator[dict[str, Any]]:
    """Yield state snapshots after each node completes."""
    graph = build_graph()
    async for chunk in graph.astream(
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
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        yield chunk


__all__ = ["ResearchState", "build_graph", "stream_research"]
