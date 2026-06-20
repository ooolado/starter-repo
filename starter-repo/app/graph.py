"""Research assistant LangGraph — planner -> researcher -> writer."""

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


_builder = None


def _get_builder():
    global _builder
    if _builder is not None:
        return _builder

    from app.nodes.planner import planner_node
    from app.nodes.researcher import researcher_node
    from app.nodes.writer import writer_node

    _builder = StateGraph(ResearchState)
    _builder.add_node("planner", planner_node)
    _builder.add_node("researcher", researcher_node)
    _builder.add_node("writer", writer_node)
    _builder.add_edge(START, "planner")
    _builder.add_edge("planner", "researcher")
    _builder.add_edge("researcher", "writer")
    _builder.add_edge("writer", END)
    return _builder


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
        },
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        yield chunk


__all__ = ["ResearchState", "build_graph", "stream_research"]
