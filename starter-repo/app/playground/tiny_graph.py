"""Tiny LangGraph demo with a LangSmith trace link."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import get_current_run_tree


class S(TypedDict):
    q: str
    a: str


_trace_url = ""


def respond(state: S) -> dict:
    global _trace_url
    if rt := get_current_run_tree():
        _trace_url = rt.get_url()
    return {"a": f"You asked: {state['q']}"}


g = StateGraph(S)
g.add_node("respond", respond)
g.add_edge(START, "respond")
g.add_edge("respond", END)
app = g.compile()

if __name__ == "__main__":
    print(app.invoke({"q": "hello"}))
    if _trace_url:
        print(f"LangSmith trace: {_trace_url}")
