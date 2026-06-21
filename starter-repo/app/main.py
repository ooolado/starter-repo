"""FastAPI entrypoint — serves the HTMX UI and the research SSE stream."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.graph import stream_research

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI(title="Oladoyin AI Research Assistant")
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

_tasks: dict[str, asyncio.Task] = {}
_queues: dict[str, asyncio.Queue] = {}


class ResearchRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(UI_DIR / "index.html")


@app.post("/research")
async def start_research(req: ResearchRequest):
    thread_id = str(uuid4())
    _queues[thread_id] = asyncio.Queue()

    async def _run():
        try:
            async for chunk in stream_research(req.question, thread_id):
                await _queues[thread_id].put(chunk)
        except Exception as exc:
            logger.exception("Graph error for thread %s", thread_id)
            await _queues[thread_id].put({"__error__": str(exc)})
        finally:
            await _queues[thread_id].put(None)

    _tasks[thread_id] = asyncio.create_task(_run())
    return {"thread_id": thread_id}


@app.get("/stream/{thread_id}")
async def stream(thread_id: str):
    queue = _queues.get(thread_id)
    if queue is None:
        return {"error": "unknown thread_id"}

    async def _generate():
        step_log: list[str] = []
        report = ""

        while True:
            event = await queue.get()
            if event is None:
                final = {"type": "done", "report": report, "step_log": step_log}
                yield {"data": json.dumps(final, default=str)}
                break

            if isinstance(event, dict) and "__error__" in event:
                err = {"type": "error", "message": event["__error__"]}
                yield {"data": json.dumps(err, default=str)}
                continue

            for node_name, updates in event.items():
                if not isinstance(updates, dict):
                    continue
                if "step_log" in updates:
                    step_log = updates["step_log"]
                    payload = {"type": "state", "step_log": step_log, "node": node_name}
                    yield {"data": json.dumps(payload, default=str)}
                if "report" in updates and updates["report"]:
                    report = updates["report"]

    return EventSourceResponse(_generate())
