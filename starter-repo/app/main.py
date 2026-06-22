"""FastAPI entrypoint — serves the HTMX UI and the research SSE stream."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.graph import stream_research

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"

# --------------- Rate Limiting & Usage Cap ---------------
RATE_LIMIT_PER_IP = int(os.environ.get("RATE_LIMIT_PER_IP", "5"))  # requests per window
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))  # window size
DAILY_QUERY_CAP = int(os.environ.get("DAILY_QUERY_CAP", "100"))  # global daily max

_ip_requests: dict[str, list[float]] = defaultdict(list)
_daily_count: int = 0
_daily_reset: float = 0.0


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> str | None:
    """Returns an error message if rate-limited, else None."""
    global _daily_count, _daily_reset

    now = time.time()

    # Reset daily counter at midnight boundary (every 86400s)
    if now - _daily_reset > 86400:
        _daily_count = 0
        _daily_reset = now

    if _daily_count >= DAILY_QUERY_CAP:
        return (
            f"Daily usage cap reached ({DAILY_QUERY_CAP} queries/day). "
            "Please try again tomorrow."
        )

    # Sliding window per-IP
    timestamps = _ip_requests[ip]
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    _ip_requests[ip] = [t for t in timestamps if t > cutoff]
    if len(_ip_requests[ip]) >= RATE_LIMIT_PER_IP:
        return (
            f"Rate limit exceeded ({RATE_LIMIT_PER_IP} requests per "
            f"{RATE_LIMIT_WINDOW_SEC}s). Please wait before trying again."
        )

    _ip_requests[ip].append(now)
    _daily_count += 1
    return None
# ---------------------------------------------------------

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
async def start_research(req: ResearchRequest, request: Request):
    # Enforce rate limit
    ip = _get_client_ip(request)
    error = _check_rate_limit(ip)
    if error:
        return JSONResponse(status_code=429, content={"error": error})

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
