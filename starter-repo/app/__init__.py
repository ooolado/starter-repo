"""Monk Technologies - AI Research Assistant (Project 1).

This package auto-loads `.env` from the project root on import so that
`MONK_MODEL`, `MONK_EMBEDDINGS`, `POSTGRES_DSN`, etc. resolve correctly when
the app is launched via `uvicorn`, `python -m`, or `pytest`. Existing
environment variables take precedence (dotenv `override=False`).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent.parent
    _env = _project_root / ".env"
    if _env.exists():
        load_dotenv(_env, override=False)
except Exception:
    pass

# LangChain reads LANGCHAIN_API_KEY; keep it aligned with LANGSMITH_API_KEY.
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]
