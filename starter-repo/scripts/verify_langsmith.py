"""Verify LangSmith credentials before enabling tracing.

Run: uv run python scripts/verify_langsmith.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

KEY = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
PROJECT = (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "").strip()
ENDPOINT = (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")


def _key_kind(key: str) -> str:
    if key.startswith("lsv2_pt_"):
        return "personal access token (lsv2_pt_)"
    if key.startswith("lsv2_sk_"):
        return "service key (lsv2_sk_)"
    if key.startswith("lsv2_"):
        return "LangSmith key"
    return "unrecognized format"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


def _print_pat_fix_checklist() -> None:
    print(
        "Your personal key (lsv2_pt_) does NOT need LANGSMITH_WORKSPACE_ID.\n"
        "LangSmith is still rejecting this key. Try in order:\n"
        "  1. In https://smith.langchain.com create project "
        f"'{PROJECT or 'OrlandoProject'}' in the left sidebar (+ New Project)\n"
        "  2. Settings → API Keys → delete old key → create a new personal key\n"
        "  3. Copy the FULL key shown once (starts with lsv2_pt_) into .env\n"
        "  4. Confirm both LANGSMITH_API_KEY and LANGCHAIN_API_KEY match\n"
        "  5. Leave LANGSMITH_WORKSPACE_ID unset for personal keys\n"
        "  6. Re-run this script\n"
        "\n"
        "If it still fails on a brand-new account, email support@langchain.dev with\n"
        "your key's short ID from the API Keys page (e.g. lsv2_pt_7213...ad3).\n"
        "\n"
        "Local bootcamp labs work without LangSmith — keep LANGSMITH_TRACING=false\n"
        "until auth passes here.",
        file=sys.stderr,
    )


def _print_service_key_checklist() -> None:
    print(
        "Service keys (lsv2_sk_) require LANGSMITH_WORKSPACE_ID.\n"
        "Personal keys (lsv2_pt_) do not — use a personal key for local dev instead.",
        file=sys.stderr,
    )


def _check_auth() -> int:
    if not KEY:
        return _fail(
            "LANGSMITH_API_KEY is missing.\n"
            "Create a personal key at https://smith.langchain.com/settings."
        )

    print(f"Key type: {_key_kind(KEY)}")

    if KEY.startswith("lsv2_sk_") and not os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        _print_service_key_checklist()
        return 1

    if not PROJECT:
        print("Warning: LANGSMITH_PROJECT is not set.\n")

    os.environ.setdefault("LANGSMITH_API_KEY", KEY)
    os.environ.setdefault("LANGCHAIN_API_KEY", KEY)
    os.environ.setdefault("LANGSMITH_ENDPOINT", ENDPOINT)

    from langsmith import Client

    try:
        projects = [p.name for p in Client().list_projects(limit=10)]
    except Exception as exc:
        print(f"LangSmith auth failed: {exc}\n", file=sys.stderr)
        if KEY.startswith("lsv2_pt_"):
            _print_pat_fix_checklist()
        else:
            _print_service_key_checklist()
        return 1

    print(f"LangSmith OK — {len(projects)} project(s) visible:")
    for name in projects:
        marker = " <-- configured" if PROJECT and name == PROJECT else ""
        print(f"  - {name}{marker}")

    if PROJECT and PROJECT not in projects:
        print(
            f"\nWarning: LANGSMITH_PROJECT={PROJECT!r} not found yet. "
            f"Create it in the UI or tracing will auto-create on first run.",
            file=sys.stderr,
        )

    if _tracing_enabled():
        print("\nTracing enabled and auth OK.")
    else:
        print("\nAuth OK. Set LANGSMITH_TRACING=true to upload traces.")
    return 0


def main() -> int:
    if not KEY:
        if _tracing_enabled():
            return _fail("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing.")
        print("No LangSmith key configured. Local labs run fine without tracing.")
        return 0

    return _check_auth()


if __name__ == "__main__":
    sys.exit(main())
