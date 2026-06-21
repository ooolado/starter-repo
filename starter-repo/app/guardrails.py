"""Bedrock Guardrail helpers and citation validation."""

from __future__ import annotations

import os
import re

import boto3

_MD_URL_RE = re.compile(r"\[.*?\]\((https?://[^\s\)]+)\)")
_BARE_URL_RE = re.compile(r"(https?://[^\s\)>\"']+)")


def extract_urls(text: str) -> set[str]:
    """Find all URLs in markdown [text](url) patterns and bare https://... patterns."""
    urls = set(_MD_URL_RE.findall(text))
    urls |= set(_BARE_URL_RE.findall(text))
    return {url.rstrip(".,;:") for url in urls}


def validate_citations(report: str, allowed_urls: set[str]) -> tuple[bool, list[str]]:
    """Check that every URL in the report is in the allowed set.

    Returns (ok, bad_urls) where ok=True means all citations are valid.
    """
    found_urls = extract_urls(report)
    bad = [url for url in found_urls if url not in allowed_urls]
    return (len(bad) == 0, bad)


def check_input_guardrail(text: str) -> tuple[bool, str]:
    """Return (blocked, message) for raw user input via Bedrock ApplyGuardrail."""
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "").strip()
    if not guardrail_id:
        return False, ""

    model = os.getenv("MONK_MODEL", "").strip()
    if not model.startswith("bedrock") or model == "fake":
        return False, ""

    version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT"
    region = os.getenv("AWS_REGION", "us-east-1")

    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=version,
        source="INPUT",
        content=[{"text": {"text": text}}],
    )

    if resp.get("action") == "GUARDRAIL_INTERVENED":
        outputs = resp.get("outputs") or []
        if outputs and isinstance(outputs[0], dict):
            return True, str(outputs[0].get("text", ""))
        return True, "Sorry — this request is not allowed."

    return False, ""
