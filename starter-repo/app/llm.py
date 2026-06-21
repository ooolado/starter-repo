"""Provider-agnostic chat-model and embeddings factory.

Every Monk Technologies project imports `get_chat_model()` and `get_embeddings()`
from here. Swap Bedrock <-> Vertex by changing `MONK_MODEL` / `MONK_EMBEDDINGS`
in `.env`. Set either to `fake` to use the offline `FakeMonkChatModel` /
`FakeMonkEmbeddings` for local dry-runs without cloud credentials.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from app._fake_llm import (
    FakeMonkChatModel,
    FakeMonkEmbeddings,
    fake_chat_model,
    fake_embeddings,
    is_fake_chat_model,
    is_fake_embeddings,
)

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"
DEFAULT_EMBEDDINGS = "bedrock:amazon.titan-embed-text-v2:0"


def _resolved_chat_name(name: str | None) -> str:
    return (name or os.getenv("MONK_MODEL") or DEFAULT_MODEL).strip()


def _resolved_embedding_name(name: str | None) -> str:
    return (name or os.getenv("MONK_EMBEDDINGS") or DEFAULT_EMBEDDINGS).strip()


@lru_cache(maxsize=8)
def _cached_chat_model(resolved: str, guardrail_key: str) -> Any:
    kwargs: dict[str, Any] = {}
    if guardrail_key:
        guardrail_id, guardrail_version = guardrail_key.split("|", 1)
        kwargs["guardrails"] = {
            "guardrailIdentifier": guardrail_id,
            "guardrailVersion": guardrail_version,
            "trace": "enabled",
        }
    return init_chat_model(resolved, **kwargs)


def get_chat_model(name: str | None = None, **kwargs: Any):
    """Return a LangChain chat model. Reads `MONK_MODEL` when name is None.

    If `MONK_MODEL=fake` (the offline-demo default) we return a deterministic
    `FakeMonkChatModel`; otherwise we hand off to `init_chat_model`.
    """
    resolved = _resolved_chat_name(name)
    if is_fake_chat_model(resolved):
        return fake_chat_model(**kwargs)

    # Google Vertex's equivalent (Model Armor, via VERTEX_MODEL_ARMOR_POLICY) is configured separately on the GCP side and is out of scope here.
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "").strip()
    guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT"
    guardrail_key = ""
    if guardrail_id and resolved.startswith("bedrock") and not is_fake_chat_model(resolved):
        guardrail_key = f"{guardrail_id}|{guardrail_version}"

    if kwargs:
        merged = dict(kwargs)
        if guardrail_key and "guardrails" not in merged:
            gid, gver = guardrail_key.split("|", 1)
            merged["guardrails"] = {
                "guardrailIdentifier": gid,
                "guardrailVersion": gver,
                "trace": "enabled",
            }
        return init_chat_model(resolved, **merged)

    return _cached_chat_model(resolved, guardrail_key)


@lru_cache(maxsize=4)
def get_embeddings(name: str | None = None, **kwargs: Any):
    """Return a LangChain embeddings model. Reads `MONK_EMBEDDINGS` when name is None.

    If `MONK_EMBEDDINGS=fake` we return deterministic hashed 1024-dim vectors so
    pgvector ingest + search work without a real embeddings API. Otherwise we
    defer to `init_embeddings`.
    """
    resolved = _resolved_embedding_name(name)
    if is_fake_embeddings(resolved):
        return fake_embeddings(**kwargs)
    return init_embeddings(resolved, **kwargs)


__all__ = [
    "DEFAULT_EMBEDDINGS",
    "DEFAULT_MODEL",
    "FakeMonkChatModel",
    "FakeMonkEmbeddings",
    "get_chat_model",
    "get_embeddings",
]
