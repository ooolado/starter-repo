"""Vector search over ingested documents in Postgres/pgvector."""

from __future__ import annotations

import os
import re

import psycopg
from langchain.embeddings import init_embeddings
from langchain_core.tools import tool

DEFAULT_EMBEDDINGS = "bedrock:amazon.titan-embed-text-v2:0"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/monk"


def _sanitize_table(table: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"invalid table name: {table!r}")
    return table


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


@tool
def search_local_docs(query: str, k: int = 5, table: str = "docs") -> list[dict]:
    """Search the ingested document corpus for content relevant to a query. Use this when the user asks about content in our internal documentation. Returns a list of citations each with a real source_url that you MUST cite back."""
    safe_table = _sanitize_table(table)
    dsn = os.getenv("POSTGRES_DSN", DEFAULT_DSN)
    model_name = os.getenv("MONK_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    query_vec = _vector_literal(init_embeddings(model_name).embed_query(query))

    sql = f"""
        SELECT chunk_id, source_url, 1 - (embedding <=> %s::vector) AS score, text
        FROM {safe_table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (query_vec, query_vec, k))
        rows = cur.fetchall()

    return [
        {
            "chunk_id": str(chunk_id),
            "source_url": str(source_url),
            "score": float(score),
            "text": str(text),
        }
        for chunk_id, source_url, score, text in rows
    ]


if __name__ == "__main__":
    hits = search_local_docs.invoke({"query": "IAM access key rotation", "k": 3})
    print(f"Got {len(hits)} hit(s)")
    for index, hit in enumerate(hits, start=1):
        print(f"{index}. [{hit['score']:.3f}] {hit['source_url']}")
        preview = hit["text"][:120].replace("\n", " ")
        print(f"   {preview}{'...' if len(hit['text']) > 120 else ''}")
