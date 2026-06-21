"""Search internal docs via AWS Bedrock Knowledge Bases (with pgvector fallback)."""

from __future__ import annotations

import os
import re
from hashlib import md5

import boto3
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


def _reference_url(location: dict, metadata: dict) -> str:
    loc_type = location.get("type", "")
    if loc_type == "WEB" or "webLocation" in location:
        return str(location.get("webLocation", {}).get("url", ""))
    if loc_type == "S3" or "s3Location" in location:
        return str(location.get("s3Location", {}).get("uri", ""))
    if "confluenceLocation" in location:
        return str(location.get("confluenceLocation", {}).get("url", ""))
    if "sharePointLocation" in location:
        return str(location.get("sharePointLocation", {}).get("url", ""))
    for key in ("source_uri", "source_url", "url", "x-amz-bedrock-kb-source-uri"):
        if key in metadata:
            return str(metadata[key])
    return str(location)


def _search_bedrock_kb(query: str, k: int) -> list[dict]:
    kb_id = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "").strip()
    if not kb_id:
        return []

    region = os.getenv("AWS_REGION", "us-east-1")
    model_arn = os.getenv("BEDROCK_KB_MODEL_ARN", "").strip()
    if not model_arn:
        model_arn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-text-premier-v1:0"

    client = boto3.client("bedrock-agent-runtime", region_name=region)
    response = client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": k},
                },
            },
        },
    )

    seen: set[str] = set()
    hits: list[dict] = []
    idx = 0

    for citation in response.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            content = ref.get("content", {})
            text = str(content.get("text", "")).strip()
            if not text:
                continue
            location = ref.get("location", {})
            metadata = ref.get("metadata") or {}
            source_url = _reference_url(location, metadata) or f"bedrock-kb://{kb_id}/{idx}"
            dedupe_key = f"{source_url}:{text[:120]}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            chunk_id = md5(dedupe_key.encode()).hexdigest()[:16]
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "source_url": source_url,
                    "score": max(0.0, 1.0 - idx * 0.05),
                    "text": text,
                }
            )
            idx += 1
            if len(hits) >= k:
                return hits

    output_text = response.get("output", {}).get("text", "").strip()
    if not hits and output_text:
        hits.append(
            {
                "chunk_id": md5(output_text[:120].encode()).hexdigest()[:16],
                "source_url": f"bedrock-kb://{kb_id}/generated",
                "score": 1.0,
                "text": output_text,
            }
        )

    return hits[:k]


def _search_pgvector(query: str, k: int, table: str) -> list[dict]:
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


@tool
def search_local_docs(query: str, k: int = 5, table: str = "docs") -> list[dict]:
    """Search the ingested document corpus for content relevant to a query. Use this when the user asks about content in our internal documentation. Returns a list of citations each with a real source_url that you MUST cite back."""
    if os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "").strip():
        return _search_bedrock_kb(query, k)
    return _search_pgvector(query, k, table)


if __name__ == "__main__":
    hits = search_local_docs.invoke({"query": "IAM access key rotation", "k": 3})
    print(f"Got {len(hits)} hit(s)")
    for index, hit in enumerate(hits, start=1):
        print(f"{index}. [{hit['score']:.3f}] {hit['source_url']}")
        preview = hit["text"][:120].replace("\n", " ")
        print(f"   {preview}{'...' if len(hit['text']) > 120 else ''}")
