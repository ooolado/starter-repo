"""Demo: read chunks → embed → cosine similarity."""

from __future__ import annotations

import math
from pathlib import Path

from langchain.embeddings import init_embeddings

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data/sample-corpus/aws-docs/iam-rotation.md"
chunks = [c.strip() for c in DOC.read_text(encoding="utf-8").split("\n\n") if c.strip()][:2]

model = init_embeddings("bedrock:amazon.titan-embed-text-v2:0")
vec_a, vec_b = model.embed_documents(chunks)

dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=True))
norm = math.sqrt(sum(x * x for x in vec_a)) * math.sqrt(sum(x * x for x in vec_b))
print(f"Chunk A: {chunks[0][:70]}...")
print(f"Chunk B: {chunks[1][:70]}...")
print(f"Cosine similarity: {dot / norm:.4f}")
