"""Download a PDF and extract plain text with pypdf."""

from __future__ import annotations

import io

import httpx
from langchain_core.tools import tool
from pypdf import PdfReader

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MAX_CHARS = 20_000
TIMEOUT_SECONDS = 30.0


@tool
def read_pdf(url: str) -> str:
    """Download a PDF from a URL and extract its text. Prefer this tool for academic papers, SEC filings, and other PDF documents instead of fetch_url."""
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    reader = PdfReader(io.BytesIO(response.content))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    body = "\n\n".join(pages)
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS]
    return f"[PDF Source: {url}]\n{body}"


if __name__ == "__main__":
    sample = read_pdf.invoke(
        {"url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"}
    )
    print(sample[:500])
