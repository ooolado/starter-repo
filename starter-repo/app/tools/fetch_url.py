"""Fetch a URL and return plain-text page content."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MAX_CHARS = 8000
TIMEOUT_SECONDS = 10.0


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its plain-text body. Use this when you have a URL and need the page content."""
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return f"[Source: {url}]\n{text}"


if __name__ == "__main__":
    page = fetch_url.invoke({"url": "https://example.com"})
    print(page[:500])
    print(f"\n... ({len(page)} chars total)")
