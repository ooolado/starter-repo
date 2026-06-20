"""Web search via Tavily with an offline mock when no API key is set."""

from __future__ import annotations

import os

from langchain_core.tools import tool


def _mock_results(query: str) -> list[dict]:
    return [{"title": "mock", "url": "https://example.com", "content": query}]


def _tavily_search(query: str, k: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=k)
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", "")),
        }
        for item in response.get("results", [])
    ]


@tool
def web_search(query: str, k: int = 5) -> list[dict]:
    """Search the public web for pages relevant to a query. Use this when you need current or external information."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _mock_results(query)
    return _tavily_search(query, k)


if __name__ == "__main__":
    hits = web_search.invoke({"query": "agentic AI bootcamp", "k": 3})
    print(f"Got {len(hits)} result(s)")
    for index, hit in enumerate(hits, start=1):
        print(f"{index}. {hit['title']} — {hit['url']}")
        preview = hit["content"][:120].replace("\n", " ")
        print(f"   {preview}{'...' if len(hit['content']) > 120 else ''}")
