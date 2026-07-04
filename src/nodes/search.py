"""
Search node: runs every query from the planner through Tavily search and
collects the results into flat list on state['search_results'].
"""

import os
from tavily import TavilyClient

from src.state import ResearchState

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

RESULTS_PER_QUERY = 4


def search_node(state: ResearchState) -> ResearchState:
    all_results = []

    for query in state["queries"]:
        try:
            response = client.search(
                query=query,
                max_results=RESULTS_PER_QUERY,
                search_depth="basic",
            )
            for r in response.get("results", []):
                all_results.append(
                    {
                        "query": query,
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                )
        except Exception as e:
            state.setdefault("errors", []).append(
                f"search_node failed for query '{query}': {e}"
            )
    seen = set()
    deduped = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    state["search_results"] = deduped
    return state