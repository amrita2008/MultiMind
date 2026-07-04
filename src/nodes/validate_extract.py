"""
Validate + Extract node.

Step 1: One batched Gemini call scores every search result's relevance to
the topic (0-1) and pulls out 1-3 key findings from the snippet. Batching
avoids N separate LLM calls.

Step 2: For sources above RELEVANCE_THRESHOLD, fetch the full page and
extract text — PyMuPDF for PDFs, BeautifulSoup for HTML. Skips fetch
entirely for low-relevance sources to save time.

Step 3: Truncate full_text so downstream synthesizer prompts stay small.
"""

import json
import os
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import google.generativeai as genai

from src.state import ResearchState

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

RELEVANCE_THRESHOLD = 0.5
MAX_CHARS = 6000
FETCH_TIMEOUT = 10

SCORING_PROMPT = """You are scoring search results for relevance to a research topic.

Topic: {topic}

For each source below, give a relevance score from 0.0 to 1.0 and up to 3 \
short key findings (each under 20 words) drawn ONLY from the snippet text \
provided — do not invent facts.

Sources:
{sources_json}

Respond with ONLY a JSON array, same order as input, like:
[{{"relevance_score": 0.8, "key_findings": ["finding one", "finding two"]}}, ...]
No markdown fences, no extra text.
"""


def _score_sources(topic: str, results: list) -> list:
    """Batched relevance scoring. Returns list aligned with `results`."""
    if not results:
        return []

    sources_for_prompt = [
        {"title": r["title"], "url": r["url"], "snippet": r["content"][:500]}
        for r in results
    ]

    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = SCORING_PROMPT.format(
        topic=topic, sources_json=json.dumps(sources_for_prompt, indent=2)
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
        if len(scores) != len(results):
            raise ValueError(
                f"Expected {len(results)} scores, got {len(scores)}"
            )
        return scores
    except Exception:
        # fallback: treat everything as mid-relevance so nothing is silently dropped
        return [{"relevance_score": 0.5, "key_findings": []} for _ in results]


def _extract_full_text(url: str) -> str:
    """Fetch and extract readable text from a URL (HTML or PDF)."""
    resp = requests.get(
        url,
        timeout=FETCH_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (research-agent bot)"},
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        doc = fitz.open(stream=resp.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def validate_extract_node(state: ResearchState) -> ResearchState:
    results = state["search_results"]
    scores = _score_sources(state["topic"], results)

    validated = []
    for result, score in zip(results, scores):
        relevance = float(score.get("relevance_score", 0.5))
        entry = {
            "url": result["url"],
            "title": result["title"],
            "relevance_score": relevance,
            "key_findings": score.get("key_findings", []),
            "full_text": result["content"],  # fallback if fetch fails/skipped
        }

        if relevance >= RELEVANCE_THRESHOLD:
            try:
                full_text = _extract_full_text(result["url"])
                if full_text.strip():
                    entry["full_text"] = full_text[:MAX_CHARS]
            except Exception as e:
                state.setdefault("errors", []).append(
                    f"validate_extract_node: failed to fetch {result['url']}: {e}"
                )
                # keep the snippet fallback already in full_text

        validated.append(entry)

    # highest relevance first, makes the synthesizer's job easier
    validated.sort(key=lambda s: s["relevance_score"], reverse=True)

    state["validated_sources"] = validated
    return state