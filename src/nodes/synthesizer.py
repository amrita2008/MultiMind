"""
Synthesizer node: takes state['validated_sources'] and produces a final
Markdown report via Gemini Flash — executive summary, sub-theme sections,
inline [n] citations, and a numbered source list at the bottom.

Only sources at/above CITE_THRESHOLD are sent to the model, keeping the
prompt compact and cutting out noise the earlier node already flagged as
low relevance.
"""

import json
import os
import google.generativeai as genai

from src.state import ResearchState

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

CITE_THRESHOLD = 0.5
MAX_SOURCES_TO_SYNTHESIZE = 10
FULL_TEXT_PREVIEW_CHARS = 2000  # per-source cap fed into the synthesis prompt

SYNTHESIS_PROMPT = """You are a research analyst writing a report.

Topic: {topic}

You have {n} sources below, each with an index number. Write a well \
organized Markdown report:

1. Start with a 2-3 sentence executive summary.
2. Organize the body into sections by sub-theme (NOT one section per \
   source) — group related findings together.
3. Whenever you state a fact drawn from a source, cite it inline as \
   [n] using that source's index number.
4. Keep it factual — do not add information beyond what the sources \
   support.
5. End with a "## Sources" section listing every source as \
   "[n] Title — URL".

Sources:
{sources_json}

Respond with ONLY the Markdown report, no preamble, no code fences.
"""


def _build_source_payload(sources: list) -> list:
    payload = []
    for i, s in enumerate(sources, 1):
        payload.append(
            {
                "index": i,
                "title": s["title"],
                "url": s["url"],
                "key_findings": s.get("key_findings", []),
                "excerpt": (s.get("full_text") or "")[:FULL_TEXT_PREVIEW_CHARS],
            }
        )
    return payload


def synthesizer_node(state: ResearchState) -> ResearchState:
    sources = [
        s
        for s in state["validated_sources"]
        if s.get("relevance_score", 0) >= CITE_THRESHOLD
    ][:MAX_SOURCES_TO_SYNTHESIZE]

    if not sources:
        state["report"] = (
            f"# Research Report: {state['topic']}\n\n"
            "No sufficiently relevant sources were found. Try broadening "
            "the topic or checking the search/validation steps."
        )
        return state

    payload = _build_source_payload(sources)
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = SYNTHESIS_PROMPT.format(
        topic=state["topic"],
        n=len(payload),
        sources_json=json.dumps(payload, indent=2),
    )

    try:
        response = model.generate_content(prompt)
        report = response.text.strip().replace("```markdown", "").replace("```", "").strip()
    except Exception as e:
        state.setdefault("errors", []).append(f"synthesizer_node failed: {e}")
        # fallback: bare source list so the run doesn't produce nothing
        lines = [f"# Research Report: {state['topic']}\n", "## Sources\n"]
        for p in payload:
            lines.append(f"[{p['index']}] {p['title']} — {p['url']}")
        report = "\n".join(lines)

    if state.get("errors"):
        report += "\n\n## Notes\n" + "\n".join(f"- {e}" for e in state["errors"])

    state["report"] = report
    return state