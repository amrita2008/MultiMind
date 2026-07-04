"""
Planner node: takes the research topic and breaks it into 4-6 targeted
search queries using Gemini Flash. Forces JSON output so downstream nodes
don't have to parse free text.
"""

import json
import os
import google.generativeai as genai

from src.state import ResearchState

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PLANNER_PROMPT = """You are a research planning assistant. Given a topic, \
produce 4 to 6 focused, non-overlapping web search queries that together \
would give a thorough understanding of the topic. Cover different angles: \
definitions/background, recent developments, data/statistics, expert \
opinions or controversies, and practical implications where relevant.

Topic: {topic}

Respond with ONLY a JSON array of strings, nothing else. No markdown fences.
Example: ["query one", "query two", "query three"]
"""


def planner_node(state: ResearchState) -> ResearchState:
    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = PLANNER_PROMPT.format(topic=state["topic"])

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw)

        if not isinstance(queries, list) or not queries:
            raise ValueError("Planner did not return a non-empty list")

        state["queries"] = [str(q) for q in queries][:6]

    except Exception as e:
        state.setdefault("errors", []).append(f"planner_node fallback: {e}")
        state["queries"] = [state["topic"]]

    return state