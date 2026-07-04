"""
CLI entrypoint.

Usage:
    python main.py "your research topic here"

Requires GEMINI_API_KEY and TAVILY_API_KEY set as env vars (see .env.example).
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph  

def run(topic: str):
    app = build_graph()

    initial_state = {
        "topic": topic,
        "queries": [],
        "search_results": [],
        "validated_sources": [],
        "report": "",
        "errors": [],
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(final_state["report"])
    print("=" * 60)

    if final_state.get("errors"):
        print(f"\n({len(final_state['errors'])} non-fatal errors — see above)")

    with open("report.md", "w") as f:
        f.write(final_state["report"])
    print("\nSaved to report.md")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research topic"')
        sys.exit(1)

    run(" ".join(sys.argv[1:]))