# 🔎 Multi-Mind

A multi-agent AI research pipeline that autonomously plans search queries, gathers and validates sources, and synthesizes a fully cited research report — built with LangGraph and Gemini Flash.

**🚀 Live demo:** https://huggingface.co/spaces/amrita24/MultiMind
---

## Overview

Give Multi-Mind a topic, and it runs a 4-stage agent pipeline to produce a structured, cited Markdown report — no manual searching, reading, or note-taking required.

Topic → Planner → Search → Validate & Extract → Synthesizer → Report

## How It Works

| Stage | What it does |
|---|---|
| **Planner** | Breaks the topic into 4–6 targeted search queries covering different angles — background, recent developments, data/statistics, expert opinions |
| **Search** | Runs each query through the Tavily search API, deduplicates results by URL |
| **Validate & Extract** | Scores every source's relevance to the topic in a single batched Gemini call, then extracts full text from sources above the relevance threshold (HTML via BeautifulSoup, PDFs via PyMuPDF) |
| **Synthesizer** | Writes a structured Markdown report grouped by sub-theme, with inline [n] citations and a numbered source list |

Each stage has error handling with graceful fallbacks — a blocked fetch or a malformed model response gets logged and skipped rather than crashing the whole run.

## Tech Stack

- **LangGraph** — orchestrates the pipeline as a stateful agent graph
- **Gemini Flash** — query planning, relevance scoring, and report synthesis
- **Tavily** — web search API
- **BeautifulSoup / PyMuPDF** — full-text extraction from HTML and PDFs
- **Streamlit** — interactive front-end
- **Hugging Face Spaces** — deployment

## Running Locally

git clone https://github.com/amrita2008/MultiMind.git

cd MultiMind

python -m venv venv

source venv/bin/activate      # venv\Scripts\activate on Windows

pip install -r requirements.txt

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_key

TAVILY_API_KEY=your_tavily_key

Run the CLI:

python main.py "your research topic"

Or run the Streamlit app:

streamlit run app.py

## Design Notes

- **Batched relevance scoring** — all sources are scored in a single LLM call per run instead of one call per source, cutting latency and API cost significantly.
- **Selective extraction** — low-relevance sources skip full-text fetching entirely, since they're unlikely to be cited anyway.
- **Graceful degradation** — every stage has a fallback path (e.g. a blocked fetch falls back to the search snippet) so one bad source never takes down the full pipeline run.

## Getting API Keys

- **Gemini**: https://aistudio.google.com/apikey (free tier)
- **Tavily**: https://app.tavily.com (free tier, 1,000 searches/month)

---

Built by [Amrita](https://github.com/amrita2008)
