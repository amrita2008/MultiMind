"""
Streamlit interface for the research agent.

Run locally with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph  # noqa: E402

st.set_page_config(page_title="MultiMind", page_icon="🔎", layout="centered")

st.title("🔎 MultiMind")
st.caption(
    "Multi-agent pipeline: Planner → Search → Validate & Extract → Synthesizer. "
    "Built with LangGraph + Gemini Flash + Tavily."
)

topic = st.text_input(
    "Research topic",
    placeholder="e.g. impact of AI on entry-level software jobs",
)

run_clicked = st.button("Run research", type="primary", disabled=not topic.strip())

if run_clicked:
    progress_placeholder = st.empty()
    progress_placeholder.info("Planning search queries...")

    app = build_graph()
    initial_state = {
        "topic": topic,
        "queries": [],
        "search_results": [],
        "validated_sources": [],
        "report": "",
        "errors": [],
    }

    try:
        with st.spinner("Researching — this can take 30-90 seconds..."):
            final_state = app.invoke(initial_state)

        progress_placeholder.empty()

        st.success("Done!")

        with st.expander("Search queries used"):
            for q in final_state["queries"]:
                st.write(f"- {q}")

        with st.expander(
            f"Sources found ({len(final_state['validated_sources'])})"
        ):
            for s in final_state["validated_sources"]:
                st.write(f"**{s['title']}** ({s['relevance_score']:.2f}) — {s['url']}")

        st.markdown("---")
        st.markdown(final_state["report"])

        st.download_button(
            "Download report as Markdown",
            data=final_state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

        if final_state.get("errors"):
            with st.expander(f"⚠️ {len(final_state['errors'])} non-fatal warnings"):
                for e in final_state["errors"]:
                    st.write(f"- {e}")

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Pipeline failed: {e}")