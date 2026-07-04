"""
Shared state object passed between every node in the LangGraph pipeline.

Flow: topic -> queries -> search_results -> validated_sources -> report
"""

from typing import TypedDict, List, Dict, Optional


class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    content: str  


class ValidatedSource(TypedDict):
    url: str
    title: str
    relevance_score: float  
    key_findings: List[str]
    full_text: Optional[str]  


class ResearchState(TypedDict):
    topic: str
    queries: List[str]
    search_results: List[SearchResult]
    validated_sources: List[ValidatedSource]
    report: str
    errors: List[str] 