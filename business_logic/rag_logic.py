import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from ..init import VECTOR_DB

# Config: JSON-mode for extraction, Text-mode for advisory
llm_json = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.1, format="json")
llm_text = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.2)


class ReportState(TypedDict):
    target_date: str
    daily_private_news: List[Dict]
    daily_recap: str
    extracted_topic_keyword: str
    raw_relevant_laws: List[Dict]
    legal_essence: str
    final_advisory: str


def summarizedTopics(state: ReportState) -> ReportState:
    """Node 1: Compresses multiple news items into a single context."""
    print(f"--- NODE 1: Recapping {len(state['daily_private_news'])} News Items ---")

    # Batch news items into a single string for analysis
    news_corpus = "\n\n".join(
        [f"URL: {n['link']}\nNews: {n['content']}" for n in state["daily_private_news"]]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strategic analyst. Summarize these events into a
        single cohesive recap (1/10 length) and extract ONE core regulatory keyword tag.

        Respond ONLY in valid JSON:
        {{
            "recap": "summary text",
            "keyword": "tag_name"
        }}""",
            ),
            ("human", "{news}"),
        ]
    )

    response = (prompt | llm_json).invoke({"news": news_corpus}).content
    data = json.loads(response)  #

    return {
        "daily_recap": data.get("recap", "No summary generated"),
        "extracted_topic_keyword": data.get("keyword", "general"),
    }


def hybrid_retrieve_node(state: ReportState) -> ReportState:
    """Node 2: Fixed variable naming error (vctrDB)."""
    print(
        f"--- NODE 2: Hybrid Retrieval (Filter: {state['extracted_topic_keyword']}) ---"
    )

    search_filter = {"keyword": state["extracted_topic_keyword"]}  #

    # Fixed: Changed vector_store to VECTOR_DB from init.py
    docs = VECTOR_DB.similarity_search(
        query=state["daily_recap"], k=2, filter=search_filter
    )

    formatted_laws = [
        {"content": d.page_content, "source": d.metadata.get("source")} for d in docs
    ]
    return {"raw_relevant_laws": formatted_laws}


def extract_essence_node(state: ReportState) -> ReportState:
    """Node 3: Legal Essence Extraction."""
    print("--- NODE 3: Distilling Legal Essence ---")
    if not state["raw_relevant_laws"]:
        return {"legal_essence": "No applicable government laws found."}

    laws_text = "\n".join(
        [f"[{l['source']}] {l['content']}" for l in state["raw_relevant_laws"]]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract core regulatory constraints from these laws related to the news. No boilerplate.",
            ),
            ("human", "Recap: {recap}\n\nLaws:\n{laws}"),
        ]
    )

    response = (prompt | llm_text).invoke(
        {"recap": state["daily_recap"], "laws": laws_text}
    )
    return {"legal_essence": response.content}


def synthesize_report_node(state: ReportState) -> ReportState:
    """Node 4: Wealth Management Drafting."""
    print("--- NODE 4: Drafting Advisory ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strategic wealth advisor. Write a narrative-style advisory citing laws.",
            ),
            ("human", "Event: {recap}\n\nRegulatory Core: {laws}"),
        ]
    )
    response = (prompt | llm_text).invoke(
        {"recap": state["daily_recap"], "laws": state["legal_essence"]}
    )
    return {"final_advisory": response.content}


def review_refine_node(state: ReportState) -> ReportState:
    """Node 5: AI Self-Correction."""
    print("--- NODE 5: Review & Refine (AI Editor) ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Review the draft report. Polishing for elite professional tone and legal accuracy.",
            ),
            ("human", "Draft: {draft}"),
        ]
    )
    response = (prompt | llm_text).invoke({"draft": state["final_advisory"]})
    return {"final_advisory": response.content}


# --- ASSEMBLY ---

workflow = StateGraph(ReportState)
workflow.add_node("recap", summarizedTopics)
workflow.add_node("retrieve", hybrid_retrieve_node)
workflow.add_node("essence", extract_essence_node)
workflow.add_node("synthesize", synthesize_report_node)
workflow.add_node("review", review_refine_node)

workflow.set_entry_point("recap")
workflow.add_edge("recap", "retrieve")
workflow.add_edge("retrieve", "essence")
workflow.add_edge("essence", "synthesize")
workflow.add_edge("synthesize", "review")
workflow.add_edge("review", END)

app = workflow.compile()
