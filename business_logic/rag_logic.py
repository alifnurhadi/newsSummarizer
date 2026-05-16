import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from init import VECTOR_DB

# Config: JSON-mode for extraction, Text-mode for advisory
llm_json = ChatOllama(model="llama3:latest", temperature=0.1, format="json")
llm_text = ChatOllama(model="llama3:latest", temperature=0.2)


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
    MAX_CHARS = 24000
    news_corpus = ""
    articles_processed = 0

    for n in state["daily_private_news"]:
        link = n.get("link", "Unknown URL")
        content = n.get("content", "")

        # Format the individual article
        article_text = f"URL: {link}\nNews: {content}\n\n"

        # Check if adding this article exceeds our safe token/character limit
        if len(news_corpus) + len(article_text) > MAX_CHARS:
            print(
                f"  ⚠️ [Agent Warning] Context limit reached! Truncated processing to {articles_processed} articles to prevent OOM crash."
            )
            break

        news_corpus += article_text
        articles_processed += 1

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
    clean_response = response.strip().strip("```json").strip("```")
    data = json.loads(clean_response)

    return {
        "daily_recap": data.get("recap", "No summary generated"),
        "extracted_topic_keyword": data.get("keyword", "general"),
    }


def agentic_retrieve_node(state: ReportState) -> ReportState:
    """Node 2: Autonomous query generation and self-correcting retrieval."""
    print("--- NODE 2: Agentic Retrieval ---")
    recap = state["daily_recap"]

    # Agent formulates the initial semantic query
    query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a legal research agent. Read the news recap and formulate ONE precise semantic search query to find relevant Indonesian regulations or laws. Output ONLY the query string. No preamble or quotes.",
            ),
            ("human", "{recap}"),
        ]
    )

    initial_query = (
        (query_prompt | llm_text).invoke({"recap": recap}).content.strip(" \"'")
    )
    print(f"  [Agent] Initial Query: '{initial_query}'")

    search_filter = {"keyword": state["extracted_topic_keyword"]}

    # First search
    docs = VECTOR_DB.similarity_search(query=initial_query, k=3, filter=search_filter)

    # Execute first search (pure semantic, no metadata filters)
    docs = VECTOR_DB.similarity_search(query=initial_query, k=3)
    formatted_laws = [
        {"content": d.page_content, "source": d.metadata.get("source", "Unknown")}
        for d in docs
    ]

    # Agent evaluates the retrieval quality
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Evaluate if these retrieved laws directly govern the events in the news recap.
        Respond strictly in JSON format:
        {{
            "is_relevant": true or false,
            "better_query": "A new, highly specific search query string if false, or null if true"
        }}""",
            ),
            ("human", "News: {recap}\n\nRetrieved Laws: {laws}"),
        ]
    )

    eval_response = (
        (eval_prompt | llm_json)
        .invoke({"recap": recap, "laws": json.dumps(formatted_laws)})
        .content
    )

    # Parse evaluation and optionally retry (Self-Correction Loop)
    try:
        clean_eval = eval_response.strip().strip("```json").strip("```")
        evaluation = json.loads(clean_eval)
    except json.JSONDecodeError:
        print("  [Agent] Evaluation parsing failed, proceeding with initial results.")
        evaluation = {"is_relevant": True}

    better_query = evaluation.get("better_query")

    if (
        not evaluation.get("is_relevant")
        and better_query
        and better_query.lower() != "null"
    ):
        print(f"  [Agent] Self-Correction Triggered. Retrying with: '{better_query}'")

        new_query = evaluation["better_query"]

        # Execute secondary search
        retry_docs = VECTOR_DB.similarity_search(
            query=new_query, k=3, filter=search_filter
        )

        formatted_laws = [
            {"content": d.page_content, "source": d.metadata.get("source", "Unknown")}
            for d in retry_docs
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

    output_dir = "data/result"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "Report.txt")
    with open(output_file, "w", encoding="utf-8") as wr:
        wr.write(response.content)

    return {"final_advisory": response.content}


# --- ASSEMBLY ---

workflow = StateGraph(ReportState)
workflow.add_node("recap", summarizedTopics)
workflow.add_node("retrieve", agentic_retrieve_node)
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

if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from init import loadNews

    all_news = loadNews()
    # (Add your date filtering logic here)

    if all_news:
        final_state = app.invoke(
            {
                "target_date": "2026-05-12",  # or dynamic date
                "daily_private_news": all_news,
            }
        )
        print(final_state["final_advisory"])
