import json
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict

from langchain_community.chat_models import ChatOllama

# LangChain & LangGraph
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .init import VECTOR_DB, EMBEDDING_prv

llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.1, format="json")

emb = EMBEDDING_prv
vctrDB = VECTOR_DB


with open("/Users/alif/Documents/newsSummarizer/data/ScrapeResult.json", "r") as read:
    latest_news = json.load(read)


class stateManagement(TypedDict):
    summary: List[str]
    keyTopic: List[str]


def summarizedTopics(state: stateManagement, news: str, model=llm) -> stateManagement:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Summarize the business event and output a statement into ( 1/10 from length of the news),
                    extract single core keyword tag
                    Format strictly as json below:
                    {{
                    "summary": 'your Summary'
                    "topic": 'your tag or default to "general"'
                    }}
                """,
            ),
            ("human", "{news}"),
        ]
    )

    response = (prompt | model).invoke({"news": news}).content

    return state(response["summary"], response["topic"])


def hybrid_retrieve_node(state: stateManagement) -> stateManagement:

    # The WHERE clause: Only search documents tagged with this keyword
    search_filter = {"keyword": state["extracted_topic_keyword"]}

    # The Vector Search: Find semantically similar chunks within the filtered subset
    docs = vector_store.similarity_search(
        query=state["daily_recap"], k=2, filter=search_filter
    )

    formatted_laws = [
        {"content": d.page_content, "source": d.metadata.get("source")} for d in docs
    ]
    return {"raw_relevant_laws": formatted_laws}


def extract_essence_node(state: stateManagement) -> stateManagement:

    if not state["raw_relevant_laws"]:
        return {
            "legal_essence": "No highly relevant government watchlists found for this specific event."
        }

    laws_text = "\n".join(
        [f"[{l['source']}] {l['content']}" for l in state["raw_relevant_laws"]]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only the specific regulatory constraints from these laws that directly impact the following business recap. Ignore boilerplate.",
            ),
            ("human", "Recap: {recap}\n\nLaws:\n{laws}"),
        ]
    )

    response = (prompt | llm).invoke({"recap": state["daily_recap"], "laws": laws_text})
    return {"legal_essence": response.content}


def synthesize_report_node(state: stateManagement) -> stateManagement:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strategic advisor. Write a concise operational report based on today's events and regulatory constraints. Suggest 2 business scenarios/actions. Cite the provided laws.",
            ),
            (
                "human",
                """
                Stories :

            [ syntesise of news {recap} and laws {laws} that are match and contradict that impact the stakeholder in news result ] +
            make it as short as 500 character and maximum of 1000 character.

            What could we do :

            [ based on synthesise result, gave number of suggestion for stakeholder in the news affected and for the general business owner ] +
            Make it in bullet points.""",
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {"recap": state["daily_recap"], "laws": state["legal_essence"]}
    )
    return {"final_advisory": response.content}


workflow = StateGraph(stateManagement)

workflow.add_node("recap_and_tag", summarizedTopics)
workflow.add_node("hybrid_retrieve", hybrid_retrieve_node)
workflow.add_node("extract_essence", extract_essence_node)
workflow.add_node("synthesize", synthesize_report_node)

workflow.set_entry_point("recap_and_tag")
workflow.add_edge("recap_and_tag", "hybrid_retrieve")
workflow.add_edge("hybrid_retrieve", "extract_essence")
workflow.add_edge("extract_essence", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()

if __name__ == "__main__":
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    final_state = app.invoke({"target_date": yesterday})
