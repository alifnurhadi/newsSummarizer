import polars as pl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..init import VECTOR_DB, loadNews
from .cleaning import CleanFilterData  # Import your cleaning logic


def transformation_nightmode():
    #   Load raw data
    raw_json_data = loadNews()

    if not raw_json_data:
        print("No data found to vectorize.")
        return

    # Clean and filter for government-only data using Polars
    df_clean = CleanFilterData(raw_json_data)

    # Initialize the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    docsToCHROMA = []

    for row in df_clean.iter_rows(named=True):
        chunks = splitter.create_documents(
            [row["content"]],
            metadatas=[
                {
                    "source": row["link"],
                    "type": row.get(
                        "sources", "news"
                    ),  # FIXED: Replaced row["sources"]
                    "keyword": row.get("keywords", "general"),
                }
            ],
        )
        docsToCHROMA.extend(chunks)

    # 5. Store in ChromaDB
    if docsToCHROMA:
        VECTOR_DB.add_documents(docsToCHROMA)
        print(
            f"Successfully embedded and stored {len(docsToCHROMA)} government chunks to ChromaDB."
        )
    else:
        print("No new government documents found to vectorize.")


if __name__ == "__main__":
    transformation_nightmode()
