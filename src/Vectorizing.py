import os
import sys

import polars as pl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..init import VECTOR_DB, loadNews  # Import your cleaning logic

# Ensure we can import init correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def transformation_nightmode():
    raw_json_data = loadNews()
    if not raw_json_data:
        print("No data found to vectorize.")
        return

    df = pl.DataFrame(raw_json_data)

    # If the scraper added a 'sources' column to mark government data, filter by it.
    # Otherwise, process everything as general legal context.
    if "sources" in df.columns:
        df_clean = df.filter(pl.col("sources") == "government")
    else:
        print("Warning: 'sources' column missing. Vectorizing available content...")
        df_clean = df

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docsToCHROMA = []

    for row in df_clean.iter_rows(named=True):
        # Use .get() to prevent KeyErrors if the schema changes
        content = row.get("content", "")
        if not content:
            continue

        chunks = splitter.create_documents(
            [content],
            metadatas=[
                {
                    "source": row.get("link", "Unknown Source"),
                    "keyword": row.get("keywords", "general"),
                }
            ],
        )
        docsToCHROMA.extend(chunks)

    if docsToCHROMA:
        VECTOR_DB.add_documents(docsToCHROMA)
        print(
            f"✅ Successfully embedded and stored {len(docsToCHROMA)} chunks to ChromaDB."
        )
    else:
        print("❌ No valid documents found to vectorize.")


if __name__ == "__main__":
    transformation_nightmode()
