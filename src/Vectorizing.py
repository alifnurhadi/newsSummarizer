from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..init import VECTOR_DB, loadNews


def transformation_nightmode():

    raw_data = loadNews()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    docsToCHROMA = []
    for row in data.iter_rows(named=True):
        chunks = splitter.create_documents(
            [row["content_summary"]],
            metadatas=[
                {
                    "source": row["sites"],
                    "type": row["sources"],
                    "keyword": row["keywords"],
                }
            ],
        )
        docsToCHROMA.extend(chunks)

    VECTOR_DB.add_documents(docsToCHROMA)

    print(
        f"Successfully embedded and stored {len(docsToCHROMA)} government chunks to ChromaDB."
    )


if __name__ == "__main__":
    transformation_nightmode()
