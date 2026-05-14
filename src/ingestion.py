import argparse
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the existing database setup from your init.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from init import VECTOR_DB


def extract_text_from_txt(txt_path: str) -> str:
    """Reads raw text directly from a .txt file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
        return ""


def ingest_law_to_db(file_path: str, source_name: str, keyword: str):
    """
    Reads a law (TXT), chunks it, and stores it in the Vector DB.
    """
    # Read the text
    print(f"Detected TXT. Reading raw text from '{file_path}'...")
    raw_text = extract_text_from_txt(file_path)

    if not raw_text.strip():
        print("❌ No text could be extracted. Skipping.")
        return

    # Chunking the text
    print("Chunking text for optimal RAG retrieval...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    # Set the metadata so the hybrid search in rag_logic.py can find it
    metadata = {"source": source_name, "type": "government", "keyword": keyword}

    chunks = splitter.create_documents([raw_text], metadatas=[metadata])
    print(f"Created {len(chunks)} semantic chunks.")

    # Store in Vector Database
    try:
        VECTOR_DB.add_documents(chunks)
        print(f"✅ Successfully embedded and saved '{source_name}' to ChromaDB!")
    except Exception as e:
        print(f"❌ Failed to save to database: {e}")


if __name__ == "__main__":
    # Create the folder for raw laws if it doesn't exist
    os.makedirs("data/raw_laws", exist_ok=True)

    # Define the mock laws we want to create and ingest
    parser = argparse.ArgumentParser(
        description="Ingest regulatory text documents into the local ChromaDB vector store."
    )

    # Define the required arguments
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .txt file (e.g., data/raw_laws/mock_uu_p2sk.txt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Human-readable name of the law (e.g., 'UU P2SK - Fintech')",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="The metadata tag for hybrid search (e.g., 'wealth_management', 'capital_markets')",
    )

    # Parse the arguments from the terminal
    args = parser.parse_args()

    # Run the ingestion function using the provided arguments
    ingest_law_to_db(file_path=args.file, source_name=args.source, keyword=args.keyword)
