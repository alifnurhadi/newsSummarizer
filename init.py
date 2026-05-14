import json
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup the Local Vector Database inside newsSummarizer
EMBEDDING_prv = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VECTOR_DB = Chroma(
    collection_name="Global_Gov_Data",
    embedding_function=EMBEDDING_prv,
    persist_directory=os.path.join(BASE_DIR, "local_vectordb"),
)

# os.path.dirname(BASE_DIR) moves UP one level to the "Documents" folder
PARENT_DIR = os.path.dirname(BASE_DIR)

# Now we build the path down into the scraper folder
PATH = os.path.join(PARENT_DIR, "Scrape_antaraNews", "data", "raw", "scrapeResult.json")


def loadNews():
    if not os.path.exists(PATH):
        print(f"❌ Error: Could not find the news JSON at: {PATH}")
        return []
    with open(PATH, "r") as rdr:
        try:
            return json.load(rdr)
        except json.JSONDecodeError:
            print("❌ Error: JSON file is corrupted or empty.")
            return []
