import json
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# set BASE path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize lightweight multilingual embedding model
EMBEDDING_prv = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Persistent Vector DB
VECTOR_DB = Chroma(
    collection_name="Global_Gov_Data",
    embedding_function=EMBEDDING_prv,
    persist_directory=os.path.join(BASE_DIR, "local_vectordb"),
)

PATH = os.path.join(BASE_DIR, "data", "raw/scrapeResult.json")


def loadNews():
    """Loads the daily news batch. Returns an empty list if file not found."""

    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
        print(f"Log: No existing data found at {PATH}")  #
        return []

    with open(PATH, "r") as rdr:
        try:
            return json.load(rdr)  #
        except json.JSONDecodeError:
            return []
