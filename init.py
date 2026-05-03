import json
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_prv = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

VECTOR_DB = Chroma(
    collection_name="Global_Gov_Data",
    embedding_function=EMBEDDING_prv,
    persist_directory="./vectordb",
)


PATH = r"/Users/alif/Documents/newsSummarizer/data/scrapeResult.json"


def loadNews():

    existPath = os.path.exists(PATH)

    if not existPath:
        print("log creating initial json data")  # this logging part of whole process
        formatJSON = {
            "sites": [],
            "titles": [],
            "content_summary": [],
            "sources": [],
            "keywords": [],
            "dates": [],
        }
        with open(PATH, "w") as wr:
            json.dump(formatJSON, wr, indent=4)

        with open(PATH, "r") as rdr:
            return json.load(rdr)

    with open(PATH, "r") as rdr:
        return json.load(rdr)


if __name__ == "__main__":
    print("initialiazing")
