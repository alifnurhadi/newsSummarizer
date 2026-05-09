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


PATH = r"/Users/alif/Documents/newsSummarizer/data/ScrapeResult.json"


class ScrapeConfig:
    def __init__(self) -> None:

        self.BASE_URL = "https://www.antaranews.com"
        self.LIST_URL = f"{self.BASE_URL}/ekonomi/bisnis"
        self.HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        self.REQUEST_TIMEOUT = 20  # seconds per request
        self.DELAY_BETWEEN = 1.0  # seconds between detail fetches (polite crawl)
        self.MAX_LIST_PAGES = 5  # safety cap on pagination

        self.ID_MONTHS = {
            "januari": 1,
            "februari": 2,
            "maret": 3,
            "april": 4,
            "mei": 5,
            "juni": 6,
            "juli": 7,
            "agustus": 8,
            "september": 9,
            "oktober": 10,
            "november": 11,
            "desember": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "agu": 8,
            "sep": 9,
            "okt": 10,
            "nov": 11,
            "des": 12,
        }
        pass


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
