import json
from datetime import datetime

import polars as pl
from dateutil.relativedelta import relativedelta

from ..init import PATH, loadNews


def CleanFilterData(newData):
    """clean new data and take only goverment data to store data in vector database"""
    news = loadNews()
    topics = news["titles"]
    sites = news["sites"]

    df = pl.DataFrame(newData).drop_nulls(subset="dates")
    initial_count = df.height

    df = df.filter(~pl.col("titles").is_in(topics) | pl.col("sites").is_in(sites))

    df = df.unique(subset=["titles", "sites"])
    print(
        f"Polars deduplication complete: Dropped {initial_count - df.height} duplicate rows."
    )

    df = df.filter(pl.col("sources") == "government")
    return df


def PrivateNewsFilter():
    """removing private news that already above 30days old"""

    news = loadNews()
    today = datetime.now()
    monthsAgo = today - relativedelta(months=1)

    newdates = [td if td < monthsAgo else None for td in news["dates"]]

    news["dates"] = newdates

    with open(PATH, "w") as wr:
        json.dump(news, wr, indent=4)

    return None


if __name__ == "__main__":
    print("cleaning")
