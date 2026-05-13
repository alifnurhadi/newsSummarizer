import json
from datetime import datetime

import polars as pl
from dateutil.relativedelta import relativedelta

from ..init import PATH, loadNews


def CleanFilterData(newData):
    """
    Cleans new scraped data and prepares it for storage.
    Ensures only new, unique news is added and identifies 'government' data for the vector DB.
    """
    # 1. Load existing data for deduplication
    existing_news = loadNews()

    # If existing_news is a list of dicts, we extract IDs or links for comparison
    existing_ids = (
        [n.get("id") for n in existing_news] if isinstance(existing_news, list) else []
    )

    # 2. Load newData into Polars and clean basic errors
    df_new = pl.DataFrame(newData)

    # Ensure 'scraped_at' (or 'dates') column exists and drop nulls
    date_col = "scraped_at" if "scraped_at" in df_new.columns else "dates"
    df_new = df_new.drop_nulls(subset=[date_col])

    initial_count = df_new.height

    # 3. Deduplication: Only keep data NOT already in existing_news
    # We use 'id' or 'link' as the unique identifier
    if existing_ids:
        df_new = df_new.filter(~pl.col("id").is_in(existing_ids))

    # 4. Intra-batch deduplication: Remove identical titles within this scrape
    df_new = df_new.unique(subset=["link", "id"])

    print(
        f"Polars deduplication complete: Found {df_new.height} new unique items (Dropped {initial_count - df_new.height} duplicates)."
    )

    return df_new


def PrivateNewsFilter():
    """
    Removes news from the local JSON ledger that is older than 30 days.
    This keeps your 'Storage A' small and relevant.
    """
    news_list = loadNews()
    if not news_list or not isinstance(news_list, list):
        return None

    # Calculate the cutoff date (one month ago)
    today = datetime.now()
    cutoff_date = today - relativedelta(months=1)

    filtered_news = []
    dropped_count = 0

    for item in news_list:
        # 1. Get the date string (scraped_at or dates)
        date_str = item.get("scraped_at") or item.get("dates")

        if not date_str:
            filtered_news.append(item)
            continue

        try:
            # 2. Convert string to datetime for comparison
            # Assuming YYYY-MM-DD format based on your fetch results
            item_date = datetime.strptime(date_str, "%Y-%m-%d")

            # 3. Keep only if it is newer than one month ago
            if item_date >= cutoff_date:
                filtered_news.append(item)
            else:
                dropped_count += 1
        except ValueError:
            # If date format is weird, keep it to be safe
            filtered_news.append(item)

    # 4. Save the purged list back to the JSON file
    with open(PATH, "w") as wr:
        json.dump(filtered_news, wr, indent=4)

    print(f"Purged {dropped_count} old news items from local storage.")
    return None


if __name__ == "__main__":
    print("Running Cleaning Module Tests...")
