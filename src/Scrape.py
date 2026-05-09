import json
import logging
import re
import time
import uuid
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from pydantic.types import ImportString

from ..init import ScrapeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sConf = ScrapeConfig()


def _get(url: str, retries: int = 3) -> BeautifulSoup | None:

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, headers=sConf.HEADERS, timeout=sConf.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s — %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2**attempt)
    return None


def _parse_timestamp(raw: str) -> date | None:

    raw = raw.strip()
    if not raw:
        return None

    try:
        iso = re.sub(r"[+-]\d{2}:\d{2}$", "", raw).strip()
        return datetime.fromisoformat(iso).date()
    except ValueError:
        pass

    # ── numeric DD/MM/YYYY ────────────────────────────────────────────────
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # ── Indonesian long form: "3 Mei 2025" ───────────────────────────────
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        day = int(m.group(1))
        month = sConf.ID_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    log.debug("Could not parse timestamp: %r", raw)
    return None


def _clean_text(text: str) -> str:
    """Strip excessive whitespace, normalise newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Hello from newssummarizer!")


if __name__ == "__main__":
    main()
