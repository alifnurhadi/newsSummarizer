import json
import logging
import re
import time
import uuid
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Indonesian month names  →  numeric month
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(raw: str) -> date | None:
    """
    Parse timestamps emitted by antaranews.com.

    Known formats:
      • "3 Mei 2025 08:45 WIB"
      • "Minggu, 4 Mei 2025 10:00 WIB"
      • "2025-05-03T08:45:00+07:00"   (ISO-8601 in <time> tags)
      • "03/05/2025 08:45"
    Returns a date object or None.
    """
    raw = raw.strip()
    if not raw:
        return None

    # ── ISO-8601 ──────────────────────────────────────────────────────────
    try:
        # strip timezone offset if present
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
        month = ID_MONTHS.get(m.group(2).lower())
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


# ---------------------------------------------------------------------------
# Step 1 – Collect candidate article links from the list page(s)
# ---------------------------------------------------------------------------


def _extract_articles_from_list(soup: BeautifulSoup) -> list[dict]:
    """
    Returns list of  {"link": str, "date": date}  dicts.

    Antaranews list-page patterns (observed across redesigns):

    Pattern A  — <div class="simple-list-*">  or  <div class="card-*">
                   └─ <a class="..." href="…">
                        ├─ <span class="...title...">…title…</span>
                        └─ <span class="...date..."><i class="…clock…"></i> 3 Mei 2025</span>

    Pattern B  — <div class="col-md-...">
                   └─ <a href="…">
                        ├─ <p class="...title...">…title…</p>
                        └─ <span class="..."><i>…</i> timestamp</span>

    We use a broad approach: find every <a> that contains a clock/date indicator,
    then extract the sibling/nested timestamp text.
    """
    results = []
    seen_links: set[str] = set()

    # ── Selector A: anchor tags that wrap the whole card ──────────────────
    # antaranews uses hrefs like /berita/XXXXXXX/slug  or  /ekonomi/bisnis/berita/…
    anchors = soup.find_all("a", href=re.compile(r"/berita/\d+/"))

    for a in anchors:
        href = a.get("href", "").strip()
        if not href:
            continue
        full_url = href if href.startswith("http") else BASE_URL + href
        if full_url in seen_links:
            continue

        # ── Find timestamp ── priority order:
        # 1. <i> tag with clock class inside this <a> or its closest container
        # 2. sibling <span> with date text
        # 3. parent container with a <span>/<time> tag

        raw_ts: str | None = None

        # 1) <i> inside the anchor or its parent container
        container = a.parent or a
        i_tags = container.find_all(
            "i", class_=re.compile(r"clock|time|calendar|date", re.I)
        )
        for i_tag in i_tags:
            # text usually follows the <i> tag in the same <span>
            parent_span = i_tag.parent
            if parent_span:
                raw_ts = parent_span.get_text(" ", strip=True)
                raw_ts = re.sub(r"^[^\d]*", "", raw_ts)  # drop icon text junk
                break

        # 2) <time datetime="…"> anywhere in the container
        if not raw_ts:
            time_tag = container.find("time")
            if time_tag:
                raw_ts = time_tag.get("datetime") or time_tag.get_text(strip=True)

        # 3) Any <span> whose text looks like a date
        if not raw_ts:
            for span in container.find_all("span"):
                txt = span.get_text(" ", strip=True)
                if re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", txt):
                    raw_ts = txt
                    break

        if not raw_ts:
            log.debug("No timestamp for %s — skipping", full_url)
            continue

        parsed = _parse_timestamp(raw_ts)
        if parsed is None:
            log.debug("Unparseable timestamp %r for %s", raw_ts, full_url)
            continue

        seen_links.add(full_url)
        results.append({"link": full_url, "date": parsed})

    return results


def collect_candidates(target_date: date) -> list[str]:
    """
    Paginate the list page, collect all article links dated == target_date.
    Returns list of unique URLs.
    """
    links: list[str] = []
    seen: set[str] = set()

    # Antaranews paginates via  ?page=N  or  /page/N  (try both)
    for page_num in range(1, MAX_LIST_PAGES + 1):
        if page_num == 1:
            url = LIST_URL
        else:
            url = f"{LIST_URL}?page={page_num}"

        log.info("Fetching list page %d: %s", page_num, url)
        soup = _get(url)
        if soup is None:
            log.warning("Failed to fetch list page %d — stopping pagination", page_num)
            break

        articles = _extract_articles_from_list(soup)
        if not articles:
            log.info("No articles found on page %d — stopping pagination", page_num)
            break

        found_on_page = 0
        older_than_target = 0

        for art in articles:
            if art["date"] == target_date:
                if art["link"] not in seen:
                    seen.add(art["link"])
                    links.append(art["link"])
                    found_on_page += 1
            elif art["date"] < target_date:
                older_than_target += 1

        log.info(
            "Page %d → %d total articles | %d matched | %d older",
            page_num,
            len(articles),
            found_on_page,
            older_than_target,
        )

        # If most articles are already older than our target, stop paginating
        if older_than_target > len(articles) * 0.6:
            log.info("Majority of articles are older — stopping pagination")
            break

        time.sleep(0.5)

    return links


# ---------------------------------------------------------------------------
# Step 2 – Fetch detail page and extract full article text
# ---------------------------------------------------------------------------


def extract_article_text(url: str) -> str | None:
    """
    Fetch an article page and return the clean full-text string,
    or None if extraction fails.

    Antaranews article body selectors (observed):
      • <div class="post-content"> … <p> … </p> … </div>
      • <div class="detail-text"> … <p> … </p> … </div>
      • <div class="single-post-content"> … </div>
    We try all candidates and pick the richest one.
    """
    soup = _get(url)
    if soup is None:
        return None

    # Remove boilerplate sections
    for tag in soup.find_all(
        ["script", "style", "noscript", "aside", "nav", "footer", "header"]
    ):
        tag.decompose()

    # Remove known ad / promo containers
    for tag in soup.find_all(
        class_=re.compile(
            r"ads|advertisement|promo|related|rekomendasi|"
            r"sidebar|widget|social|share|comment|subscribe|"
            r"newsletter|banner|iklan",
            re.I,
        )
    ):
        tag.decompose()

    # Candidate content selectors (ordered by specificity)
    CONTENT_SELECTORS = [
        {"class": re.compile(r"post.?content", re.I)},
        {"class": re.compile(r"detail.?text", re.I)},
        {"class": re.compile(r"single.?post", re.I)},
        {"class": re.compile(r"article.?body", re.I)},
        {"class": re.compile(r"entry.?content", re.I)},
        {"class": re.compile(r"news.?content", re.I)},
        {"class": re.compile(r"content.?detail", re.I)},
    ]

    best_div = None
    best_len = 0

    for sel in CONTENT_SELECTORS:
        div = soup.find("div", sel)
        if div:
            candidate_len = len(div.get_text(strip=True))
            if candidate_len > best_len:
                best_len = candidate_len
                best_div = div

    # Fallback: find the <div> with the most <p> children
    if best_div is None:
        divs = soup.find_all("div")
        for div in divs:
            p_count = len(div.find_all("p", recursive=False))
            text_len = len(div.get_text(strip=True))
            if p_count >= 3 and text_len > best_len:
                best_len = text_len
                best_div = div

    if best_div is None:
        log.warning("No content container found for %s", url)
        return None

    # ── Extract paragraphs ────────────────────────────────────────────────
    paragraphs: list[str] = []
    seen_para: set[str] = set()

    for p in best_div.find_all("p"):
        # Skip paragraphs that are inside aside/figure/caption/footer
        if p.find_parent(["aside", "figure", "figcaption", "footer"]):
            continue
        text = _clean_text(p.get_text(separator=" "))
        if not text:
            continue

        # Skip very short "paragraphs" (likely labels / buttons)
        if len(text) < 20:
            continue

        # Skip duplicates
        if text in seen_para:
            continue
        seen_para.add(text)

        # Skip common boilerplate snippets
        BOILERPLATE_PATTERNS = [
            r"^Baca juga",
            r"^Pewarta\s*:",
            r"^Editor\s*:",
            r"^COPYRIGHT\s*©",
            r"^Hak Cipta",
            r"^Antara News",
            r"^\*\s*",
        ]
        if any(re.match(pat, text, re.I) for pat in BOILERPLATE_PATTERNS):
            continue

        paragraphs.append(text)

    if not paragraphs:
        log.warning("Zero usable paragraphs extracted from %s", url)
        return None

    return " ".join(paragraphs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def scrape(output_path: str | None = None) -> list[dict]:
    yesterday = date.today() - timedelta(days=1)
    log.info("Target date: %s", yesterday.isoformat())

    # ── Phase 1: collect links ────────────────────────────────────────────
    candidate_links = collect_candidates(yesterday)
    log.info("Total candidate links from yesterday: %d", len(candidate_links))

    if not candidate_links:
        log.warning("No articles found for %s", yesterday.isoformat())
        return []

    # ── Phase 2: fetch article content ───────────────────────────────────
    results: list[dict] = []

    for idx, link in enumerate(candidate_links, 1):
        log.info("[%d/%d] Fetching article: %s", idx, len(candidate_links), link)
        time.sleep(DELAY_BETWEEN)

        text = extract_article_text(link)
        if not text:
            log.warning("Skipping (no content): %s", link)
            continue

        results.append(
            {
                "id": str(uuid.uuid4()),
                "link": link,
                "news": text,
            }
        )
        log.info("  ✓ extracted %d characters", len(text))

    log.info("Final records: %d", len(results))

    # ── Phase 3: output ───────────────────────────────────────────────────
    json_out = json.dumps(results, ensure_ascii=False, indent=2)

    if output_path is None:
        output_path = f"antaranews_bisnis_{yesterday.isoformat()}.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(json_out)
    log.info("Saved → %s", output_path)

    # Also print to stdout (strict JSON only)
    print(json_out)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scrape()
