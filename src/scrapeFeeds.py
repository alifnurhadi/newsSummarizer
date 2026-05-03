import json

import requests
from bs4 import BeautifulSoup


def PublicNews():
    # Define the target URLs based on your pagination rules
    urls_to_scrape = [
        "https://en.antaranews.com/current-issue",
        "https://en.antaranews.com/current-issue/2",
        "https://en.antaranews.com/current-issue/3",
        "https://en.antaranews.com/business-investment",
        "https://en.antaranews.com/business-investment/2",
    ]

    # Initialize the output structure
    output_data = {"source": [], "title": [], "time": []}

    # Updated to a standard Safari on macOS User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    }

    for url in urls_to_scrape:
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all article cards based on the wrapper class in your screenshot
            articles = soup.find_all("div", class_="card__post__body")

            for article in articles:
                # Extract Title and URL from the h2 > a tag
                title_tag = (
                    article.find("h2", class_="post_title").find("a")
                    if article.find("h2", class_="post_title")
                    else None
                )

                # Extract Time from the span tag within author info
                time_tag = article.find("span", class_="text-secondary")

                if title_tag:
                    output_data["source"].append(title_tag.get("href", ""))
                    output_data["title"].append(title_tag.text.strip())

                    # Append time if found, else empty string
                    if time_tag:
                        output_data["time"].append(time_tag.text.strip())
                    else:
                        output_data["time"].append("Time not found")

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    # Output the final JSON
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    PublicNews()
