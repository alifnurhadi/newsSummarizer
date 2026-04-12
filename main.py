# import stuff

# do scraping logic
"""3. The "Engineer's Way" (View Page Source)

If you can't guess the URL and there is no icon, the website might still have a hidden feed.

Go to the homepage of the news site.

Right-click anywhere on the page and select "View Page Source" (or press Ctrl+U / Cmd+U).

Press Ctrl+F (or Cmd+F) to search the raw HTML.

Type in rss or application/rss+xml.

If it exists, you will see a line of code that looks like this:
<link rel="alternate" type="application/rss+xml" title="News Feed" href="https://website.com/hidden-feed-url" />
That href link is your golden ticket.

As a Data Engineer, you don't give up. You use a "Feed Generator."
There are open-source tools and services (like RSSHub, PolitePol, or FetchRSS) where you paste the URL of the normal HTML website, tell the tool which CSS tags hold the news titles, and it will artificially generate a stable RSS feed for your Airflow pipeline to consume.

Your Next Step for the Portfolio:
Before you write the Python code we discussed, try going to Bisnis.com, CNBC Indonesia, or Kontan.co.id in your browser. Try adding /rss to the end of their URLs. You will see a wall of raw XML code appear—that is your data source.

"""


# do processing result

# do analyze with AI


def main():
    print("Hello from newssummarizer!")


if __name__ == "__main__":
    main()
