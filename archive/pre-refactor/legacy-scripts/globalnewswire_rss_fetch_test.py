#!/usr/bin/env python3
import time, sys, requests, feedparser
from urllib.parse import urljoin
from bs4 import BeautifulSoup

FEED_URL = "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States"
UA = "MyRSSClient/1.0 (+contact@example.com)"  # set a real contact if you can
TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = 1.0  # be polite

def fetch(url):
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def try_trafilatura(html, base_url):
    try:
        import trafilatura
        downloaded = trafilatura.extract(html, url=base_url, include_comments=False, include_tables=False)
        if downloaded and len(downloaded.strip()) > 600:  # crude sanity check
            return downloaded.strip()
    except Exception:
        pass
    return None

def bs_text(el):
    # Convert an element’s HTML to readable text
    text = el.get_text("\n", strip=True)
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join([l for l in lines if l])

def heuristic_extract(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    # remove obvious non-content
    for tag in soup(["script","style","noscript","iframe","form","nav","header","footer","aside","button","svg"]):
        tag.decompose()

    # First, try common article containers (generic + GlobeNewswire guesses)
    selectors = [
        "article",
        '[itemprop="articleBody"]',
        ".article-body", ".articleBody", "#article-body", "#articleBody",
        "#article-content", ".article-content", "section.article-content",
        ".press-release", ".pressrelease", ".pr-body", "#release-body",
        ".entry-content", "#content .content", ".content__body", ".story-body",
        ".gnw-article", ".gnw-press-release"
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            txt = bs_text(el)
            if len(txt) > 400:  # avoid nav or stubs
                return txt

    # If we didn’t match a specific container, take the biggest texty <div>
    candidates = []
    for div in soup.find_all(["div","section"], limit=200):
        txt = bs_text(div)
        candidates.append((len(txt), txt))
    if candidates:
        candidates.sort(reverse=True)
        longest = candidates[0][1]
        if len(longest) > 400:
            return longest

    return None

def maybe_follow_amp(html, base_url):
    # If content is short, try the AMP version (often cleaner)
    soup = BeautifulSoup(html, "html.parser")
    amp = soup.find("link", rel="amphtml")
    if amp and amp.get("href"):
        amp_url = urljoin(base_url, amp["href"])
        try:
            r = fetch(amp_url)
            return r.text, r.url
        except Exception:
            return None, None
    return None, None

def extract_article(url):
    r = fetch(url)
    html, final_url = r.text, r.url

    # 1) Best: trafilatura
    text = try_trafilatura(html, final_url)
    if text: return text, final_url

    # 2) Heuristic extraction
    text = heuristic_extract(html, final_url)
    if text and len(text) > 600:
        return text, final_url

    # 3) Try AMP if available
    amp_html, amp_url = maybe_follow_amp(html, final_url)
    if amp_html:
        text = try_trafilatura(amp_html, amp_url) or heuristic_extract(amp_html, amp_url)
        if text and len(text) > 400:
            return text, amp_url

    # 4) Fallback: entire page text (last resort)
    from bs4 import BeautifulSoup
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    return text, final_url

def main(max_items=5):
    feed_bytes = fetch(FEED_URL).content
    parsed = feedparser.parse(feed_bytes)
    print(f"Feed: {parsed.feed.get('title', '(no title)')}")
    print(f"Items: {len(parsed.entries)}\n")

    for i, e in enumerate(parsed.entries[:max_items], 1):
        title = e.get("title","")
        link  = e.get("link","")
        print("="*80)
        print(f"{i}. {title}")
        print(link)
        try:
            text, final_url = extract_article(link)
            print(f"(fetched: {final_url})\n")
            print(text.strip())
        except requests.HTTPError as err:
            print(f"[HTTP error] {err}", file=sys.stderr)
        except Exception as ex:
            print(f"[Error] {type(ex).__name__}: {ex}", file=sys.stderr)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

if __name__ == "__main__":
    # pip install trafilatura beautifulsoup4 lxml feedparser
    main(max_items=5)
