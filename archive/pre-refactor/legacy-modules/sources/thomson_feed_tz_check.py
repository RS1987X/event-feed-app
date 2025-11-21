import feedparser
from pprint import pprint

THOMSON_FEED = "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15"

feed = feedparser.parse(THOMSON_FEED)
entry = feed.entries[0]  # look at the first item

print("Raw published string:   ", entry.get("published"))
print("published_parsed tuple: ", entry.get("published_parsed"))



DI_FEED_URL = "https://www.di.se/rss"

feed  = feedparser.parse(DI_FEED_URL)
entry = feed.entries[0]

print("Raw published string:   ", entry.get("published"))
print("published_parsed tuple: ", entry.get("published_parsed"))