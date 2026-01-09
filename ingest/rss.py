"""
Myth Museum - RSS Ingestion

Ingest content from RSS/Atom feeds.
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_sources
from core.constants import DEFAULT_RATE_LIMIT_SECONDS, SourceTypeEnum
from core.db import insert_raw_item
from core.logging import get_logger
from core.models import RawItem, SourceConfig
from core.textnorm import compute_hash

logger = get_logger(__name__)

# HTTP client settings
HTTP_TIMEOUT = 30
USER_AGENT = "MythMuseum/1.0 (Fact-checking bot; +https://github.com/myth-museum)"

# SSL verification - can be disabled for development on Windows
# Set MYTH_MUSEUM_SKIP_SSL=1 to bypass SSL verification
SKIP_SSL = os.getenv("MYTH_MUSEUM_SKIP_SSL", "0") == "1"


def get_http_client(**kwargs) -> httpx.AsyncClient:
    """Get HTTP client with appropriate SSL settings."""
    if SKIP_SSL:
        return httpx.AsyncClient(verify=False, timeout=HTTP_TIMEOUT, **kwargs)
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT, **kwargs)


async def fetch_feed(feed_url: str) -> list[dict[str, Any]]:
    """
    Fetch and parse an RSS/Atom feed.
    
    Args:
        feed_url: URL of the feed
    
    Returns:
        List of entry dicts with title, link, summary, published
    """
    logger.debug(f"Fetching feed: {feed_url}")
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                feed_url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            response.raise_for_status()
            content = response.text
    except Exception as e:
        logger.error(f"Failed to fetch feed {feed_url}: {e}")
        return []
    
    # Parse feed
    feed = feedparser.parse(content)
    
    if feed.bozo and not feed.entries:
        logger.warning(f"Failed to parse feed {feed_url}: {feed.bozo_exception}")
        return []
    
    entries = []
    for entry in feed.entries:
        # Extract published date
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6])
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published = datetime(*entry.updated_parsed[:6])
        
        entries.append({
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "summary": getattr(entry, "summary", ""),
            "published": published,
        })
    
    logger.info(f"Parsed {len(entries)} entries from {feed_url}")
    return entries


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def fetch_full_text(url: str) -> Optional[str]:
    """
    Fetch full text content from a URL.
    
    Attempts to extract article content using BeautifulSoup.
    Falls back to title/summary if full text cannot be extracted.
    
    Args:
        url: Article URL
    
    Returns:
        Extracted text content or None
    """
    logger.debug(f"Fetching full text: {url}")
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None
    
    # Parse HTML
    soup = BeautifulSoup(html, "lxml")
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
        script.decompose()
    
    # Try common article selectors
    article_selectors = [
        "article",
        '[role="main"]',
        ".article-content",
        ".article-body",
        ".post-content",
        ".entry-content",
        ".content",
        "main",
    ]
    
    content = None
    for selector in article_selectors:
        element = soup.select_one(selector)
        if element:
            content = element.get_text(separator=" ", strip=True)
            if len(content) > 200:  # Reasonable minimum
                break
    
    # Fallback to body text
    if not content or len(content) < 200:
        body = soup.find("body")
        if body:
            content = body.get_text(separator=" ", strip=True)
    
    if content:
        # Clean up whitespace
        content = " ".join(content.split())
        return content[:10000]  # Limit size
    
    return None


async def ingest_rss_source(
    conn,
    source: SourceConfig,
    rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
) -> int:
    """
    Ingest items from an RSS source.
    
    Args:
        conn: Database connection
        source: Source configuration
        rate_limit: Seconds between requests
    
    Returns:
        Number of new items ingested
    """
    if not source.enabled:
        logger.debug(f"Skipping disabled source: {source.name}")
        return 0
    
    feed_url = source.feed_url
    if not feed_url:
        logger.warning(f"No feed_url for source: {source.name}")
        return 0
    
    # Fetch feed
    entries = await fetch_feed(feed_url)
    
    if not entries:
        return 0
    
    ingested = 0
    
    for entry in entries:
        if not entry.get("link"):
            continue
        
        url = entry["link"]
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = entry.get("published")
        
        # Try to fetch full text
        content = await fetch_full_text(url)
        
        # Fall back to summary if full text unavailable
        if not content:
            content = summary
        
        if not content:
            logger.debug(f"No content for {url}")
            continue
        
        # Create raw item
        item = RawItem(
            source_id=source.id or 0,
            url=url,
            title=title,
            content=content,
            published_at=published,
            fetched_at=datetime.now(),
            hash=compute_hash(content),
        )
        
        # Insert (will skip duplicates)
        item_id = insert_raw_item(conn, item)
        
        if item_id:
            ingested += 1
            logger.debug(f"Ingested: {title[:50]}")
        
        # Rate limiting
        await asyncio.sleep(rate_limit)
    
    logger.info(f"Ingested {ingested} items from {source.name}")
    return ingested


async def ingest_all_rss(
    conn,
    config: dict,
    source_type: Optional[SourceTypeEnum] = None,
) -> int:
    """
    Ingest from all configured RSS sources.
    
    Args:
        conn: Database connection
        config: Configuration dict
        source_type: Filter by source type (None = all RSS-compatible)
    
    Returns:
        Total number of new items ingested
    """
    sources = get_sources(config)
    
    # Filter to RSS-compatible sources
    rss_types = {SourceTypeEnum.RSS, SourceTypeEnum.FACTCHECK, SourceTypeEnum.NEWS, SourceTypeEnum.OFFICIAL}
    
    if source_type:
        sources = [s for s in sources if s.type == source_type and s.enabled]
    else:
        sources = [s for s in sources if s.type in rss_types and s.enabled]
    
    if not sources:
        logger.warning("No RSS sources configured")
        return 0
    
    total = 0
    for source in sources:
        try:
            count = await ingest_rss_source(conn, source)
            total += count
        except Exception as e:
            logger.error(f"Error ingesting from {source.name}: {e}")
    
    return total
