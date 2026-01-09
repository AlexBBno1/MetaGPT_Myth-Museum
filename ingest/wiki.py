"""
Myth Museum - Wikipedia Ingestion

Ingest content from Wikipedia API.
"""

import asyncio
import os
import ssl
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote

import httpx

from core.config import get_wiki_topics
from core.db import insert_raw_item
from core.logging import get_logger
from core.models import RawItem
from core.textnorm import compute_hash

logger = get_logger(__name__)

# Wikipedia API settings
WIKI_API_BASE = "https://en.wikipedia.org/api/rest_v1"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
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


async def search_wiki(keyword: str, limit: int = 5) -> list[str]:
    """
    Search Wikipedia for pages matching a keyword.
    
    Args:
        keyword: Search term
        limit: Maximum results to return
    
    Returns:
        List of page titles
    """
    logger.debug(f"Searching Wikipedia for: {keyword}")
    
    params = {
        "action": "opensearch",
        "search": keyword,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    }
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                WIKI_SEARCH_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.error(f"Wikipedia search failed for '{keyword}': {e}")
        return []
    
    # opensearch returns [query, [titles], [descriptions], [urls]]
    if len(data) >= 2:
        titles = data[1]
        logger.info(f"Found {len(titles)} Wikipedia pages for '{keyword}'")
        return titles
    
    return []


async def get_summary(title: str) -> Optional[dict[str, Any]]:
    """
    Get Wikipedia page summary.
    
    Args:
        title: Page title
    
    Returns:
        Dict with title, extract, url, or None
    """
    # URL-encode the title
    encoded_title = quote(title.replace(" ", "_"), safe="")
    url = f"{WIKI_API_BASE}/page/summary/{encoded_title}"
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"Failed to get Wikipedia summary for '{title}': {e}")
        return None
    
    # Check for disambiguation or missing
    if data.get("type") == "disambiguation":
        logger.debug(f"Skipping disambiguation page: {title}")
        return None
    
    if "extract" not in data:
        return None
    
    return {
        "title": data.get("title", title),
        "extract": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}"),
        "description": data.get("description", ""),
    }


async def get_full_content(title: str) -> Optional[str]:
    """
    Get full Wikipedia page content.
    
    Args:
        title: Page title
    
    Returns:
        Full text content or None
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
    }
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                WIKI_SEARCH_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"Failed to get Wikipedia content for '{title}': {e}")
        return None
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id != "-1" and "extract" in page:
            content = page["extract"]
            # Limit size
            return content[:15000] if content else None
    
    return None


async def ingest_wiki_topics(conn, config: dict) -> int:
    """
    Ingest Wikipedia articles for configured topics.
    
    Args:
        conn: Database connection
        config: Configuration dict
    
    Returns:
        Number of new items ingested
    """
    topics = get_wiki_topics(config)
    
    if not topics:
        logger.warning("No Wikipedia topics configured")
        return 0
    
    ingested = 0
    
    for topic in topics:
        # Search for pages
        titles = await search_wiki(topic, limit=5)
        
        for title in titles:
            # Get summary
            summary_data = await get_summary(title)
            
            if not summary_data:
                continue
            
            # Get full content for more detailed extraction
            full_content = await get_full_content(title)
            content = full_content or summary_data.get("extract", "")
            
            if not content or len(content) < 100:
                continue
            
            url = summary_data.get("url", f"https://en.wikipedia.org/wiki/{title}")
            
            # Create raw item
            item = RawItem(
                source_id=0,  # Wikipedia doesn't have a source_id in sources table
                url=url,
                title=summary_data.get("title", title),
                content=content,
                published_at=None,
                fetched_at=datetime.now(),
                hash=compute_hash(content),
            )
            
            # Insert
            item_id = insert_raw_item(conn, item)
            
            if item_id:
                ingested += 1
                logger.debug(f"Ingested Wikipedia: {title}")
            
            # Rate limiting
            await asyncio.sleep(0.5)
    
    logger.info(f"Ingested {ingested} Wikipedia articles")
    return ingested
