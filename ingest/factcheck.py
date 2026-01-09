"""
Myth Museum - Factcheck RSS Ingestion

Wrapper for ingesting from fact-checking RSS feeds.
"""

from typing import Optional

from core.config import get_sources
from core.constants import SourceTypeEnum
from core.logging import get_logger
from ingest.rss import ingest_rss_source

logger = get_logger(__name__)


# Known factcheck RSS feeds (placeholders - users should configure actual feeds)
FACTCHECK_FEEDS = [
    {
        "name": "PolitiFact",
        "url": "https://www.politifact.com/rss/all/",
    },
    {
        "name": "Snopes",
        "url": "https://www.snopes.com/feed/",
    },
    {
        "name": "Full Fact",
        "url": "https://fullfact.org/feed/",
    },
    {
        "name": "AFP Fact Check",
        "url": "https://factcheck.afp.com/rss.xml",
    },
    {
        "name": "Reuters Fact Check",
        "url": "https://www.reuters.com/fact-check/feed/",
    },
]


async def ingest_factcheck_sources(conn, config: dict) -> int:
    """
    Ingest from all configured factcheck sources.
    
    Uses the same RSS ingestion logic, but filters to factcheck sources.
    
    Args:
        conn: Database connection
        config: Configuration dict
    
    Returns:
        Number of new items ingested
    """
    sources = get_sources(config)
    
    # Filter to factcheck sources only
    factcheck_sources = [s for s in sources if s.type == SourceTypeEnum.FACTCHECK and s.enabled]
    
    if not factcheck_sources:
        logger.warning("No factcheck sources configured")
        return 0
    
    total = 0
    
    for source in factcheck_sources:
        try:
            count = await ingest_rss_source(conn, source)
            total += count
        except Exception as e:
            logger.error(f"Error ingesting from {source.name}: {e}")
    
    logger.info(f"Ingested {total} factcheck items")
    return total
