"""
Myth Museum - Evidence Builder

Gather evidence for claims from multiple sources.
"""

import asyncio
import os
import re
from typing import Any, Optional
from urllib.parse import quote

import httpx

from core.constants import (
    CREDIBILITY_MAP,
    DEFAULT_MIN_EVIDENCE_SOURCES,
    ClaimStatusEnum,
    SourceTypeEnum,
)
from core.db import (
    get_claims_by_status,
    get_evidence_by_claim,
    insert_evidence,
    update_claim_status,
)
from core.logging import get_logger
from core.models import Claim, Evidence
from core.textnorm import normalize_text, similarity_score

logger = get_logger(__name__)

HTTP_TIMEOUT = 30
USER_AGENT = "MythMuseum/1.0 (Fact-checking bot)"

# SSL verification - can be disabled for development on Windows
# Set MYTH_MUSEUM_SKIP_SSL=1 to bypass SSL verification
SKIP_SSL = os.getenv("MYTH_MUSEUM_SKIP_SSL", "0") == "1"


def get_http_client(**kwargs) -> httpx.AsyncClient:
    """Get HTTP client with appropriate SSL settings."""
    if SKIP_SSL:
        return httpx.AsyncClient(verify=False, timeout=HTTP_TIMEOUT, **kwargs)
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT, **kwargs)


def generate_queries(claim_text: str) -> list[str]:
    """
    Generate search queries from a claim.
    
    Args:
        claim_text: The claim text
    
    Returns:
        List of 3-6 search queries (optimized for Wikipedia/academic search)
    """
    queries = []
    
    # Extract key terms (prioritize short, focused queries for better API results)
    normalized = normalize_text(claim_text)
    
    # Remove common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "must", "that",
        "this", "these", "those", "it", "its", "they", "them",
        "people", "believe", "think", "say", "said", "says", "claim",
        "claims", "believed", "thinks", "myth", "says", "many",
        "的", "是", "有", "在", "和", "了", "不", "也", "都",
    }
    
    words = normalized.split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if key_words:
        # Short 2-3 word query (best for Wikipedia)
        queries.append(" ".join(key_words[:2]))
        
        # Slightly longer query
        queries.append(" ".join(key_words[:3]))
        
        # Add subject-specific query
        queries.append(f"{key_words[0]} science" if len(key_words) > 0 else "")
        
        # Add "fact" query
        queries.append(f"{' '.join(key_words[:2])} fact")
    
    # Clean up empty queries
    queries = [q for q in queries if q and len(q) > 3]
    
    # Add original claim only if it's short (< 50 chars)
    if len(claim_text) < 50:
        queries.insert(0, claim_text)
    
    # Add Chinese queries if claim has Chinese
    if re.search(r"[\u4e00-\u9fff]", claim_text):
        queries.append(f"{claim_text[:30]} 事實查核")
        queries.append(f"{claim_text[:30]} 迷思 破解")
    
    return queries[:6]


async def search_wikipedia(query: str) -> list[dict[str, Any]]:
    """
    Search Wikipedia for evidence.
    
    Args:
        query: Search query
    
    Returns:
        List of evidence dicts
    """
    from ingest.wiki import search_wiki, get_summary
    
    evidence = []
    
    try:
        titles = await search_wiki(query, limit=3)
        
        for title in titles[:2]:
            summary = await get_summary(title)
            if summary:
                evidence.append({
                    "source_name": "Wikipedia",
                    "source_type": SourceTypeEnum.WIKIPEDIA.value,
                    "url": summary.get("url", f"https://en.wikipedia.org/wiki/{title}"),
                    "title": summary.get("title", title),
                    "snippet": summary.get("extract", "")[:500],
                    "credibility_score": CREDIBILITY_MAP.get(SourceTypeEnum.WIKIPEDIA.value, 70),
                })
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
    
    return evidence


async def search_crossref(query: str) -> list[dict[str, Any]]:
    """
    Search Crossref for academic evidence.
    
    Args:
        query: Search query
    
    Returns:
        List of evidence dicts
    """
    evidence = []
    
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": 3,
        "select": "title,DOI,abstract,published-print,URL",
    }
    
    try:
        async with get_http_client() as client:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
        
        items = data.get("message", {}).get("items", [])
        
        for item in items[:2]:
            title = item.get("title", [""])[0] if item.get("title") else ""
            doi = item.get("DOI", "")
            abstract = item.get("abstract", "")
            
            if title:
                evidence.append({
                    "source_name": "Crossref",
                    "source_type": SourceTypeEnum.ACADEMIC.value,
                    "url": f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                    "title": title,
                    "snippet": abstract[:500] if abstract else title,
                    "credibility_score": CREDIBILITY_MAP.get(SourceTypeEnum.ACADEMIC.value, 90),
                })
    except Exception as e:
        logger.warning(f"Crossref search failed: {e}")
    
    return evidence


async def search_factcheck_db(query: str, conn) -> list[dict[str, Any]]:
    """
    Search existing factcheck items in database.
    
    Args:
        query: Search query
        conn: Database connection
    
    Returns:
        List of evidence dicts
    """
    evidence = []
    
    # Simple text search in raw_items
    query_normalized = normalize_text(query)
    
    try:
        # Get factcheck raw items
        rows = conn.execute(
            """
            SELECT ri.* FROM raw_items ri
            JOIN sources s ON ri.source_id = s.id
            WHERE s.type = 'factcheck'
            LIMIT 50
            """
        ).fetchall()
        
        for row in rows:
            title = row["title"]
            content = row["content"]
            
            # Check similarity
            title_sim = similarity_score(query, title)
            content_sim = similarity_score(query, content[:500])
            
            if title_sim > 60 or content_sim > 50:
                evidence.append({
                    "source_name": "Factcheck DB",
                    "source_type": SourceTypeEnum.FACTCHECK.value,
                    "url": row["url"],
                    "title": title,
                    "snippet": content[:500],
                    "credibility_score": CREDIBILITY_MAP.get(SourceTypeEnum.FACTCHECK.value, 80),
                })
                
                if len(evidence) >= 2:
                    break
    except Exception as e:
        logger.warning(f"Factcheck DB search failed: {e}")
    
    return evidence


async def build_evidence_for_claim(claim: Claim, conn) -> int:
    """
    Gather evidence for a single claim.
    
    Args:
        claim: The claim to find evidence for
        conn: Database connection
    
    Returns:
        Number of evidence items added
    """
    logger.debug(f"Building evidence for: {claim.claim_text[:50]}...")
    
    # Generate queries
    queries = generate_queries(claim.claim_text)
    
    all_evidence = []
    source_types = set()
    
    # Search each source type
    for query in queries[:3]:  # Limit to 3 queries
        # Wikipedia
        wiki_evidence = await search_wikipedia(query)
        for e in wiki_evidence:
            e["query"] = query
            all_evidence.append(e)
            source_types.add(e["source_type"])
        
        # Crossref (academic)
        academic_evidence = await search_crossref(query)
        for e in academic_evidence:
            e["query"] = query
            all_evidence.append(e)
            source_types.add(e["source_type"])
        
        # Factcheck DB
        factcheck_evidence = await search_factcheck_db(query, conn)
        for e in factcheck_evidence:
            e["query"] = query
            all_evidence.append(e)
            source_types.add(e["source_type"])
        
        # Rate limiting
        await asyncio.sleep(0.5)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_evidence = []
    for e in all_evidence:
        if e["url"] not in seen_urls:
            seen_urls.add(e["url"])
            unique_evidence.append(e)
    
    # Insert evidence
    inserted = 0
    for e in unique_evidence[:10]:  # Max 10 evidence items
        evidence = Evidence(
            claim_id=claim.id,
            query=e["query"],
            source_name=e["source_name"],
            source_type=e["source_type"],
            url=e["url"],
            title=e["title"],
            snippet=e["snippet"],
            credibility_score=e["credibility_score"],
        )
        evidence_id = insert_evidence(conn, evidence)
        if evidence_id:
            inserted += 1
    
    # Update claim status based on evidence
    # Lenient mode: proceed with at least 1 evidence item
    # Strict mode: require 2+ different source types
    min_sources = int(os.getenv("MYTH_MUSEUM_MIN_EVIDENCE_SOURCES", str(DEFAULT_MIN_EVIDENCE_SOURCES)))
    
    if len(source_types) >= min_sources or (inserted >= 2 and min_sources <= 1):
        update_claim_status(conn, claim.id, ClaimStatusEnum.HAS_EVIDENCE.value)
    elif inserted > 0:
        # Allow proceeding with single source type if we have multiple evidence items
        if inserted >= 2:
            update_claim_status(conn, claim.id, ClaimStatusEnum.HAS_EVIDENCE.value)
        else:
            update_claim_status(conn, claim.id, ClaimStatusEnum.NEEDS_MORE_EVIDENCE.value)
    else:
        update_claim_status(conn, claim.id, ClaimStatusEnum.NEEDS_MORE_EVIDENCE.value)
    
    return inserted


async def process_claims_for_evidence(
    conn,
    min_score: int = 0,
    topic: Optional[str] = None,
    limit: int = 100,
) -> int:
    """
    Process claims to gather evidence.
    
    Args:
        conn: Database connection
        min_score: Minimum claim score
        topic: Filter by topic (optional)
        limit: Maximum claims to process
    
    Returns:
        Number of claims processed
    """
    # Get pending claims
    claims = get_claims_by_status(conn, ClaimStatusEnum.NEW.value, limit=limit, min_score=min_score)
    
    if topic:
        claims = [c for c in claims if c.topic == topic]
    
    if not claims:
        logger.info("No claims to process for evidence")
        return 0
    
    logger.info(f"Building evidence for {len(claims)} claims")
    
    processed = 0
    for claim in claims:
        try:
            evidence_count = await build_evidence_for_claim(claim, conn)
            if evidence_count > 0:
                processed += 1
                logger.debug(f"Added {evidence_count} evidence items for claim {claim.id}")
        except Exception as e:
            logger.error(f"Error processing claim {claim.id}: {e}")
    
    logger.info(f"Processed {processed} claims with evidence")
    return processed
