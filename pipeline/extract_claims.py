"""
Myth Museum - Claim Extraction

Extract claims from raw content using regex patterns and optional LLM.
"""

import re
from typing import Optional

from core.constants import (
    MAX_CLAIMS_PER_ITEM,
    MAX_CLAIM_LENGTH,
    TOPIC_KEYWORDS,
    ClaimStatusEnum,
    TopicEnum,
)
from core.db import (
    get_unprocessed_raw_items,
    insert_claim,
    update_claim_status,
)
from core.logging import get_logger
from core.models import Claim, RawItem
from core.textnorm import deduplicate_claims, normalize_text, extract_sentences

logger = get_logger(__name__)


# Claim-indicating patterns (English and Chinese)
CLAIM_PATTERNS = [
    # English patterns
    r"(?:people|many|some|experts)\s+(?:say|believe|think|claim)\s+(?:that\s+)?(.{20,140})",
    r"(?:it is|it's)\s+(?:believed|said|known|claimed|thought)\s+(?:that\s+)?(.{20,140})",
    r"(?:studies|research|scientists)\s+(?:show|suggest|indicate|prove)\s+(?:that\s+)?(.{20,140})",
    r"(?:the\s+)?(?:myth|belief|idea|notion)\s+(?:that|of)\s+(.{20,140})",
    r"(?:contrary to|despite)\s+popular\s+(?:belief|opinion)\s*,?\s*(.{20,140})",
    r"(?:fact|true|false|myth):\s*(.{20,140})",
    
    # Chinese patterns
    r"(?:很多人|有人|人們|專家|研究)(?:認為|相信|說|聲稱|指出)(.{10,70})",
    r"(?:據說|傳說|坊間流傳|網路上說)(.{10,70})",
    r"(?:迷思|謠言|誤解|事實)(?:是|：|:)(.{10,70})",
]


def extract_claims_regex(text: str) -> list[str]:
    """
    Extract potential claims from text using regex patterns.
    
    Args:
        text: Input text content
    
    Returns:
        List of extracted claim strings (max 3)
    """
    if not text:
        return []
    
    claims = []
    
    # Try each pattern
    for pattern in CLAIM_PATTERNS:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Clean up the match
                claim = match.strip()
                claim = re.sub(r"\s+", " ", claim)
                
                # Skip if too short or too long
                if len(claim) < 20 or len(claim) > MAX_CLAIM_LENGTH:
                    continue
                
                claims.append(claim)
                
                if len(claims) >= MAX_CLAIMS_PER_ITEM:
                    break
        except re.error:
            continue
        
        if len(claims) >= MAX_CLAIMS_PER_ITEM:
            break
    
    # If no pattern matches, try extracting interesting sentences
    if not claims:
        claims = _extract_from_sentences(text)
    
    return claims[:MAX_CLAIMS_PER_ITEM]


def _extract_from_sentences(text: str) -> list[str]:
    """
    Fall back to extracting claim-like sentences.
    
    Args:
        text: Input text
    
    Returns:
        List of potential claim sentences
    """
    sentences = extract_sentences(text)
    
    claim_indicators = [
        "myth", "belief", "claim", "fact", "true", "false",
        "actually", "really", "proved", "debunked",
        "迷思", "謠言", "事實", "真相", "假", "證實",
    ]
    
    claims = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check for claim indicators
        has_indicator = any(ind in sentence_lower for ind in claim_indicators)
        
        # Also check for question patterns that indicate claims
        is_questioning = sentence.endswith("?") or "是否" in sentence or "嗎" in sentence
        
        if has_indicator or is_questioning:
            if 20 <= len(sentence) <= MAX_CLAIM_LENGTH:
                claims.append(sentence)
        
        if len(claims) >= MAX_CLAIMS_PER_ITEM:
            break
    
    return claims


async def extract_claims_llm(text: str, llm_client) -> list[str]:
    """
    Extract claims using LLM.
    
    Args:
        text: Input text content
        llm_client: LLM client instance
    
    Returns:
        List of extracted claim strings
    """
    if not llm_client.is_configured():
        logger.warning("LLM not configured, falling back to regex")
        return extract_claims_regex(text)
    
    prompt = f"""Analyze the following text and extract up to 3 claims that could be fact-checked.

A claim should be:
- A statement that can be verified as true or false
- About health, history, science, or common beliefs
- Between 20-140 characters
- Not a question, but a declarative statement

Text:
{text[:2000]}

Return ONLY a JSON array of claim strings, like:
["claim 1", "claim 2", "claim 3"]

If no claims found, return: []
"""
    
    try:
        result = await llm_client.chat_json([
            {"role": "system", "content": "You are a fact-checker extracting verifiable claims from text."},
            {"role": "user", "content": prompt},
        ])
        
        if isinstance(result, list):
            return [c for c in result if isinstance(c, str) and 20 <= len(c) <= MAX_CLAIM_LENGTH][:MAX_CLAIMS_PER_ITEM]
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
    
    return extract_claims_regex(text)


def classify_topic(claim_text: str) -> TopicEnum:
    """
    Classify claim into a topic category.
    
    Args:
        claim_text: The claim text
    
    Returns:
        TopicEnum value
    """
    claim_lower = normalize_text(claim_text)
    
    # Check each topic's keywords
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in claim_lower)
        if score > 0:
            scores[topic] = score
    
    if scores:
        # Return topic with highest score
        return max(scores, key=scores.get)
    
    return TopicEnum.UNKNOWN


def score_claim(claim_text: str, title: str = "") -> int:
    """
    Score a claim for interest/virality potential (0-100).
    
    Factors:
    - Contains controversial keywords
    - Length is optimal
    - Contains numbers/statistics
    - Contains emotional words
    
    Args:
        claim_text: The claim text
        title: Original article title
    
    Returns:
        Score 0-100
    """
    score = 50  # Base score
    
    claim_lower = claim_text.lower()
    title_lower = title.lower() if title else ""
    
    # Controversial/viral keywords (+10 each, max 30)
    viral_words = [
        "myth", "false", "wrong", "never", "always", "actually",
        "secret", "truth", "lie", "fake", "real", "proven",
        "迷思", "謠言", "真相", "假", "真", "證實", "揭穿",
    ]
    viral_score = min(30, sum(10 for w in viral_words if w in claim_lower))
    score += viral_score
    
    # Contains numbers (+10)
    if re.search(r"\d+", claim_text):
        score += 10
    
    # Optimal length (60-100 chars is ideal)
    length = len(claim_text)
    if 60 <= length <= 100:
        score += 10
    elif 40 <= length <= 120:
        score += 5
    
    # Health topic (+5 - always popular)
    health_words = ["health", "doctor", "medical", "病", "醫", "健康"]
    if any(w in claim_lower for w in health_words):
        score += 5
    
    # Cap at 100
    return min(100, score)


async def process_raw_items(
    conn,
    use_llm: bool = False,
    limit: int = 100,
    llm_client=None,
) -> int:
    """
    Process raw items to extract claims.
    
    Args:
        conn: Database connection
        use_llm: Whether to use LLM for extraction
        limit: Maximum items to process
        llm_client: Optional LLM client (created if needed)
    
    Returns:
        Number of claims extracted
    """
    # Get unprocessed raw items
    raw_items = get_unprocessed_raw_items(conn, limit=limit)
    
    if not raw_items:
        logger.info("No unprocessed raw items")
        return 0
    
    logger.info(f"Processing {len(raw_items)} raw items")
    
    # Create LLM client if needed
    if use_llm and llm_client is None:
        from core.llm import LLMClient
        llm_client = LLMClient.from_config()
    
    total_claims = 0
    all_claims = []
    
    for item in raw_items:
        # Combine title and content for extraction
        text = f"{item.title}\n\n{item.content}"
        
        # Extract claims
        if use_llm:
            claim_texts = await extract_claims_llm(text, llm_client)
        else:
            claim_texts = extract_claims_regex(text)
        
        for claim_text in claim_texts:
            topic = classify_topic(claim_text)
            score = score_claim(claim_text, item.title)
            
            # Detect language (simple heuristic)
            has_chinese = bool(re.search(r"[\u4e00-\u9fff]", claim_text))
            language = "zh" if has_chinese else "en"
            
            claim = Claim(
                raw_item_id=item.id,
                claim_text=claim_text,
                topic=topic.value,
                language=language,
                score=score,
                status=ClaimStatusEnum.NEW.value,
            )
            
            all_claims.append({
                "claim": claim,
                "claim_text": claim_text,
            })
    
    # Deduplicate across all claims
    unique_claims = deduplicate_claims(all_claims, text_key="claim_text")
    
    # Insert unique claims
    for item in unique_claims:
        claim = item["claim"]
        claim_id = insert_claim(conn, claim)
        if claim_id:
            total_claims += 1
            logger.debug(f"Extracted claim: {claim.claim_text[:50]}...")
    
    logger.info(f"Extracted {total_claims} claims from {len(raw_items)} items")
    return total_claims
