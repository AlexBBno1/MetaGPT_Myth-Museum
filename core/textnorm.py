"""
Myth Museum - Text Normalization

Text processing utilities for deduplication and similarity detection.
"""

import hashlib
import re
import unicodedata
from typing import Optional

from rapidfuzz import fuzz

from core.constants import DEFAULT_SIMILARITY_THRESHOLD
from core.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison and hashing.
    
    - Converts to lowercase
    - Normalizes Unicode (NFKC)
    - Collapses whitespace
    - Strips leading/trailing whitespace
    - Removes excessive punctuation
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace various whitespace with single space
    text = re.sub(r"[\s\n\r\t]+", " ", text)
    
    # Remove excessive punctuation (keep single instances)
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    
    # Strip
    text = text.strip()
    
    return text


def compute_hash(text: str) -> str:
    """
    Compute SHA256 hash of normalized text.
    
    Args:
        text: Input text
    
    Returns:
        Hex digest of hash
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_similar(
    text1: str,
    text2: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Check if two texts are similar using fuzzy matching.
    
    Uses rapidfuzz's token_sort_ratio for robust comparison
    that handles word order differences.
    
    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0-100)
    
    Returns:
        True if similarity >= threshold
    """
    if not text1 or not text2:
        return False
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Use token_sort_ratio for word-order-independent comparison
    ratio = fuzz.token_sort_ratio(norm1, norm2)
    
    return ratio >= threshold


def similarity_score(text1: str, text2: str) -> float:
    """
    Get similarity score between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0-100)
    """
    if not text1 or not text2:
        return 0.0
    
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    return fuzz.token_sort_ratio(norm1, norm2)


def deduplicate_claims(
    claims: list[dict],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    text_key: str = "claim_text",
) -> list[dict]:
    """
    Deduplicate a list of claims based on text similarity.
    
    Keeps the first occurrence of each similar group.
    
    Args:
        claims: List of claim dictionaries
        threshold: Similarity threshold (0-100)
        text_key: Key for the text field in claim dicts
    
    Returns:
        Deduplicated list of claims
    """
    if not claims:
        return []
    
    unique_claims = []
    
    for claim in claims:
        claim_text = claim.get(text_key, "")
        
        # Check if similar to any existing claim
        is_duplicate = False
        for existing in unique_claims:
            existing_text = existing.get(text_key, "")
            if is_similar(claim_text, existing_text, threshold):
                is_duplicate = True
                logger.debug(f"Duplicate detected: '{claim_text[:50]}...' ~ '{existing_text[:50]}...'")
                break
        
        if not is_duplicate:
            unique_claims.append(claim)
    
    return unique_claims


def find_similar_in_list(
    text: str,
    texts: list[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Optional[tuple[int, float]]:
    """
    Find the most similar text in a list.
    
    Args:
        text: Text to search for
        texts: List of texts to search in
        threshold: Minimum similarity threshold
    
    Returns:
        Tuple of (index, score) or None if no match above threshold
    """
    if not text or not texts:
        return None
    
    best_idx = -1
    best_score = 0.0
    
    for idx, candidate in enumerate(texts):
        score = similarity_score(text, candidate)
        if score >= threshold and score > best_score:
            best_idx = idx
            best_score = score
    
    if best_idx >= 0:
        return (best_idx, best_score)
    
    return None


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncating
    
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.
    
    Handles Chinese and English sentence boundaries.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Split on common sentence terminators (English and Chinese)
    pattern = r'[.!?。！？]+'
    
    # Split and filter empty strings
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences
