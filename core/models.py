"""
Myth Museum - Pydantic Models

Data models for all entities in the pipeline.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.constants import (
    ClaimStatusEnum,
    SourceTypeEnum,
    TopicEnum,
    VerdictEnum,
)


# ============================================================================
# Source Configuration
# ============================================================================


class SourceConfig(BaseModel):
    """Configuration for a data source."""
    
    id: Optional[int] = None
    name: str
    type: SourceTypeEnum
    config_json: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    
    # For RSS sources
    @property
    def feed_url(self) -> Optional[str]:
        return self.config_json.get("feed_url")
    
    # For Wikipedia sources
    @property
    def topics(self) -> list[str]:
        return self.config_json.get("topics", [])


# ============================================================================
# Raw Items (Ingested Content)
# ============================================================================


class RawItem(BaseModel):
    """Raw content ingested from a source."""
    
    id: Optional[int] = None
    source_id: int
    url: str
    title: str
    content: str
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=datetime.now)
    hash: str  # SHA256 of normalized content


# ============================================================================
# Claims
# ============================================================================


class Claim(BaseModel):
    """A claim extracted from raw content."""
    
    id: Optional[int] = None
    raw_item_id: int
    claim_text: str
    topic: str  # TopicEnum value
    language: str = "en"
    score: int = 0  # 0-100
    status: str = ClaimStatusEnum.NEW.value
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Evidence
# ============================================================================


class Evidence(BaseModel):
    """Evidence gathered for a claim."""
    
    id: Optional[int] = None
    claim_id: int
    query: str  # The search query used
    source_name: str  # Source identifier
    source_type: str = SourceTypeEnum.RSS.value
    url: str
    title: str
    snippet: str
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=datetime.now)
    credibility_score: int = 50  # 0-100


# ============================================================================
# Verdict and Explanation
# ============================================================================


class ExplanationJSON(BaseModel):
    """Structured explanation for a verdict."""
    
    one_line_verdict: str = ""
    why_believed: list[str] = Field(default_factory=list)  # 3-5 points
    what_wrong: str = ""  # Clear explanation with analogy
    why_reasonable: str = ""  # Why the misconception is understandable
    truth: str = ""  # The correct statement
    citation_map: dict[str, list[int]] = Field(default_factory=dict)
    # Maps each conclusion key to list of evidence IDs
    # e.g., {"why_believed": [1, 2], "what_wrong": [3, 4]}
    disclaimer: Optional[str] = None


class Verdict(BaseModel):
    """Verdict for a claim."""
    
    id: Optional[int] = None
    claim_id: int
    verdict: str = VerdictEnum.UNVERIFIED.value
    explanation_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0  # 0.0-1.0
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_explanation(self) -> ExplanationJSON:
        """Parse explanation_json into ExplanationJSON model."""
        return ExplanationJSON(**self.explanation_json)


# ============================================================================
# Script Components
# ============================================================================


class ShortsSegment(BaseModel):
    """A segment of a short-form video script."""
    
    time_start: int  # seconds
    time_end: int  # seconds
    narration: str
    visual_suggestion: str
    on_screen_text: Optional[str] = None


class ShortsScript(BaseModel):
    """Complete short-form video script (30-60 seconds)."""
    
    hook: str
    segments: list[ShortsSegment] = Field(default_factory=list)
    cta: str  # Call to action
    total_duration: int = 0  # seconds


class ChapterOutline(BaseModel):
    """A chapter in a long-form video outline."""
    
    chapter_number: int
    title: str
    duration_minutes: float
    key_points: list[str] = Field(default_factory=list)
    visual_suggestions: list[str] = Field(default_factory=list)


class LongOutline(BaseModel):
    """Complete long-form video outline (6-10 minutes)."""
    
    chapters: list[ChapterOutline] = Field(default_factory=list)
    total_duration_minutes: float = 0


# ============================================================================
# Packet (Final Output)
# ============================================================================


class SourceReference(BaseModel):
    """A source reference for citation."""
    
    evidence_id: int
    title: str
    url: str
    source_type: str
    credibility_score: int


class PacketJSON(BaseModel):
    """Complete packet JSON structure for output."""
    
    # Core claim info
    claim: str
    claim_id: int
    topic: str
    language: str
    
    # Verdict
    verdict: str
    confidence: float
    
    # Explanation
    one_line_verdict: str
    why_believed: list[str]
    what_wrong: str
    why_reasonable: str
    truth: str
    citation_map: dict[str, list[int]]
    disclaimer: Optional[str] = None
    
    # Scripts
    shorts_script: dict[str, Any] = Field(default_factory=dict)
    long_outline: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    thumbnail_text_suggestions: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    description: str = ""
    visuals: list[str] = Field(default_factory=list)
    next_myths: list[str] = Field(default_factory=list)
    
    # Sources
    sources: list[SourceReference] = Field(default_factory=list)
    
    # Timestamps
    created_at: str = ""


class Packet(BaseModel):
    """Database record for a packet."""
    
    id: Optional[int] = None
    claim_id: int
    packet_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_packet(self) -> PacketJSON:
        """Parse packet_json into PacketJSON model."""
        return PacketJSON(**self.packet_json)


# ============================================================================
# LLM Configuration
# ============================================================================


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
