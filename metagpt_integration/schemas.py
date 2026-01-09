"""
Myth Museum - MetaGPT Integration Schemas

Pydantic models for message passing between MetaGPT roles.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    """Input for the Researcher role to gather evidence."""
    
    claim_id: int
    claim_text: str
    topic: str
    language: str = "en"
    score: int = 0
    raw_item_id: int
    
    def to_message_content(self) -> str:
        return f"Claim ID: {self.claim_id}\nClaim: {self.claim_text}\nTopic: {self.topic}"


class EvidenceItem(BaseModel):
    """A single evidence item found by the Researcher."""
    
    id: Optional[int] = None
    query: str
    source_name: str
    source_type: str
    url: str
    title: str
    snippet: str
    credibility_score: int


class EvidenceOutput(BaseModel):
    """Output from the Researcher role."""
    
    claim_id: int
    claim_text: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    source_types_found: list[str] = Field(default_factory=list)
    total_evidence_count: int = 0
    
    def to_message_content(self) -> str:
        evidence_text = "\n".join([
            f"- [{e.source_type}] {e.title}: {e.snippet[:100]}..."
            for e in self.evidence_items[:5]
        ])
        return f"Claim: {self.claim_text}\n\nEvidence ({self.total_evidence_count} items):\n{evidence_text}"


class VerdictInput(BaseModel):
    """Input for the FactChecker role."""
    
    claim_id: int
    claim_text: str
    topic: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    
    def to_message_content(self) -> str:
        evidence_text = "\n".join([
            f"Evidence #{i+1} [{e.source_type}, credibility: {e.credibility_score}]: {e.snippet[:200]}"
            for i, e in enumerate(self.evidence_items)
        ])
        return f"Claim: {self.claim_text}\nTopic: {self.topic}\n\nEvidence:\n{evidence_text}"


class VerdictOutput(BaseModel):
    """Output from the FactChecker role."""
    
    claim_id: int
    claim_text: str
    verdict: str  # False, Misleading, Depends, True, Unverified
    confidence: float = Field(ge=0.0, le=1.0)
    one_line_verdict: str
    why_believed: list[str] = Field(default_factory=list)
    what_wrong: str
    why_reasonable: str
    truth: str
    citation_map: dict[str, list[int]] = Field(default_factory=dict)
    disclaimer: Optional[str] = None
    
    def to_message_content(self) -> str:
        return f"""Claim: {self.claim_text}

Verdict: {self.verdict} (Confidence: {self.confidence:.0%})

Summary: {self.one_line_verdict}

The Truth: {self.truth}
"""


class ScriptInput(BaseModel):
    """Input for the ScriptWriter role."""
    
    claim_id: int
    claim_text: str
    topic: str
    verdict: str
    confidence: float
    explanation: dict[str, Any] = Field(default_factory=dict)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    
    def to_message_content(self) -> str:
        return f"""Create video scripts for:

Claim: {self.claim_text}
Verdict: {self.verdict}
Topic: {self.topic}

Explanation:
{self.explanation.get('truth', '')}
"""


class ShortsSegment(BaseModel):
    """A segment of a short-form video."""
    
    time_start: int
    time_end: int
    narration: str
    visual_suggestion: str
    on_screen_text: Optional[str] = None


class ScriptOutput(BaseModel):
    """Output from the ScriptWriter role."""
    
    claim_id: int
    
    # Shorts script (30-60 seconds)
    shorts_hook: str
    shorts_segments: list[ShortsSegment] = Field(default_factory=list)
    shorts_cta: str
    shorts_total_duration: int = 35
    
    # Long outline (6-10 minutes)
    long_chapters: list[dict[str, Any]] = Field(default_factory=list)
    long_total_duration: float = 8.0
    
    # Metadata
    titles: list[str] = Field(default_factory=list)
    thumbnail_suggestions: list[str] = Field(default_factory=list)
    description: str = ""
    next_myths: list[str] = Field(default_factory=list)
    
    def to_message_content(self) -> str:
        return f"""Scripts Generated for Claim #{self.claim_id}

Shorts Script ({self.shorts_total_duration}s):
Hook: {self.shorts_hook}
CTA: {self.shorts_cta}

Long Video ({self.long_total_duration:.1f} min):
Chapters: {len(self.long_chapters)}

Titles Generated: {len(self.titles)}
"""


class QAInput(BaseModel):
    """Input for the QA Reviewer role."""
    
    claim_id: int
    claim_text: str
    topic: str
    verdict_output: VerdictOutput
    script_output: ScriptOutput
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    
    def to_message_content(self) -> str:
        return f"""QA Review for Claim #{self.claim_id}

Claim: {self.claim_text}
Topic: {self.topic}
Verdict: {self.verdict_output.verdict}

Please check:
1. Citation accuracy
2. Disclaimer presence (if health/legal)
3. Script format compliance
4. Source credibility
"""


class QAIssue(BaseModel):
    """A single QA issue found."""
    
    category: str  # citation, disclaimer, format, credibility
    severity: str  # error, warning, info
    description: str
    fix_suggestion: Optional[str] = None


class QAOutput(BaseModel):
    """Output from the QA Reviewer role."""
    
    claim_id: int
    passed: bool = False
    issues: list[QAIssue] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    
    # Quality scores
    citation_score: float = Field(ge=0.0, le=1.0, default=0.0)
    format_score: float = Field(ge=0.0, le=1.0, default=0.0)
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    def to_message_content(self) -> str:
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        issues_text = "\n".join([f"- [{i.severity}] {i.description}" for i in self.issues[:5]])
        return f"""QA Review Result: {status}

Errors: {self.error_count}
Warnings: {self.warning_count}
Overall Score: {self.overall_score:.0%}

Issues:
{issues_text if issues_text else "No issues found"}
"""


class PipelineContext(BaseModel):
    """Context passed through the pipeline."""
    
    claim_id: int
    claim_text: str
    topic: str
    language: str = "en"
    
    # Pipeline outputs
    evidence_output: Optional[EvidenceOutput] = None
    verdict_output: Optional[VerdictOutput] = None
    script_output: Optional[ScriptOutput] = None
    qa_output: Optional[QAOutput] = None
    
    # Metadata
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
