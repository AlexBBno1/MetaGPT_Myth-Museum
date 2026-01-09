"""
Myth Museum - MetaGPT Actions

Custom Actions extending metagpt.actions.Action for the fact-checking pipeline.

Note: This module requires MetaGPT to be installed and accessible.
If MetaGPT is not available, use the local pipeline instead.
"""

import sys
from pathlib import Path
from typing import Any

# Add project root and MetaGPT root to path for imports
project_root = Path(__file__).parent.parent
metagpt_root = project_root.parent.parent  # Go up to MetaGPT root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(metagpt_root) not in sys.path:
    sys.path.insert(0, str(metagpt_root))

try:
    from metagpt.actions import Action
    from metagpt.logs import logger
    METAGPT_AVAILABLE = True
except ImportError:
    # Create stub classes if MetaGPT is not available
    from core.logging import get_logger
    logger = get_logger(__name__)
    
    class Action:
        """Stub Action class when MetaGPT is not available."""
        name: str = ""
        
        async def _aask(self, prompt: str) -> str:
            return ""
        
        async def run(self, *args, **kwargs):
            raise NotImplementedError("MetaGPT is not installed")
    
    METAGPT_AVAILABLE = False
    logger.warning("MetaGPT not available. Using stub Action class.")

from core.constants import (
    DISCLAIMER_TOPICS,
    HEALTH_LEGAL_DISCLAIMER,
    TopicEnum,
    VerdictEnum,
)
from metagpt_integration.schemas import (
    ClaimInput,
    EvidenceItem,
    EvidenceOutput,
    QAInput,
    QAIssue,
    QAOutput,
    ScriptInput,
    ScriptOutput,
    ShortsSegment,
    VerdictInput,
    VerdictOutput,
)


class GatherEvidence(Action):
    """
    Action to gather evidence for a claim from multiple sources.
    
    Uses Wikipedia, Crossref, and fact-check database.
    """
    
    name: str = "GatherEvidence"
    
    PROMPT_TEMPLATE: str = """You are a research assistant gathering evidence to fact-check a claim.

CLAIM: {claim_text}
TOPIC: {topic}

Based on the following evidence sources, summarize the key findings:

{evidence_summary}

Please provide a structured analysis:
1. What do the sources agree on?
2. What contradictions exist?
3. Overall credibility assessment (0-100)
4. Key evidence items to cite

Return your analysis as a JSON object.
"""
    
    async def run(self, claim_input: ClaimInput) -> EvidenceOutput:
        """
        Gather evidence for a claim.
        
        Args:
            claim_input: The claim to research
        
        Returns:
            EvidenceOutput with gathered evidence
        """
        logger.info(f"[GatherEvidence] Researching claim: {claim_input.claim_text[:50]}...")
        
        try:
            # Import pipeline functions
            from pipeline.build_evidence import (
                generate_queries,
                search_crossref,
                search_wikipedia,
            )
            
            # Generate queries
            queries = generate_queries(claim_input.claim_text)
            
            evidence_items = []
            source_types = set()
            
            # Search each source
            for query in queries[:3]:
                # Wikipedia
                wiki_results = await search_wikipedia(query)
                for result in wiki_results:
                    item = EvidenceItem(
                        query=query,
                        source_name=result["source_name"],
                        source_type=result["source_type"],
                        url=result["url"],
                        title=result["title"],
                        snippet=result["snippet"][:500],
                        credibility_score=result["credibility_score"],
                    )
                    evidence_items.append(item)
                    source_types.add(result["source_type"])
                
                # Crossref
                academic_results = await search_crossref(query)
                for result in academic_results:
                    item = EvidenceItem(
                        query=query,
                        source_name=result["source_name"],
                        source_type=result["source_type"],
                        url=result["url"],
                        title=result["title"],
                        snippet=result["snippet"][:500],
                        credibility_score=result["credibility_score"],
                    )
                    evidence_items.append(item)
                    source_types.add(result["source_type"])
            
            # Deduplicate by URL
            seen_urls = set()
            unique_evidence = []
            for e in evidence_items:
                if e.url not in seen_urls:
                    seen_urls.add(e.url)
                    unique_evidence.append(e)
            
            logger.info(f"[GatherEvidence] Found {len(unique_evidence)} evidence items")
            
            return EvidenceOutput(
                claim_id=claim_input.claim_id,
                claim_text=claim_input.claim_text,
                evidence_items=unique_evidence[:10],
                source_types_found=list(source_types),
                total_evidence_count=len(unique_evidence),
            )
            
        except Exception as e:
            logger.error(f"[GatherEvidence] Error: {e}")
            return EvidenceOutput(
                claim_id=claim_input.claim_id,
                claim_text=claim_input.claim_text,
                evidence_items=[],
                source_types_found=[],
                total_evidence_count=0,
            )


class JudgeClaim(Action):
    """
    Action to generate a verdict for a claim based on evidence.
    """
    
    name: str = "JudgeClaim"
    
    PROMPT_TEMPLATE: str = """You are an expert fact-checker. Evaluate the following claim based on the provided evidence.

CLAIM: {claim_text}
TOPIC: {topic}

EVIDENCE:
{evidence_text}

Provide your verdict in the following JSON format:
{{
    "verdict": "False|Misleading|Depends|True|Unverified",
    "confidence": 0.0-1.0,
    "one_line_verdict": "Brief summary",
    "why_believed": ["reason1", "reason2", "reason3"],
    "what_wrong": "Explanation of what's incorrect or misleading",
    "why_reasonable": "Why this misconception is understandable",
    "truth": "The correct information",
    "citation_map": {{
        "one_line_verdict": [evidence_ids],
        "what_wrong": [evidence_ids],
        "truth": [evidence_ids]
    }}
}}
"""
    
    async def run(self, verdict_input: VerdictInput) -> VerdictOutput:
        """
        Generate a verdict for a claim.
        
        Args:
            verdict_input: Claim and evidence to judge
        
        Returns:
            VerdictOutput with verdict and explanation
        """
        logger.info(f"[JudgeClaim] Judging claim: {verdict_input.claim_text[:50]}...")
        
        try:
            # If no evidence, return Unverified
            if not verdict_input.evidence_items:
                return VerdictOutput(
                    claim_id=verdict_input.claim_id,
                    claim_text=verdict_input.claim_text,
                    verdict=VerdictEnum.UNVERIFIED.value,
                    confidence=0.3,
                    one_line_verdict="Insufficient evidence to verify this claim.",
                    why_believed=["This is a commonly shared belief"],
                    what_wrong="We could not find enough evidence to verify or refute this claim.",
                    why_reasonable="Without clear evidence, it's understandable to be uncertain.",
                    truth="More research is needed to confirm or deny this claim.",
                    citation_map={},
                )
            
            # Build evidence text
            evidence_text = "\n".join([
                f"Evidence #{i+1} ({e.source_type}, credibility: {e.credibility_score}):\n"
                f"Title: {e.title}\nURL: {e.url}\nContent: {e.snippet[:300]}\n"
                for i, e in enumerate(verdict_input.evidence_items)
            ])
            
            # Try LLM if available
            try:
                prompt = self.PROMPT_TEMPLATE.format(
                    claim_text=verdict_input.claim_text,
                    topic=verdict_input.topic,
                    evidence_text=evidence_text,
                )
                
                response = await self._aask(prompt)
                
                # Parse response (simplified - real implementation would parse JSON)
                import json
                import re
                
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Add disclaimer for health/legal topics
                    disclaimer = None
                    try:
                        topic_enum = TopicEnum(verdict_input.topic)
                        if topic_enum in DISCLAIMER_TOPICS:
                            disclaimer = HEALTH_LEGAL_DISCLAIMER
                    except ValueError:
                        pass
                    
                    return VerdictOutput(
                        claim_id=verdict_input.claim_id,
                        claim_text=verdict_input.claim_text,
                        verdict=result.get("verdict", VerdictEnum.UNVERIFIED.value),
                        confidence=result.get("confidence", 0.5),
                        one_line_verdict=result.get("one_line_verdict", ""),
                        why_believed=result.get("why_believed", []),
                        what_wrong=result.get("what_wrong", ""),
                        why_reasonable=result.get("why_reasonable", ""),
                        truth=result.get("truth", ""),
                        citation_map=result.get("citation_map", {}),
                        disclaimer=disclaimer,
                    )
            except Exception as llm_error:
                logger.warning(f"[JudgeClaim] LLM failed: {llm_error}, using rule-based")
            
            # Fall back to rule-based
            return self._judge_rule_based(verdict_input)
            
        except Exception as e:
            logger.error(f"[JudgeClaim] Error: {e}")
            return VerdictOutput(
                claim_id=verdict_input.claim_id,
                claim_text=verdict_input.claim_text,
                verdict=VerdictEnum.UNVERIFIED.value,
                confidence=0.3,
                one_line_verdict="Error during verdict generation.",
                why_believed=[],
                what_wrong="",
                why_reasonable="",
                truth="",
                citation_map={},
            )
    
    def _judge_rule_based(self, verdict_input: VerdictInput) -> VerdictOutput:
        """Rule-based verdict generation as fallback."""
        evidence = verdict_input.evidence_items
        
        # Handle empty evidence
        if not evidence:
            return VerdictOutput(
                claim_id=verdict_input.claim_id,
                claim_text=verdict_input.claim_text,
                verdict=VerdictEnum.UNVERIFIED.value,
                confidence=0.3,
                one_line_verdict="Insufficient evidence to verify this claim.",
                why_believed=["This is a commonly shared belief"],
                what_wrong="We could not find enough evidence.",
                why_reasonable="Without clear evidence, uncertainty is understandable.",
                truth="More research is needed.",
                citation_map={},
            )
        
        # Calculate average credibility
        avg_credibility = sum(e.credibility_score for e in evidence) / len(evidence)
        
        # Check for factcheck sources
        factcheck = [e for e in evidence if e.source_type == "factcheck"]
        academic = [e for e in evidence if e.source_type == "academic"]
        
        if factcheck:
            verdict = VerdictEnum.FALSE.value
            confidence = min(0.8, 0.6 + len(factcheck) * 0.1)
        elif academic and avg_credibility >= 80:
            verdict = VerdictEnum.DEPENDS.value
            confidence = 0.7
        elif avg_credibility >= 70:
            verdict = VerdictEnum.MISLEADING.value
            confidence = 0.6
        else:
            verdict = VerdictEnum.UNVERIFIED.value
            confidence = 0.4
        
        # Build citation map
        evidence_ids = [e.id or i+1 for i, e in enumerate(evidence)]
        citation_map = {
            "one_line_verdict": evidence_ids[:2],
            "what_wrong": evidence_ids,
            "truth": evidence_ids[:2],
        }
        
        # Add disclaimer
        disclaimer = None
        try:
            topic_enum = TopicEnum(verdict_input.topic)
            if topic_enum in DISCLAIMER_TOPICS:
                disclaimer = HEALTH_LEGAL_DISCLAIMER
        except ValueError:
            pass
        
        return VerdictOutput(
            claim_id=verdict_input.claim_id,
            claim_text=verdict_input.claim_text,
            verdict=verdict,
            confidence=confidence,
            one_line_verdict=f"Based on {len(evidence)} sources, this claim is {verdict.lower()}.",
            why_believed=[
                "This belief is commonly shared",
                "Similar claims circulate online",
                "The claim sounds plausible",
            ],
            what_wrong=f"The claim '{verdict_input.claim_text[:50]}...' is not fully supported by evidence.",
            why_reasonable="This misconception is understandable given how information spreads.",
            truth="The scientific consensus does not fully support this claim.",
            citation_map=citation_map,
            disclaimer=disclaimer,
        )


class GenerateScript(Action):
    """
    Action to generate video scripts for a fact-checked claim.
    """
    
    name: str = "GenerateScript"
    
    PROMPT_TEMPLATE: str = """You are a video content creator specializing in fact-checking videos.

Create scripts for the following fact-check:

CLAIM: {claim_text}
VERDICT: {verdict}
EXPLANATION: {explanation}

Generate:
1. A 30-60 second shorts script with:
   - Hook (attention-grabbing opening)
   - 4-5 segments (5-8 seconds each)
   - CTA (call to action)

2. A 6-10 minute long-form outline with:
   - 6 chapters
   - Key points per chapter
   - Visual suggestions

Return as JSON.
"""
    
    async def run(self, script_input: ScriptInput) -> ScriptOutput:
        """
        Generate video scripts.
        
        Args:
            script_input: Verdict and claim info
        
        Returns:
            ScriptOutput with scripts and metadata
        """
        logger.info(f"[GenerateScript] Creating scripts for claim: {script_input.claim_text[:50]}...")
        
        try:
            # Generate baseline scripts
            shorts = self._generate_shorts(script_input)
            long = self._generate_long_outline(script_input)
            titles = self._generate_titles(script_input.claim_text, script_input.verdict)
            next_myths = self._generate_next_myths(script_input.topic)
            
            return ScriptOutput(
                claim_id=script_input.claim_id,
                shorts_hook=shorts["hook"],
                shorts_segments=shorts["segments"],
                shorts_cta=shorts["cta"],
                shorts_total_duration=shorts["duration"],
                long_chapters=long["chapters"],
                long_total_duration=long["duration"],
                titles=titles,
                thumbnail_suggestions=["MYTH BUSTED!", script_input.verdict.upper(), "THE TRUTH"],
                description=self._generate_description(script_input),
                next_myths=next_myths,
            )
            
        except Exception as e:
            logger.error(f"[GenerateScript] Error: {e}")
            return ScriptOutput(
                claim_id=script_input.claim_id,
                shorts_hook="Did you know this common belief might be wrong?",
                shorts_segments=[],
                shorts_cta="Follow for more fact-checks!",
                shorts_total_duration=30,
                long_chapters=[],
                long_total_duration=6.0,
                titles=["Fact Check: Is It True?"],
                thumbnail_suggestions=["FACT CHECK"],
                description="",
                next_myths=[],
            )
    
    def _generate_shorts(self, script_input: ScriptInput) -> dict[str, Any]:
        """Generate shorts script template."""
        claim_short = script_input.claim_text[:50]
        verdict = script_input.verdict
        explanation = script_input.explanation
        
        hooks = {
            "False": f"You've been told {claim_short}... But is it true?",
            "Misleading": f"There's something wrong with {claim_short}...",
            "Depends": f"{claim_short}... It's complicated.",
            "True": f"Is {claim_short} true? Yes, here's why.",
            "Unverified": f"Can we verify {claim_short}? Let's find out.",
        }
        
        hook = hooks.get(verdict, f"Let's fact-check: {claim_short}")
        
        segments = [
            ShortsSegment(
                time_start=0, time_end=5,
                narration=hook,
                visual_suggestion="Hook visual with claim text",
                on_screen_text=claim_short,
            ),
            ShortsSegment(
                time_start=5, time_end=12,
                narration=explanation.get("one_line_verdict", f"This claim is {verdict.lower()}."),
                visual_suggestion="Verdict reveal animation",
                on_screen_text=verdict.upper(),
            ),
            ShortsSegment(
                time_start=12, time_end=25,
                narration=explanation.get("what_wrong", "Here's what the evidence shows...")[:150],
                visual_suggestion="Evidence visualization",
                on_screen_text="The Evidence",
            ),
            ShortsSegment(
                time_start=25, time_end=35,
                narration=explanation.get("truth", "The correct answer is...")[:100],
                visual_suggestion="Summary graphic",
                on_screen_text="Remember This",
            ),
        ]
        
        return {
            "hook": hook,
            "segments": segments,
            "cta": "Follow for more fact-checks! Link in bio for sources.",
            "duration": 35,
        }
    
    def _generate_long_outline(self, script_input: ScriptInput) -> dict[str, Any]:
        """Generate long-form outline template."""
        explanation = script_input.explanation
        
        chapters = [
            {
                "number": 1,
                "title": "The Myth Everyone Believes",
                "duration": 1.5,
                "key_points": [
                    f"Introduction: {script_input.claim_text[:60]}",
                    "Why this myth is widespread",
                    "Personal anecdotes",
                ],
            },
            {
                "number": 2,
                "title": "Where Did This Come From?",
                "duration": 1.5,
                "key_points": [
                    "Origins of the belief",
                    "Why people find it convincing",
                    explanation.get("why_reasonable", "Historical context"),
                ],
            },
            {
                "number": 3,
                "title": "What Does The Evidence Say?",
                "duration": 2.0,
                "key_points": [
                    "Scientific studies",
                    "Expert opinions",
                    "Data and statistics",
                ],
            },
            {
                "number": 4,
                "title": "The Truth Revealed",
                "duration": 1.5,
                "key_points": [
                    explanation.get("what_wrong", "What's actually wrong"),
                    explanation.get("truth", "The correct information"),
                    "Why this matters",
                ],
            },
            {
                "number": 5,
                "title": "What You Should Know",
                "duration": 1.0,
                "key_points": [
                    "Key takeaways",
                    "How to spot similar myths",
                    "Additional resources",
                ],
            },
            {
                "number": 6,
                "title": "Related Myths",
                "duration": 0.5,
                "key_points": [
                    "Preview of related topics",
                    "Call to action",
                ],
            },
        ]
        
        return {
            "chapters": chapters,
            "duration": 8.0,
        }
    
    def _generate_titles(self, claim_text: str, verdict: str) -> list[str]:
        """Generate video title suggestions."""
        short = claim_text[:40]
        return [
            f"DEBUNKED: {short}... - Here's What Science Says",
            f"Is {short} Actually True? I Investigated",
            f"The {short} Myth - BUSTED",
            f"Why Everyone Believes {short} (They're Wrong)",
            f"I Fact-Checked {short} - Surprising Results!",
            f"{short}? Let's Look at the Evidence",
            f"The Truth About {short} Will Shock You",
            f"Scientists React to {short} Claim",
            f"Breaking Down the {short} Myth",
            f"{short} - Fact or Fiction?",
        ]
    
    def _generate_next_myths(self, topic: str) -> list[str]:
        """Generate related myth suggestions."""
        myths = {
            "health": [
                "Vitamin C prevents colds",
                "You need 8 glasses of water daily",
                "Cracking knuckles causes arthritis",
                "Sugar makes children hyperactive",
                "We only use 10% of our brain",
            ],
            "history": [
                "Napoleon was short",
                "Vikings wore horned helmets",
                "The Great Wall is visible from space",
                "Columbus discovered America",
                "Medieval people thought Earth was flat",
            ],
            "science": [
                "Lightning never strikes twice",
                "Goldfish have 3-second memory",
                "Bats are blind",
                "Humans have 5 senses",
                "Evolution says we came from monkeys",
            ],
        }
        return myths.get(topic, ["Related myth 1", "Related myth 2", "Related myth 3"])[:5]
    
    def _generate_description(self, script_input: ScriptInput) -> str:
        """Generate video description."""
        return f"""In this video, we fact-check: "{script_input.claim_text}"

Verdict: {script_input.verdict}

{script_input.explanation.get('truth', '')}

#FactCheck #MythBusted #Science #Education
"""


class QACheck(Action):
    """
    Action to validate format, citations, and disclaimer presence.
    """
    
    name: str = "QACheck"
    
    async def run(self, qa_input: QAInput) -> QAOutput:
        """
        Perform QA checks on the generated content.
        
        Args:
            qa_input: Content to review
        
        Returns:
            QAOutput with issues and scores
        """
        logger.info(f"[QACheck] Reviewing claim: {qa_input.claim_text[:50]}...")
        
        issues = []
        
        # Check 1: Citation accuracy
        citation_score = self._check_citations(qa_input)
        if citation_score < 0.8:
            issues.append(QAIssue(
                category="citation",
                severity="warning",
                description="Some conclusions are not properly linked to evidence.",
                fix_suggestion="Ensure citation_map references valid evidence IDs.",
            ))
        
        # Check 2: Disclaimer presence for health/legal
        disclaimer_ok = self._check_disclaimer(qa_input)
        if not disclaimer_ok:
            issues.append(QAIssue(
                category="disclaimer",
                severity="error",
                description=f"Health/legal topic '{qa_input.topic}' requires disclaimer.",
                fix_suggestion="Add HEALTH_LEGAL_DISCLAIMER to the verdict.",
            ))
        
        # Check 3: Script format
        format_score = self._check_format(qa_input)
        if format_score < 0.8:
            issues.append(QAIssue(
                category="format",
                severity="warning",
                description="Script format does not meet specifications.",
                fix_suggestion="Ensure shorts is 30-60s and long outline has 6-10 chapters.",
            ))
        
        # Check 4: Source credibility
        credibility_ok = self._check_credibility(qa_input)
        if not credibility_ok:
            issues.append(QAIssue(
                category="credibility",
                severity="info",
                description="Evidence relies heavily on lower-credibility sources.",
                fix_suggestion="Try to include more academic or official sources.",
            ))
        
        # Calculate overall
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        
        overall_score = (citation_score + format_score + (1.0 if disclaimer_ok else 0.0)) / 3
        passed = error_count == 0 and overall_score >= 0.7
        
        return QAOutput(
            claim_id=qa_input.claim_id,
            passed=passed,
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
            citation_score=citation_score,
            format_score=format_score,
            overall_score=overall_score,
        )
    
    def _check_citations(self, qa_input: QAInput) -> float:
        """Check citation accuracy."""
        citation_map = qa_input.verdict_output.citation_map
        evidence_ids = {e.id or i+1 for i, e in enumerate(qa_input.evidence_items)}
        
        if not citation_map:
            return 0.5  # No citations, partial score
        
        valid_citations = 0
        total_citations = 0
        
        for key, ids in citation_map.items():
            for evidence_id in ids:
                total_citations += 1
                if evidence_id in evidence_ids or evidence_id <= len(qa_input.evidence_items):
                    valid_citations += 1
        
        return valid_citations / total_citations if total_citations > 0 else 0.5
    
    def _check_disclaimer(self, qa_input: QAInput) -> bool:
        """Check disclaimer presence for health/legal topics."""
        try:
            topic_enum = TopicEnum(qa_input.topic)
            if topic_enum in DISCLAIMER_TOPICS:
                return qa_input.verdict_output.disclaimer is not None
        except ValueError:
            pass
        return True  # Non-health/legal topics don't need disclaimer
    
    def _check_format(self, qa_input: QAInput) -> float:
        """Check script format compliance."""
        score = 1.0
        
        # Check shorts duration
        duration = qa_input.script_output.shorts_total_duration
        if not (30 <= duration <= 60):
            score -= 0.3
        
        # Check shorts segments
        if len(qa_input.script_output.shorts_segments) < 3:
            score -= 0.2
        
        # Check long chapters
        chapters = len(qa_input.script_output.long_chapters)
        if not (6 <= chapters <= 10):
            score -= 0.3
        
        # Check titles
        if len(qa_input.script_output.titles) < 5:
            score -= 0.2
        
        return max(0.0, score)
    
    def _check_credibility(self, qa_input: QAInput) -> bool:
        """Check if evidence has sufficient credibility."""
        if not qa_input.evidence_items:
            return False
        
        avg_credibility = sum(e.credibility_score for e in qa_input.evidence_items) / len(qa_input.evidence_items)
        return avg_credibility >= 60
