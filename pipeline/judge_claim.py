"""
Myth Museum - Claim Judge

Generate verdicts for claims based on evidence.
"""

from pathlib import Path
from typing import Any, Optional

from core.constants import (
    DISCLAIMER_TOPICS,
    HEALTH_LEGAL_DISCLAIMER,
    ClaimStatusEnum,
    TopicEnum,
    VerdictEnum,
)
from core.db import (
    get_claims_by_status,
    get_evidence_by_claim,
    get_verdict_by_claim,
    insert_verdict,
    update_claim_status,
)
from core.logging import get_logger
from core.models import Claim, Evidence, ExplanationJSON, Verdict

logger = get_logger(__name__)

# Path to judge prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge_prompt.txt"


def load_judge_prompt() -> str:
    """Load the judge prompt template."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a fact-checker. Evaluate the claim based on evidence."


def judge_rule_based(claim: Claim, evidence_list: list[Evidence]) -> Verdict:
    """
    Generate verdict using rule-based heuristics.
    
    Args:
        claim: The claim to judge
        evidence_list: List of evidence items
    
    Returns:
        Verdict with explanation
    """
    if not evidence_list:
        return _create_verdict(
            claim_id=claim.id,
            verdict=VerdictEnum.UNVERIFIED,
            confidence=0.3,
            explanation=ExplanationJSON(
                one_line_verdict="Insufficient evidence to verify this claim.",
                why_believed=["This is a commonly shared belief"],
                what_wrong="We could not find enough evidence to verify or refute this claim.",
                why_reasonable="Without clear evidence, it's understandable to be uncertain.",
                truth="More research is needed to confirm or deny this claim.",
                citation_map={},
            ),
        )
    
    # Calculate average credibility
    avg_credibility = sum(e.credibility_score for e in evidence_list) / len(evidence_list)
    
    # Check for factcheck sources (they often have explicit verdicts)
    factcheck_evidence = [e for e in evidence_list if e.source_type == "factcheck"]
    academic_evidence = [e for e in evidence_list if e.source_type == "academic"]
    
    # Simple heuristic based on evidence types and credibility
    if factcheck_evidence:
        # Factcheck sources tend to debunk myths
        verdict = VerdictEnum.FALSE
        confidence = min(0.8, 0.6 + len(factcheck_evidence) * 0.1)
    elif academic_evidence and avg_credibility >= 80:
        # Strong academic evidence
        verdict = VerdictEnum.DEPENDS
        confidence = min(0.75, 0.5 + avg_credibility / 200)
    elif avg_credibility >= 70:
        verdict = VerdictEnum.MISLEADING
        confidence = 0.6
    else:
        verdict = VerdictEnum.UNVERIFIED
        confidence = 0.4
    
    # Build citation map
    evidence_ids = [e.id for e in evidence_list if e.id]
    citation_map = {
        "one_line_verdict": evidence_ids[:2],
        "why_believed": evidence_ids[:1],
        "what_wrong": evidence_ids,
        "truth": evidence_ids[:2],
    }
    
    # Build explanation
    why_believed = [
        "This belief is commonly shared in popular culture",
        "Similar claims have been circulating online",
        "The claim sounds plausible based on common understanding",
    ]
    
    what_wrong = _generate_what_wrong(claim, evidence_list, verdict)
    truth = _generate_truth(claim, evidence_list, verdict)
    
    explanation = ExplanationJSON(
        one_line_verdict=f"Based on {len(evidence_list)} sources, this claim is {verdict.value.lower()}.",
        why_believed=why_believed,
        what_wrong=what_wrong,
        why_reasonable="This misconception is understandable given how the information is often presented.",
        truth=truth,
        citation_map=citation_map,
    )
    
    return _create_verdict(claim.id, verdict, confidence, explanation)


def _generate_what_wrong(claim: Claim, evidence_list: list[Evidence], verdict: VerdictEnum) -> str:
    """Generate what_wrong explanation."""
    if verdict == VerdictEnum.FALSE:
        return f"The claim '{claim.claim_text[:50]}...' is not supported by credible evidence. The available sources suggest this is a misconception."
    elif verdict == VerdictEnum.MISLEADING:
        return "While there may be some truth to this claim, it oversimplifies or misrepresents the full picture."
    elif verdict == VerdictEnum.DEPENDS:
        return "The accuracy of this claim depends on specific context and conditions that are not specified."
    elif verdict == VerdictEnum.TRUE:
        return "The evidence supports this claim."
    else:
        return "There is insufficient evidence to determine the accuracy of this claim."


def _generate_truth(claim: Claim, evidence_list: list[Evidence], verdict: VerdictEnum) -> str:
    """Generate the truth statement."""
    if verdict == VerdictEnum.FALSE:
        return "The scientific consensus does not support this claim."
    elif verdict == VerdictEnum.MISLEADING:
        return "The reality is more nuanced than this simplified claim suggests."
    elif verdict == VerdictEnum.DEPENDS:
        return "The answer depends on specific circumstances and context."
    elif verdict == VerdictEnum.TRUE:
        return f"The claim is accurate: {claim.claim_text[:80]}"
    else:
        return "More research is needed to establish the facts."


def _create_verdict(
    claim_id: int,
    verdict: VerdictEnum,
    confidence: float,
    explanation: ExplanationJSON,
) -> Verdict:
    """Create a Verdict instance."""
    return Verdict(
        claim_id=claim_id,
        verdict=verdict.value,
        explanation_json=explanation.model_dump(),
        confidence=confidence,
    )


def add_disclaimer_if_needed(verdict: Verdict, topic: str) -> Verdict:
    """
    Add disclaimer to verdict if topic requires it.
    
    Args:
        verdict: The verdict to potentially modify
        topic: The claim topic
    
    Returns:
        Modified verdict with disclaimer if needed
    """
    try:
        topic_enum = TopicEnum(topic)
    except ValueError:
        return verdict
    
    if topic_enum in DISCLAIMER_TOPICS:
        explanation = verdict.explanation_json.copy()
        explanation["disclaimer"] = HEALTH_LEGAL_DISCLAIMER
        verdict.explanation_json = explanation
    
    return verdict


async def judge_claim_llm(
    claim: Claim,
    evidence_list: list[Evidence],
    llm_client,
) -> Verdict:
    """
    Generate verdict using LLM.
    
    Args:
        claim: The claim to judge
        evidence_list: List of evidence items
        llm_client: LLM client instance
    
    Returns:
        Verdict with explanation
    """
    if not llm_client.is_configured():
        logger.warning("LLM not configured, falling back to rule-based")
        return judge_rule_based(claim, evidence_list)
    
    # Load prompt
    system_prompt = load_judge_prompt()
    
    # Build evidence text
    evidence_text = ""
    for i, e in enumerate(evidence_list):
        evidence_text += f"""
Evidence #{e.id or i+1}:
- Source: {e.source_name} ({e.source_type})
- Title: {e.title}
- URL: {e.url}
- Content: {e.snippet[:300]}
- Credibility Score: {e.credibility_score}/100
"""
    
    user_prompt = f"""
CLAIM: {claim.claim_text}

TOPIC: {claim.topic}

EVIDENCE:
{evidence_text}

Please evaluate this claim and return a JSON verdict.
"""
    
    try:
        result = await llm_client.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        
        # Parse result
        verdict_str = result.get("verdict", "Unverified")
        try:
            verdict_enum = VerdictEnum(verdict_str)
        except ValueError:
            verdict_enum = VerdictEnum.UNVERIFIED
        
        explanation = ExplanationJSON(
            one_line_verdict=result.get("one_line_verdict", ""),
            why_believed=result.get("why_believed", []),
            what_wrong=result.get("what_wrong", ""),
            why_reasonable=result.get("why_reasonable", ""),
            truth=result.get("truth", ""),
            citation_map=result.get("citation_map", {}),
        )
        
        return _create_verdict(
            claim_id=claim.id,
            verdict=verdict_enum,
            confidence=result.get("confidence", 0.7),
            explanation=explanation,
        )
        
    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return judge_rule_based(claim, evidence_list)


def judge_claim_baseline(claim: Claim, evidence_list: list[Evidence]) -> Verdict:
    """
    Baseline verdict generation (alias for rule_based).
    
    Args:
        claim: The claim to judge
        evidence_list: List of evidence items
    
    Returns:
        Verdict
    """
    return judge_rule_based(claim, evidence_list)


async def process_claims_for_verdict(
    conn,
    use_llm: bool = False,
    limit: int = 100,
    llm_client=None,
) -> int:
    """
    Process claims to generate verdicts.
    
    Args:
        conn: Database connection
        use_llm: Whether to use LLM for verdict generation
        limit: Maximum claims to process
        llm_client: Optional LLM client
    
    Returns:
        Number of verdicts generated
    """
    # Get claims with evidence
    claims = get_claims_by_status(conn, ClaimStatusEnum.HAS_EVIDENCE.value, limit=limit)
    
    if not claims:
        logger.info("No claims ready for verdict")
        return 0
    
    logger.info(f"Generating verdicts for {len(claims)} claims")
    
    # Create LLM client if needed
    if use_llm and llm_client is None:
        from core.llm import LLMClient
        llm_client = LLMClient.from_config()
    
    processed = 0
    for claim in claims:
        try:
            # Get evidence
            evidence_list = get_evidence_by_claim(conn, claim.id)
            
            # Check if already has verdict
            existing = get_verdict_by_claim(conn, claim.id)
            if existing:
                logger.debug(f"Claim {claim.id} already has verdict")
                continue
            
            # Generate verdict
            if use_llm:
                verdict = await judge_claim_llm(claim, evidence_list, llm_client)
            else:
                verdict = judge_rule_based(claim, evidence_list)
            
            # Add disclaimer if needed
            verdict = add_disclaimer_if_needed(verdict, claim.topic)
            
            # Save verdict
            insert_verdict(conn, verdict)
            update_claim_status(conn, claim.id, ClaimStatusEnum.JUDGED.value)
            
            processed += 1
            logger.debug(f"Generated verdict for claim {claim.id}: {verdict.verdict}")
            
        except Exception as e:
            logger.error(f"Error generating verdict for claim {claim.id}: {e}")
    
    logger.info(f"Generated {processed} verdicts")
    return processed
