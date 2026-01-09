"""
Myth Museum - MetaGPT Orchestrator

Team setup and pipeline execution using MetaGPT.

Note: This module requires MetaGPT to be installed and accessible.
If MetaGPT is not available, it will fall back to the local pipeline.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root and MetaGPT root to path for imports
project_root = Path(__file__).parent.parent
metagpt_root = project_root.parent.parent  # Go up to MetaGPT root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(metagpt_root) not in sys.path:
    sys.path.insert(0, str(metagpt_root))

try:
    from metagpt.logs import logger
    from metagpt.schema import Message
    from metagpt.team import Team
    METAGPT_AVAILABLE = True
except ImportError:
    from core.logging import get_logger
    logger = get_logger(__name__)
    
    class Message:
        """Stub Message when MetaGPT not available."""
        pass
    
    class Team:
        """Stub Team when MetaGPT not available."""
        def __init__(self):
            self.env = None
        
        def hire(self, roles):
            pass
        
        def invest(self, investment):
            pass
        
        def run_project(self, idea):
            pass
        
        async def run(self, n_round=1):
            raise NotImplementedError("MetaGPT is not installed")
    
    METAGPT_AVAILABLE = False
    logger.warning("MetaGPT not available. Will use local pipeline fallback.")

from core.config import get_output_path
from core.constants import ClaimStatusEnum
from core.db import (
    get_claims_by_status,
    get_connection,
    get_evidence_by_claim,
    get_verdict_by_claim,
    insert_evidence,
    insert_packet,
    insert_verdict,
    update_claim_status,
)
from core.logging import get_logger
from core.models import Claim, Evidence, Packet, Verdict
from metagpt_integration.roles import (
    FactChecker,
    QAReviewer,
    Researcher,
    ScriptWriter,
)
from metagpt_integration.schemas import (
    ClaimInput,
    EvidenceItem,
    EvidenceOutput,
    PacketJSON,
    PipelineContext,
    QAOutput,
    ScriptOutput,
    VerdictOutput,
)
from pipeline.generate_scripts import export_packet_json, export_packet_md

local_logger = get_logger(__name__)


def create_myth_team() -> Team:
    """
    Create a MetaGPT Team for fact-checking.
    
    Returns:
        Team with Researcher, FactChecker, ScriptWriter, and QAReviewer
    """
    team = Team()
    
    team.hire([
        Researcher(),
        FactChecker(),
        ScriptWriter(),
        QAReviewer(),
    ])
    
    logger.info("Created Myth Museum team with 4 roles")
    
    return team


async def run_metagpt_pipeline(
    claims: list[Claim],
    conn,
    investment: float = 10.0,
    n_round: int = 4,  # One round per role
) -> list[int]:
    """
    Run the MetaGPT pipeline for a list of claims.
    
    Args:
        claims: List of claims to process
        conn: Database connection
        investment: Budget for LLM calls
        n_round: Number of rounds to run
    
    Returns:
        List of claim IDs that were successfully processed
    """
    if not METAGPT_AVAILABLE:
        local_logger.warning("MetaGPT not available, cannot run MetaGPT pipeline")
        raise ImportError("MetaGPT is not installed")
    
    processed_claim_ids = []
    output_dir = get_output_path()
    
    for claim in claims:
        local_logger.info(f"Processing claim {claim.id}: {claim.claim_text[:50]}...")
        
        try:
            # Create team for this claim
            team = create_myth_team()
            team.invest(investment=investment)
            
            # Create claim input
            claim_input = ClaimInput(
                claim_id=claim.id,
                claim_text=claim.claim_text,
                topic=claim.topic,
                language=claim.language,
                score=claim.score,
                raw_item_id=claim.raw_item_id,
            )
            
            # Run the project
            team.run_project(claim_input.to_message_content())
            
            # Run the team
            await team.run(n_round=n_round)
            
            # Extract results from team environment
            context = _extract_pipeline_results(team)
            
            if context.verdict_output:
                # Save verdict to DB
                verdict = _convert_verdict_output(context.verdict_output)
                insert_verdict(conn, verdict)
                update_claim_status(conn, claim.id, ClaimStatusEnum.JUDGED.value)
                
                # Save evidence to DB
                if context.evidence_output:
                    for evidence_item in context.evidence_output.evidence_items:
                        evidence = _convert_evidence_item(claim.id, evidence_item)
                        insert_evidence(conn, evidence)
                
                # Generate and save packet if scripts available
                if context.script_output:
                    packet_json = _build_packet_json(
                        claim, context.verdict_output, context.script_output,
                        context.evidence_output.evidence_items if context.evidence_output else []
                    )
                    
                    packet = Packet(
                        claim_id=claim.id,
                        packet_json=packet_json.model_dump(),
                    )
                    insert_packet(conn, packet)
                    
                    # Export files
                    export_packet_json(packet, output_dir)
                    export_packet_md(packet, output_dir)
                    
                    update_claim_status(conn, claim.id, ClaimStatusEnum.PACKAGED.value)
                
                processed_claim_ids.append(claim.id)
                local_logger.info(f"Successfully processed claim {claim.id}")
            else:
                local_logger.warning(f"No verdict generated for claim {claim.id}")
                
        except Exception as e:
            local_logger.error(f"Error processing claim {claim.id}: {e}")
            update_claim_status(conn, claim.id, ClaimStatusEnum.FAILED.value)
    
    return processed_claim_ids


def _extract_pipeline_results(team: Team) -> PipelineContext:
    """
    Extract pipeline results from team environment.
    
    Args:
        team: The MetaGPT team
    
    Returns:
        PipelineContext with extracted outputs
    """
    context = PipelineContext(
        claim_id=0,
        claim_text="",
        topic="unknown",
    )
    
    # Get messages from team memory
    try:
        if team.env and hasattr(team.env, 'memory'):
            messages = team.env.memory.get()
            
            for msg in messages:
                if hasattr(msg, 'instruct_content'):
                    content = msg.instruct_content
                    
                    if isinstance(content, EvidenceOutput):
                        context.evidence_output = content
                        context.claim_id = content.claim_id
                        context.claim_text = content.claim_text
                    elif isinstance(content, VerdictOutput):
                        context.verdict_output = content
                        context.claim_id = content.claim_id
                        context.claim_text = content.claim_text
                    elif isinstance(content, ScriptOutput):
                        context.script_output = content
                    elif isinstance(content, QAOutput):
                        context.qa_output = content
    except Exception as e:
        local_logger.warning(f"Error extracting pipeline results: {e}")
    
    return context


def _convert_verdict_output(verdict_output: VerdictOutput) -> Verdict:
    """Convert VerdictOutput to Verdict model."""
    return Verdict(
        claim_id=verdict_output.claim_id,
        verdict=verdict_output.verdict,
        explanation_json={
            "one_line_verdict": verdict_output.one_line_verdict,
            "why_believed": verdict_output.why_believed,
            "what_wrong": verdict_output.what_wrong,
            "why_reasonable": verdict_output.why_reasonable,
            "truth": verdict_output.truth,
            "citation_map": verdict_output.citation_map,
            "disclaimer": verdict_output.disclaimer,
        },
        confidence=verdict_output.confidence,
    )


def _convert_evidence_item(claim_id: int, item: EvidenceItem) -> Evidence:
    """Convert EvidenceItem to Evidence model."""
    return Evidence(
        claim_id=claim_id,
        query=item.query,
        source_name=item.source_name,
        source_type=item.source_type,
        url=item.url,
        title=item.title,
        snippet=item.snippet,
        credibility_score=item.credibility_score,
    )


def _build_packet_json(
    claim: Claim,
    verdict_output: VerdictOutput,
    script_output: ScriptOutput,
    evidence_items: list[EvidenceItem],
) -> PacketJSON:
    """Build PacketJSON from pipeline outputs."""
    from pipeline.generate_scripts import SourceReference
    
    sources = [
        {
            "evidence_id": e.id or i+1,
            "title": e.title,
            "url": e.url,
            "source_type": e.source_type,
            "credibility_score": e.credibility_score,
        }
        for i, e in enumerate(evidence_items)
    ]
    
    return PacketJSON(
        claim=claim.claim_text,
        claim_id=claim.id,
        topic=claim.topic,
        language=claim.language,
        verdict=verdict_output.verdict,
        confidence=verdict_output.confidence,
        one_line_verdict=verdict_output.one_line_verdict,
        why_believed=verdict_output.why_believed,
        what_wrong=verdict_output.what_wrong,
        why_reasonable=verdict_output.why_reasonable,
        truth=verdict_output.truth,
        citation_map=verdict_output.citation_map,
        disclaimer=verdict_output.disclaimer,
        shorts_script={
            "hook": script_output.shorts_hook,
            "segments": [s.model_dump() for s in script_output.shorts_segments],
            "cta": script_output.shorts_cta,
            "total_duration": script_output.shorts_total_duration,
        },
        long_outline={
            "chapters": script_output.long_chapters,
            "total_duration_minutes": script_output.long_total_duration,
        },
        thumbnail_text_suggestions=script_output.thumbnail_suggestions,
        titles=script_output.titles,
        description=script_output.description,
        visuals=["Hook animation", "Evidence graphics", "Verdict reveal", "Summary"],
        next_myths=script_output.next_myths,
        sources=sources,
        created_at=datetime.now().isoformat(),
    )


async def process_claims_metagpt(
    conn,
    limit: int = 10,
    min_score: int = 0,
    topic: Optional[str] = None,
) -> int:
    """
    Process claims using MetaGPT orchestration.
    
    Args:
        conn: Database connection
        limit: Maximum claims to process
        min_score: Minimum claim score
        topic: Filter by topic (optional)
    
    Returns:
        Number of claims processed
    """
    # Get claims ready for processing (status = 'new')
    claims = get_claims_by_status(conn, ClaimStatusEnum.NEW.value, limit=limit, min_score=min_score)
    
    if topic:
        claims = [c for c in claims if c.topic == topic]
    
    if not claims:
        local_logger.info("No claims to process with MetaGPT")
        return 0
    
    local_logger.info(f"Processing {len(claims)} claims with MetaGPT orchestrator")
    
    processed_ids = await run_metagpt_pipeline(claims, conn)
    
    return len(processed_ids)


# Fallback to local pipeline if MetaGPT fails
async def process_claims_with_fallback(
    conn,
    limit: int = 10,
    min_score: int = 0,
    topic: Optional[str] = None,
    use_metagpt: bool = True,
) -> int:
    """
    Process claims with MetaGPT, falling back to local if needed.
    
    Args:
        conn: Database connection
        limit: Maximum claims to process
        min_score: Minimum claim score
        topic: Filter by topic
        use_metagpt: Whether to try MetaGPT first
    
    Returns:
        Number of claims processed
    """
    if use_metagpt:
        try:
            return await process_claims_metagpt(conn, limit, min_score, topic)
        except Exception as e:
            local_logger.error(f"MetaGPT orchestration failed: {e}")
            local_logger.info("Falling back to local pipeline...")
    
    # Fall back to local pipeline
    from pipeline.build_evidence import process_claims_for_evidence
    from pipeline.extract_claims import process_raw_items
    from pipeline.generate_scripts import process_verdicts_for_scripts
    from pipeline.judge_claim import process_claims_for_verdict
    
    # Run local pipeline stages
    extracted = await process_raw_items(conn, limit=limit)
    evidence_count = await process_claims_for_evidence(conn, min_score=min_score, topic=topic, limit=limit)
    verdicts_count = await process_claims_for_verdict(conn, limit=limit)
    packets_count = await process_verdicts_for_scripts(conn, limit=limit)
    
    return packets_count
