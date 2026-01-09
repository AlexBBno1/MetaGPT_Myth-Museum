"""
Myth Museum - FastAPI Application

REST API for viewing and managing fact-check packets.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config import get_db_path, get_output_path, load_config
from core.db import (
    get_all_packets,
    get_claims_by_status,
    get_claims_by_topic,
    get_connection,
    get_evidence_by_claim,
    get_packet_by_claim,
    get_table_counts,
    get_verdict_by_claim,
)
from core.logging import get_logger
from core.models import Claim, Evidence, Packet, Verdict

logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Myth Museum API",
    description="REST API for the Myth Museum fact-checking content pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================


class StatusResponse(BaseModel):
    """Database status response."""
    status: str = "ok"
    tables: dict[str, int] = Field(default_factory=dict)
    topics: dict[str, int] = Field(default_factory=dict)
    timestamp: str = ""


class PacketSummary(BaseModel):
    """Summary of a packet for list views."""
    claim_id: int
    claim: str
    verdict: str
    confidence: float
    topic: str
    created_at: str


class PacketDetail(BaseModel):
    """Full packet details."""
    claim_id: int
    packet_json: dict[str, Any]
    created_at: str


class ClaimResponse(BaseModel):
    """Claim details response."""
    id: int
    claim_text: str
    topic: str
    language: str
    score: int
    status: str
    created_at: str


class EvidenceResponse(BaseModel):
    """Evidence details response."""
    id: int
    claim_id: int
    query: str
    source_name: str
    source_type: str
    url: str
    title: str
    snippet: str
    credibility_score: int


class VerdictResponse(BaseModel):
    """Verdict details response."""
    id: int
    claim_id: int
    verdict: str
    confidence: float
    explanation_json: dict[str, Any]
    created_at: str


class PipelineRunRequest(BaseModel):
    """Request to trigger pipeline run."""
    limit: int = Field(default=10, ge=1, le=100)
    min_score: int = Field(default=50, ge=0, le=100)
    topic: Optional[str] = None
    use_llm: bool = False


class PipelineRunResponse(BaseModel):
    """Response after triggering pipeline."""
    status: str = "started"
    message: str = ""
    task_id: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Myth Museum API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status",
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get database status and statistics."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            tables = get_table_counts(conn)
            topics = get_claims_by_topic(conn)
        
        return StatusResponse(
            status="ok",
            tables=tables,
            topics=topics,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/packets", response_model=list[PacketSummary])
async def list_packets(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    topic: Optional[str] = Query(default=None),
):
    """List all packets with optional filtering."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            packets = get_all_packets(conn, limit=limit + offset)
        
        # Apply offset
        packets = packets[offset:offset + limit]
        
        # Filter by topic if specified
        if topic:
            packets = [p for p in packets if p.packet_json.get("topic") == topic]
        
        return [
            PacketSummary(
                claim_id=p.claim_id,
                claim=p.packet_json.get("claim", "")[:100],
                verdict=p.packet_json.get("verdict", "Unknown"),
                confidence=p.packet_json.get("confidence", 0.0),
                topic=p.packet_json.get("topic", "unknown"),
                created_at=p.packet_json.get("created_at", p.created_at.isoformat() if p.created_at else ""),
            )
            for p in packets
        ]
    except Exception as e:
        logger.error(f"Failed to list packets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/packets/{claim_id}", response_model=PacketDetail)
async def get_packet(claim_id: int):
    """Get a specific packet by claim ID."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            packet = get_packet_by_claim(conn, claim_id)
        
        if not packet:
            raise HTTPException(status_code=404, detail=f"Packet not found for claim {claim_id}")
        
        return PacketDetail(
            claim_id=packet.claim_id,
            packet_json=packet.packet_json,
            created_at=packet.created_at.isoformat() if packet.created_at else "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get packet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/claims", response_model=list[ClaimResponse])
async def list_claims(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    topic: Optional[str] = Query(default=None),
    min_score: int = Query(default=0, ge=0, le=100),
):
    """List claims with optional filtering."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            if status:
                claims = get_claims_by_status(conn, status, limit=limit, min_score=min_score)
            else:
                # Get all claims
                cursor = conn.execute(
                    "SELECT * FROM claims WHERE score >= ? ORDER BY created_at DESC LIMIT ?",
                    (min_score, limit)
                )
                rows = cursor.fetchall()
                claims = [
                    Claim(
                        id=row["id"],
                        raw_item_id=row["raw_item_id"],
                        claim_text=row["claim_text"],
                        topic=row["topic"],
                        language=row["language"],
                        score=row["score"],
                        status=row["status"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
        
        # Filter by topic
        if topic:
            claims = [c for c in claims if c.topic == topic]
        
        return [
            ClaimResponse(
                id=c.id,
                claim_text=c.claim_text,
                topic=c.topic,
                language=c.language,
                score=c.score,
                status=c.status,
                created_at=c.created_at.isoformat() if c.created_at else "",
            )
            for c in claims
        ]
    except Exception as e:
        logger.error(f"Failed to list claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/claims/{claim_id}/evidence", response_model=list[EvidenceResponse])
async def get_claim_evidence(claim_id: int):
    """Get evidence for a specific claim."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            evidence_list = get_evidence_by_claim(conn, claim_id)
        
        return [
            EvidenceResponse(
                id=e.id,
                claim_id=e.claim_id,
                query=e.query,
                source_name=e.source_name,
                source_type=e.source_type,
                url=e.url,
                title=e.title,
                snippet=e.snippet[:500],
                credibility_score=e.credibility_score,
            )
            for e in evidence_list
        ]
    except Exception as e:
        logger.error(f"Failed to get evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/claims/{claim_id}/verdict", response_model=VerdictResponse)
async def get_claim_verdict(claim_id: int):
    """Get verdict for a specific claim."""
    db_path = get_db_path()
    
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        with get_connection(db_path) as conn:
            verdict = get_verdict_by_claim(conn, claim_id)
        
        if not verdict:
            raise HTTPException(status_code=404, detail=f"Verdict not found for claim {claim_id}")
        
        return VerdictResponse(
            id=verdict.id,
            claim_id=verdict.claim_id,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            explanation_json=verdict.explanation_json,
            created_at=verdict.created_at.isoformat() if verdict.created_at else "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get verdict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for pipeline
_running_tasks: dict[str, bool] = {}


async def _run_pipeline_task(task_id: str, request: PipelineRunRequest):
    """Background task to run pipeline."""
    logger.info(f"Starting pipeline task {task_id}")
    
    try:
        from pipeline.build_evidence import process_claims_for_evidence
        from pipeline.extract_claims import process_raw_items
        from pipeline.generate_scripts import process_verdicts_for_scripts
        from pipeline.judge_claim import process_claims_for_verdict
        
        db_path = get_db_path()
        
        with get_connection(db_path) as conn:
            # Run pipeline stages
            await process_raw_items(conn, use_llm=request.use_llm, limit=request.limit)
            conn.commit()
            
            await process_claims_for_evidence(
                conn,
                min_score=request.min_score,
                topic=request.topic,
                limit=request.limit,
            )
            conn.commit()
            
            await process_claims_for_verdict(conn, use_llm=request.use_llm, limit=request.limit)
            conn.commit()
            
            await process_verdicts_for_scripts(conn, use_llm=request.use_llm, limit=request.limit)
            conn.commit()
        
        logger.info(f"Pipeline task {task_id} completed")
        
    except Exception as e:
        logger.error(f"Pipeline task {task_id} failed: {e}")
    finally:
        _running_tasks.pop(task_id, None)


@app.post("/run", response_model=PipelineRunResponse)
async def trigger_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a pipeline run (async)."""
    # Check if a task is already running
    if _running_tasks:
        return PipelineRunResponse(
            status="busy",
            message="A pipeline task is already running",
        )
    
    # Generate task ID
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _running_tasks[task_id] = True
    
    # Add background task
    background_tasks.add_task(_run_pipeline_task, task_id, request)
    
    return PipelineRunResponse(
        status="started",
        message=f"Pipeline started with limit={request.limit}, min_score={request.min_score}",
        task_id=task_id,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
