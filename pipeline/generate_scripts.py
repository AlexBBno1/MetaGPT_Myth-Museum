"""
Myth Museum - Script Generator

Generate video scripts and packet output.

Features:
- Direct myth script generation from topic (no database required)
- LLM-powered script generation for myth-busting videos
- Template-based fallback for quick generation
- Database-backed packet generation for full pipeline
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import get_output_path
from core.constants import (
    LONG_OUTLINE_CHAPTERS,
    SHORTS_SEGMENT_SECONDS,
    SHORTS_TOTAL_SECONDS,
    ClaimStatusEnum,
)
from core.db import (
    get_claims_by_status,
    get_evidence_by_claim,
    get_packet_by_claim,
    get_verdict_by_claim,
    insert_packet,
    update_claim_status,
)
from core.llm import LLMClient
from core.logging import get_logger
from core.models import (
    ChapterOutline,
    Claim,
    Evidence,
    LongOutline,
    Packet,
    PacketJSON,
    ShortsScript,
    ShortsSegment,
    SourceReference,
    Verdict,
)

logger = get_logger(__name__)


# ============================================================================
# Direct Myth Script Generation (No Database Required)
# ============================================================================

MYTH_SCRIPT_PROMPT = '''You are a viral YouTube Shorts script writer specializing in myth-busting educational content.

Write a 45-60 second voiceover script about this topic:

TOPIC: {topic}
{myth_context}

SCRIPT REQUIREMENTS:
1. Total length: 110-150 words (for 45-60 seconds at natural pace)
2. Structure:
   - Hook (0-5s): Attention-grabbing opening that challenges a common belief
   - Setup (5-15s): What everyone thinks is true
   - Twist (15-25s): "But wait..." - introduce doubt
   - Evidence (25-40s): The scientific/historical truth
   - Reveal (40-50s): The real explanation
   - CTA (50-60s): Engaging call to action

3. Writing style:
   - Conversational, like talking to a friend
   - Short punchy sentences
   - Use "you" to address viewer directly
   - Include specific numbers/facts for credibility
   - End with a memorable takeaway

4. Avoid:
   - Starting with "Did you know"
   - Ending with generic "Subscribe for more"
   - Using complex jargon
   - Being preachy or condescending

OUTPUT: Just the script text, no timestamps or formatting. Each paragraph is a natural pause point for image transitions.'''


async def generate_myth_script(
    topic: str,
    myth: str = "",
    truth: str = "",
    llm_client: Optional[LLMClient] = None,
) -> str:
    """
    Generate a myth-busting voiceover script using LLM.
    
    Args:
        topic: The topic (e.g., "Da Vinci Mona Lisa")
        myth: The myth/misconception (optional, for context)
        truth: The true fact (optional, for context)
        llm_client: Optional LLM client (creates one if not provided)
    
    Returns:
        Script text (110-150 words, 45-60 seconds when spoken)
    """
    # Create LLM client if needed
    client = llm_client
    if client is None:
        client = LLMClient.from_config()
    
    # Build context from myth and truth
    myth_context = ""
    if myth:
        myth_context += f"\nMYTH: {myth}"
    if truth:
        myth_context += f"\nTRUTH: {truth}"
    
    # Check if LLM is configured
    if not client.is_configured():
        logger.warning("LLM not configured, using template script")
        return _generate_template_script(topic, myth, truth)
    
    try:
        prompt = MYTH_SCRIPT_PROMPT.format(
            topic=topic,
            myth_context=myth_context,
        )
        
        messages = [
            {"role": "system", "content": "You write viral educational scripts. Output only the script text."},
            {"role": "user", "content": prompt},
        ]
        
        response = await client.chat(messages, temperature=0.7)
        
        if response:
            script = response.strip()
            
            # Validate length
            word_count = len(script.split())
            if word_count < 80:
                logger.warning(f"Script too short ({word_count} words), regenerating...")
                # Try again with emphasis on length
                messages[1]["content"] += "\n\nIMPORTANT: Make sure the script is at least 110 words!"
                response = await client.chat(messages, temperature=0.8)
                if response:
                    script = response.strip()
            
            logger.info(f"Generated myth script: {len(script.split())} words")
            return script
            
    except Exception as e:
        logger.error(f"LLM script generation failed: {e}")
    
    # Fallback to template
    return _generate_template_script(topic, myth, truth)


def _generate_template_script(topic: str, myth: str = "", truth: str = "") -> str:
    """Generate a template-based script as fallback."""
    # Extract key elements
    topic_clean = topic.replace("_", " ").title()
    
    if not myth:
        myth = f"There's a common belief about {topic_clean} that everyone accepts without question."
    
    if not truth:
        truth = f"The real story behind {topic_clean} is far more interesting than the myth."
    
    return f"""Everyone thinks they know the truth about {topic_clean}.

{myth}

But here's the thing most people miss.

Scientists and historians have actually studied this extensively. And what they found might surprise you.

{truth}

The myth persists because it's simple and memorable. But the truth? It's actually more fascinating.

Next time someone mentions {topic_clean}, you'll know the real story.

Follow for more myth busters!"""


async def generate_myth_script_with_structure(
    topic: str,
    myth: str = "",
    truth: str = "",
    num_segments: int = 6,
    llm_client: Optional[LLMClient] = None,
) -> list[dict]:
    """
    Generate a structured myth script with segments for each image.
    
    Args:
        topic: The topic
        myth: The myth (optional)
        truth: The truth (optional)
        num_segments: Number of segments (default 6)
        llm_client: Optional LLM client
    
    Returns:
        List of segment dicts with 'text' and 'duration' keys
    """
    # Generate full script first
    full_script = await generate_myth_script(topic, myth, truth, llm_client)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in full_script.split('\n\n') if p.strip()]
    
    # If we have fewer paragraphs than segments, split further
    if len(paragraphs) < num_segments:
        # Split by sentences and regroup
        import re
        sentences = re.split(r'(?<=[.!?])\s+', full_script)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into segments
        sentences_per_segment = max(1, len(sentences) // num_segments)
        paragraphs = []
        for i in range(num_segments):
            start = i * sentences_per_segment
            end = start + sentences_per_segment if i < num_segments - 1 else len(sentences)
            paragraphs.append(' '.join(sentences[start:end]))
    
    # Distribute to segments
    segments = []
    words_per_second = 2.5
    target_duration = 50  # seconds
    duration_per_segment = target_duration / num_segments
    
    for i in range(num_segments):
        if i < len(paragraphs):
            text = paragraphs[i]
        else:
            text = paragraphs[-1] if paragraphs else f"Segment {i+1}"
        
        # Estimate duration based on word count
        word_count = len(text.split())
        estimated_duration = word_count / words_per_second
        
        segments.append({
            "segment_num": i + 1,
            "text": text,
            "word_count": word_count,
            "estimated_duration": estimated_duration,
        })
    
    return segments

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "script_prompt.txt"


def load_script_prompt() -> str:
    """Load the script prompt template."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a video script writer."


def generate_shorts_script_baseline(verdict: Verdict, claim_text: str) -> dict[str, Any]:
    """
    Generate shorts script using templates.
    
    Args:
        verdict: The verdict for the claim
        claim_text: The claim text
    
    Returns:
        Shorts script dict
    """
    explanation = verdict.explanation_json
    verdict_type = verdict.verdict
    
    # Hook based on verdict
    hooks = {
        "False": f"You've been told {claim_text[:50]}... But is it actually true?",
        "Misleading": f"There's something wrong with {claim_text[:40]}... Let me explain.",
        "Depends": f"{claim_text[:50]}... It's complicated.",
        "True": f"Is {claim_text[:50]} actually true? Yes, and here's why.",
        "Unverified": f"Can we verify {claim_text[:50]}? Let's find out.",
    }
    
    hook = hooks.get(verdict_type, f"Let's fact-check: {claim_text[:50]}")
    
    # Generate segments
    segments = [
        ShortsSegment(
            time_start=0,
            time_end=5,
            narration=hook,
            visual_suggestion="Hook visual with claim text overlay",
            on_screen_text=claim_text[:40],
        ),
        ShortsSegment(
            time_start=5,
            time_end=12,
            narration=explanation.get("one_line_verdict", f"This claim is {verdict_type.lower()}."),
            visual_suggestion="Verdict reveal animation",
            on_screen_text=verdict_type.upper(),
        ),
        ShortsSegment(
            time_start=12,
            time_end=25,
            narration=explanation.get("what_wrong", "Here's what the evidence shows...")[:150],
            visual_suggestion="Evidence/data visualization",
            on_screen_text="The Truth",
        ),
        ShortsSegment(
            time_start=25,
            time_end=35,
            narration=explanation.get("truth", "The correct answer is...")[:100],
            visual_suggestion="Summary graphic",
            on_screen_text="Remember This",
        ),
    ]
    
    return ShortsScript(
        hook=hook,
        segments=segments,
        cta="Follow for more fact-checks! Link in bio for sources.",
        total_duration=35,
    ).model_dump()


def generate_long_outline_baseline(verdict: Verdict, claim_text: str) -> dict[str, Any]:
    """
    Generate long-form video outline using templates.
    
    Args:
        verdict: The verdict for the claim
        claim_text: The claim text
    
    Returns:
        Long outline dict
    """
    explanation = verdict.explanation_json
    
    chapters = [
        ChapterOutline(
            chapter_number=1,
            title="The Myth Everyone Believes",
            duration_minutes=1.5,
            key_points=[
                f"Introduction to the claim: {claim_text[:60]}",
                "Why this myth is so widespread",
                "Personal anecdotes and examples",
            ],
            visual_suggestions=[
                "Social media screenshots of the myth",
                "Person-on-street interviews",
            ],
        ),
        ChapterOutline(
            chapter_number=2,
            title="Where Did This Come From?",
            duration_minutes=1.5,
            key_points=[
                "Origins of the belief",
                "Why people find it convincing",
                explanation.get("why_reasonable", "Historical context"),
            ],
            visual_suggestions=[
                "Historical footage or images",
                "Timeline animation",
            ],
        ),
        ChapterOutline(
            chapter_number=3,
            title="What Does The Evidence Say?",
            duration_minutes=2.0,
            key_points=[
                "Scientific studies and findings",
                "Expert opinions",
                "Data and statistics",
            ],
            visual_suggestions=[
                "Charts and graphs",
                "Expert interview clips",
            ],
        ),
        ChapterOutline(
            chapter_number=4,
            title="The Truth Revealed",
            duration_minutes=1.5,
            key_points=[
                explanation.get("what_wrong", "What's actually wrong"),
                explanation.get("truth", "The correct information"),
                "Why this matters",
            ],
            visual_suggestions=[
                "Verdict animation",
                "Summary infographic",
            ],
        ),
        ChapterOutline(
            chapter_number=5,
            title="What You Should Know",
            duration_minutes=1.0,
            key_points=[
                "Key takeaways",
                "How to spot similar myths",
                "Additional resources",
            ],
            visual_suggestions=[
                "Checklist graphic",
                "Source links",
            ],
        ),
        ChapterOutline(
            chapter_number=6,
            title="Related Myths to Watch Out For",
            duration_minutes=0.5,
            key_points=[
                "Preview of related topics",
                "Call to action",
            ],
            visual_suggestions=[
                "Thumbnail montage of related videos",
            ],
        ),
    ]
    
    return LongOutline(
        chapters=chapters,
        total_duration_minutes=8.0,
    ).model_dump()


def generate_titles(claim_text: str, verdict: str) -> list[str]:
    """Generate video title suggestions."""
    claim_short = claim_text[:40]
    
    return [
        f"DEBUNKED: {claim_short}... - Here's What Science Says",
        f"Is {claim_short} Actually True? I Investigated",
        f"The {claim_short} Myth - BUSTED",
        f"Why Everyone Believes {claim_short} (They're Wrong)",
        f"I Fact-Checked {claim_short} - Surprising Results!",
        f"{claim_short}? Let's Look at the Evidence",
        f"The Truth About {claim_short} Will Shock You",
        f"Scientists React to {claim_short} Claim",
        f"Breaking Down the {claim_short} Myth",
        f"{claim_short} - Fact or Fiction?",
    ]


def generate_next_myths(claim_text: str, topic: str) -> list[str]:
    """Generate related myth suggestions."""
    topic_myths = {
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
    
    return topic_myths.get(topic, [
        "Related myth 1",
        "Related myth 2",
        "Related myth 3",
        "Related myth 4",
        "Related myth 5",
    ])[:5]


def generate_packet(
    claim: Claim,
    verdict: Verdict,
    evidence_list: list[Evidence],
) -> PacketJSON:
    """
    Generate complete packet JSON.
    
    Args:
        claim: The claim
        verdict: The verdict
        evidence_list: List of evidence
    
    Returns:
        PacketJSON instance
    """
    explanation = verdict.explanation_json
    
    # Generate scripts
    shorts = generate_shorts_script_baseline(verdict, claim.claim_text)
    outline = generate_long_outline_baseline(verdict, claim.claim_text)
    
    # Build source references
    sources = [
        SourceReference(
            evidence_id=e.id or 0,
            title=e.title,
            url=e.url,
            source_type=e.source_type,
            credibility_score=e.credibility_score,
        )
        for e in evidence_list
    ]
    
    # Generate metadata
    titles = generate_titles(claim.claim_text, verdict.verdict)
    next_myths = generate_next_myths(claim.claim_text, claim.topic)
    
    # Build description
    sources_text = "\n".join(f"- {s.title}: {s.url}" for s in sources[:5])
    description = f"""In this video, we fact-check: "{claim.claim_text}"

Verdict: {verdict.verdict}

{explanation.get('truth', '')}

SOURCES:
{sources_text}

#FactCheck #MythBusted #Science #Education
"""
    
    return PacketJSON(
        claim=claim.claim_text,
        claim_id=claim.id,
        topic=claim.topic,
        language=claim.language,
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        one_line_verdict=explanation.get("one_line_verdict", ""),
        why_believed=explanation.get("why_believed", []),
        what_wrong=explanation.get("what_wrong", ""),
        why_reasonable=explanation.get("why_reasonable", ""),
        truth=explanation.get("truth", ""),
        citation_map=explanation.get("citation_map", {}),
        disclaimer=explanation.get("disclaimer"),
        shorts_script=shorts,
        long_outline=outline,
        thumbnail_text_suggestions=[
            "MYTH BUSTED!",
            f"{verdict.verdict.upper()}",
            "THE TRUTH",
        ],
        titles=titles,
        description=description,
        visuals=[
            "Opening hook animation",
            "Evidence presentation graphics",
            "Verdict reveal animation",
            "Summary infographic",
        ],
        next_myths=next_myths,
        sources=sources,
        created_at=datetime.now().isoformat(),
    )


async def generate_scripts_llm(
    verdict: Verdict,
    claim: Claim,
    llm_client,
) -> dict[str, Any]:
    """
    Generate scripts using LLM.
    
    Args:
        verdict: The verdict
        claim: The claim
        llm_client: LLM client
    
    Returns:
        Script generation result
    """
    if not llm_client.is_configured():
        return {
            "shorts_script": generate_shorts_script_baseline(verdict, claim.claim_text),
            "long_outline": generate_long_outline_baseline(verdict, claim.claim_text),
        }
    
    system_prompt = load_script_prompt()
    
    user_prompt = f"""
CLAIM: {claim.claim_text}
VERDICT: {verdict.verdict}
EXPLANATION: {json.dumps(verdict.explanation_json, indent=2)}

Generate video scripts for this fact-check.
"""
    
    try:
        result = await llm_client.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return result
    except Exception as e:
        logger.error(f"LLM script generation failed: {e}")
        return {
            "shorts_script": generate_shorts_script_baseline(verdict, claim.claim_text),
            "long_outline": generate_long_outline_baseline(verdict, claim.claim_text),
        }


def export_packet_json(packet: Packet, output_dir: Path) -> Path:
    """
    Export packet to JSON file.
    
    Args:
        packet: The packet to export
        output_dir: Output directory
    
    Returns:
        Path to exported file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"{packet.claim_id}.json"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(packet.packet_json, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Exported JSON: {file_path}")
    return file_path


def export_packet_md(packet: Packet, output_dir: Path) -> Path:
    """
    Export packet to Markdown file.
    
    Args:
        packet: The packet to export
        output_dir: Output directory
    
    Returns:
        Path to exported file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = packet.packet_json
    file_path = output_dir / f"{packet.claim_id}.md"
    
    # Build markdown
    md = f"""# Fact Check: {data.get('claim', '')}

**Verdict:** {data.get('verdict', 'Unknown')}
**Confidence:** {data.get('confidence', 0):.0%}
**Topic:** {data.get('topic', 'Unknown')}

---

## Summary

{data.get('one_line_verdict', '')}

## Why People Believe This

"""
    
    for reason in data.get('why_believed', []):
        md += f"- {reason}\n"
    
    md += f"""
## What's Wrong

{data.get('what_wrong', '')}

## Why It's Understandable

{data.get('why_reasonable', '')}

## The Truth

{data.get('truth', '')}

"""
    
    if data.get('disclaimer'):
        md += f"""
---

{data.get('disclaimer')}

"""
    
    md += """
---

## Sources

"""
    
    for source in data.get('sources', []):
        if isinstance(source, dict):
            md += f"- [{source.get('title', 'Source')}]({source.get('url', '')})\n"
    
    md += f"""
---

*Generated: {data.get('created_at', '')}*
"""
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    logger.debug(f"Exported MD: {file_path}")
    return file_path


async def process_verdicts_for_scripts(
    conn,
    use_llm: bool = False,
    limit: int = 100,
    llm_client=None,
) -> int:
    """
    Process verdicts to generate scripts and packets.
    
    Args:
        conn: Database connection
        use_llm: Whether to use LLM for script generation
        limit: Maximum claims to process
        llm_client: Optional LLM client
    
    Returns:
        Number of packets created
    """
    # Get judged claims
    claims = get_claims_by_status(conn, ClaimStatusEnum.JUDGED.value, limit=limit)
    
    if not claims:
        logger.info("No claims ready for script generation")
        return 0
    
    logger.info(f"Generating scripts for {len(claims)} claims")
    
    # Create LLM client if needed
    if use_llm and llm_client is None:
        from core.llm import LLMClient
        llm_client = LLMClient.from_config()
    
    output_dir = get_output_path()
    processed = 0
    
    for claim in claims:
        try:
            # Check if already has packet
            existing = get_packet_by_claim(conn, claim.id)
            if existing:
                logger.debug(f"Claim {claim.id} already has packet")
                continue
            
            # Get verdict and evidence
            verdict = get_verdict_by_claim(conn, claim.id)
            if not verdict:
                logger.warning(f"No verdict for claim {claim.id}")
                continue
            
            evidence_list = get_evidence_by_claim(conn, claim.id)
            
            # Generate packet
            packet_json = generate_packet(claim, verdict, evidence_list)
            
            # Optionally enhance with LLM
            if use_llm:
                llm_result = await generate_scripts_llm(verdict, claim, llm_client)
                if llm_result.get("shorts_script"):
                    packet_json.shorts_script = llm_result["shorts_script"]
                if llm_result.get("long_outline"):
                    packet_json.long_outline = llm_result["long_outline"]
            
            # Save packet
            packet = Packet(
                claim_id=claim.id,
                packet_json=packet_json.model_dump(),
            )
            insert_packet(conn, packet)
            
            # Export files
            export_packet_json(packet, output_dir)
            export_packet_md(packet, output_dir)
            
            # Update status
            update_claim_status(conn, claim.id, ClaimStatusEnum.PACKAGED.value)
            
            processed += 1
            logger.debug(f"Generated packet for claim {claim.id}")
            
        except Exception as e:
            logger.error(f"Error generating packet for claim {claim.id}: {e}")
    
    logger.info(f"Generated {processed} packets")
    return processed
