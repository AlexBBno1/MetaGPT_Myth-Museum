"""
Myth Museum - Shorts Export Pack

Export packets to production-ready folders for YouTube Shorts.
Each packet generates a folder with:
- voiceover.txt (narration script)
- shotlist.csv (scene breakdown)
- captions.srt (subtitles)
- metadata.json (title, hashtags, etc.)
- sources.md (source citations)
- assets_needed.md (visual asset checklist)
"""

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from core.config import get_db_path, get_output_path
from core.constants import (
    DEFAULT_HASHTAGS_BY_TOPIC,
    GENERIC_HASHTAGS,
    SHOTLIST_CSV_COLUMNS,
    SHORT_DISCLAIMERS,
    VoiceoverLimits,
    OnScreenTextLimits,
    get_disclaimer,
    needs_disclaimer,
    pick_hashtags,
    truncate_onscreen_text,
)
from core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# Typer CLI app
app = typer.Typer(
    name="export-shorts",
    help="Export packets to YouTube Shorts production folders",
    add_completion=False,
)


# ============================================================================
# Data Loading Functions
# ============================================================================


def load_packets_from_db(
    limit: int = 100,
    topic: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Load packets from SQLite database.
    
    Args:
        limit: Maximum number of packets to load
        topic: Filter by topic (optional)
        min_confidence: Minimum confidence threshold (optional)
    
    Returns:
        List of packet dictionaries
    """
    from core.db import get_connection, get_all_packets
    
    db_path = get_db_path()
    
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []
    
    packets = []
    
    try:
        with get_connection(db_path) as conn:
            db_packets = get_all_packets(conn, limit=limit)
            
            for packet in db_packets:
                packet_json = packet.packet_json
                
                # Filter by topic
                if topic and packet_json.get("topic", "").lower() != topic.lower():
                    continue
                
                # Filter by confidence
                if min_confidence is not None:
                    conf = packet_json.get("confidence", 0)
                    if conf < min_confidence:
                        continue
                
                packets.append(packet_json)
        
        logger.info(f"Loaded {len(packets)} packets from database")
        
    except Exception as e:
        logger.error(f"Failed to load packets from database: {e}")
    
    return packets


def load_packets_from_files(
    input_dir: Optional[Path] = None,
    limit: int = 100,
    topic: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Load packets from JSON files in outputs/packets/.
    
    Args:
        input_dir: Directory containing packet JSON files
        limit: Maximum number of packets to load
        topic: Filter by topic (optional)
        min_confidence: Minimum confidence threshold (optional)
    
    Returns:
        List of packet dictionaries
    """
    if input_dir is None:
        input_dir = get_output_path()
    
    if not input_dir.exists():
        logger.warning(f"Packets directory not found: {input_dir}")
        return []
    
    packets = []
    json_files = sorted(input_dir.glob("*.json"))
    
    for json_path in json_files[:limit * 2]:  # Load extra to account for filtering
        if len(packets) >= limit:
            break
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                packet_json = json.load(f)
            
            # Filter by topic
            if topic and packet_json.get("topic", "").lower() != topic.lower():
                continue
            
            # Filter by confidence
            if min_confidence is not None:
                conf = packet_json.get("confidence", 0)
                if conf < min_confidence:
                    continue
            
            packets.append(packet_json)
            
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    
    logger.info(f"Loaded {len(packets)} packets from files in {input_dir}")
    return packets


# ============================================================================
# Topic-Specific Templates for LLM
# ============================================================================

TOPIC_TEMPLATES = {
    "health": {
        "tone": "professional and caring, like a friendly doctor",
        "structure": "Start with the verdict, then explain the evidence",
        "visual_style": "medical icons, body diagrams, research graphics",
        "example_hook": "Your doctor might not tell you this, but...",
    },
    "history": {
        "tone": "storytelling and engaging, like a history teacher",
        "structure": "Tell it as a story with a timeline",
        "visual_style": "old photographs, maps, historical documents",
        "example_hook": "In 1492, something happened that textbooks got wrong...",
    },
    "science": {
        "tone": "curious and wonder-filled, like a science communicator",
        "structure": "Present the experiment or proof that settles the debate",
        "visual_style": "lab imagery, scientific diagrams, demonstrations",
        "example_hook": "Scientists tested this myth in a lab, and here's what happened...",
    },
    "psychology": {
        "tone": "relatable and insightful, like talking to a wise friend",
        "structure": "Start with a personal example everyone can relate to",
        "visual_style": "brain graphics, behavior demonstrations, social experiments",
        "example_hook": "Ever notice how you always remember the bad things? Here's why...",
    },
}

DEFAULT_TOPIC_TEMPLATE = {
    "tone": "informative and engaging",
    "structure": "Hook, evidence, conclusion",
    "visual_style": "clean graphics and text overlays",
    "example_hook": "You've probably heard this claim before...",
}


async def generate_enhanced_voiceover(packet: dict[str, Any]) -> str:
    """
    Generate enhanced voiceover using LLM for topic-specific, educational content.
    
    Creates voiceover with:
    - Specific counter-examples (not generic "it depends")
    - Topic-appropriate tone and structure
    - Educational facts that make viewers feel they learned something
    
    Falls back to template-based generation if LLM fails.
    
    Args:
        packet: Packet dictionary with claim, verdict, evidence, etc.
    
    Returns:
        Voiceover text string
    """
    from core.llm import LLMClient
    
    claim = packet.get("claim", "")
    topic = packet.get("topic", "unknown")
    verdict = packet.get("verdict", "")
    language = packet.get("language", "en")
    sources = packet.get("sources", [])
    
    # Get topic-specific template
    template = TOPIC_TEMPLATES.get(topic.lower(), DEFAULT_TOPIC_TEMPLATE)
    
    # Summarize evidence for LLM
    evidence_summary = ""
    for src in sources[:3]:  # Top 3 sources
        title = src.get("title", "")
        snippet = src.get("snippet", "")[:200] if src.get("snippet") else ""
        if title:
            evidence_summary += f"- {title}: {snippet}\n"
    
    if not evidence_summary:
        evidence_summary = "No specific evidence available."
    
    # Build LLM prompt
    is_chinese = language.lower().startswith("zh")
    
    if is_chinese:
        prompt = f"""你正在製作一個35秒的YouTube Shorts破解迷思影片。

主題: {claim}
類別: {topic}
判定: {verdict}
證據來源:
{evidence_summary}

請用{template['tone']}的語氣，生成一個voiceover腳本。

要求：
1. 開場要有「具體的反例或驚人事實」（不要用「你知道嗎」這種通用開場）
2. 解釋為什麼這個迷思會存在（歷史或文化原因）
3. 提供1-2個「具體的事實或數據」來證明/反駁
4. 結尾要有「讓人記住的收穫」

格式要求（每段分開，用空行隔開）：
[開場hook - 1句話，要具體、驚人]

[核心澄清 - 2-3句話，說明真相]

[為什麼人們會相信 - 1句話]

[具體證據 - 1-2句話，引用來源]

[結論收穫 - 1句話，讓觀眾有收穫感]

追蹤更多事實查核！

留言告訴我：你還聽過哪些迷思？"""
    else:
        prompt = f"""You are creating a 35-second YouTube Shorts myth-busting video.

Topic: {claim}
Category: {topic}  
Verdict: {verdict}
Evidence sources:
{evidence_summary}

Generate a voiceover script using a {template['tone']} tone.

Requirements:
1. Open with a SPECIFIC counter-example or surprising fact (NOT "Did you know...")
2. Explain WHY this myth exists (historical/cultural reason)
3. Provide 1-2 CONCRETE facts or statistics to prove/disprove
4. End with a MEMORABLE takeaway that makes viewers feel they learned something

Format (separate each section with blank lines):
[Opening hook - 1 sentence, specific and surprising]

[Core clarification - 2-3 sentences explaining the truth]

[Why people believe this - 1 sentence]

[Specific evidence - 1-2 sentences citing sources]

[Takeaway - 1 sentence, memorable conclusion]

Follow for more fact-checks!

Comment below: What other myths have you heard?"""
    
    # Try LLM generation
    try:
        client = LLMClient.from_config()
        
        if not client.is_configured():
            logger.warning("LLM not configured, using template-based voiceover")
            return generate_voiceover(packet)
        
        response = await client.chat([
            {"role": "system", "content": f"You are an expert myth-buster creating educational short-form video content. Use {template['tone']} tone. Be specific, not generic."},
            {"role": "user", "content": prompt},
        ], temperature=0.7, max_tokens=500)
        
        await client.close()
        
        if response and len(response) > 100:
            # Add disclaimer if needed
            disclaimer = get_disclaimer(topic, language)
            if disclaimer:
                response = response.strip() + f"\n\n{disclaimer}"
            
            logger.info(f"Generated enhanced voiceover for {topic} topic")
            return response
        else:
            logger.warning("LLM response too short, using template")
            return generate_voiceover(packet)
            
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}, using template")
        return generate_voiceover(packet)


# ============================================================================
# File Generators
# ============================================================================


def generate_voiceover(packet: dict[str, Any]) -> str:
    """
    Generate voiceover script from packet data.
    
    Creates a single narration script suitable for voice recording.
    Target length: 30-45 seconds (140-220 Chinese chars / 90-140 English words).
    
    Structure:
    1. Hook (1 sentence) - Counter-intuitive opener
    2. Core clarification (2-4 sentences) - What's wrong and why
    3. Example/analogy (1 sentence) - Concrete illustration
    4. CTA (1 sentence) - Call to action
    5. Disclaimer (if health/law topic)
    
    Args:
        packet: Packet dictionary with shorts_script and verdict data
    
    Returns:
        Voiceover text string
    """
    from core.constants import (
        VOICEOVER_MIN_ZH_CHARS,
        VOICEOVER_MIN_EN_WORDS,
        VOICEOVER_TARGET_ZH_CHARS,
        VOICEOVER_TARGET_EN_WORDS,
    )
    
    shorts_script = packet.get("shorts_script", {})
    topic = packet.get("topic", "unknown")
    language = packet.get("language", "en")
    claim = packet.get("claim", "")
    verdict = packet.get("verdict", "")
    
    is_chinese = language.lower().startswith("zh")
    lines = []
    
    # ========== 1. HOOK (Opening) ==========
    hook = shorts_script.get("hook", "")
    if hook:
        lines.append(hook)
    elif claim:
        # Generate a hook from the claim
        if is_chinese:
            lines.append(f"你有聽過「{claim[:50]}」這個說法嗎？")
        else:
            lines.append(f"Have you ever heard that {claim[:60]}?")
    
    lines.append("")
    
    # ========== 2. CORE CLARIFICATION ==========
    # Get explanation content from packet
    one_line_verdict = packet.get("one_line_verdict", "")
    what_wrong = packet.get("what_wrong", "")
    truth = packet.get("truth", "")
    why_believed = packet.get("why_believed", [])
    
    # Add one-line verdict
    if one_line_verdict:
        lines.append(one_line_verdict)
    elif verdict:
        if is_chinese:
            verdict_map = {
                "False": "這個說法是錯誤的。",
                "Misleading": "這個說法有誤導性。",
                "Depends": "這個說法要看情況。",
                "True": "這個說法是正確的。",
            }
            lines.append(verdict_map.get(verdict, f"經查證，這是{verdict}。"))
        else:
            verdict_map = {
                "False": "This claim is false.",
                "Misleading": "This claim is misleading.",
                "Depends": "This depends on the context.",
                "True": "This claim is actually true.",
            }
            lines.append(verdict_map.get(verdict, f"Our verdict: {verdict}."))
    
    # Add what's wrong explanation
    if what_wrong:
        lines.append("")
        # Split into sentences if too long
        if len(what_wrong) > 150:
            sentences = what_wrong.replace("。", "。|").replace(". ", ".|").split("|")
            for s in sentences[:3]:  # Max 3 sentences
                if s.strip():
                    lines.append(s.strip())
        else:
            lines.append(what_wrong)
    
    # Add why people believe it (adds context)
    if why_believed and len(why_believed) > 0:
        lines.append("")
        if is_chinese:
            lines.append(f"很多人相信這點，因為{why_believed[0]}")
        else:
            lines.append(f"Many people believe this because {why_believed[0].lower()}")
    
    # ========== 3. TRUTH / EXAMPLE ==========
    if truth:
        lines.append("")
        if is_chinese:
            lines.append(f"事實是：{truth}")
        else:
            lines.append(f"The truth is: {truth}")
    
    # ========== 4. CTA (Call to Action) ==========
    cta = shorts_script.get("cta", "")
    lines.append("")
    if cta:
        lines.append(cta)
    else:
        if is_chinese:
            lines.append("追蹤更多事實查核！")
        else:
            lines.append("Follow for more fact-checks!")
    
    # Engagement CTA
    lines.append("")
    if is_chinese:
        lines.append("留言告訴我：你還聽過哪些迷思？")
    else:
        lines.append("Comment below: What other myths have you heard?")
    
    # ========== 5. DISCLAIMER ==========
    disclaimer = get_disclaimer(topic, language)
    if disclaimer:
        lines.append("")
        lines.append(disclaimer)
    
    # ========== PADDING IF TOO SHORT ==========
    voiceover = "\n".join(lines)
    voiceover = _pad_voiceover_if_too_short(voiceover, packet, is_chinese)
    
    return voiceover


def _pad_voiceover_if_too_short(
    voiceover: str,
    packet: dict[str, Any],
    is_chinese: bool,
) -> str:
    """
    Add padding content if voiceover is below minimum length.
    
    Args:
        voiceover: Current voiceover text
        packet: Packet dictionary for additional content
        is_chinese: Whether the content is Chinese
    
    Returns:
        Padded voiceover text
    """
    from core.constants import (
        VOICEOVER_MIN_ZH_CHARS,
        VOICEOVER_MIN_EN_WORDS,
    )
    
    # Check current length
    if is_chinese:
        # Count Chinese characters
        char_count = sum(1 for c in voiceover if '\u4e00' <= c <= '\u9fff')
        if char_count >= VOICEOVER_MIN_ZH_CHARS:
            return voiceover
    else:
        # Count words
        word_count = len(voiceover.split())
        if word_count >= VOICEOVER_MIN_EN_WORDS:
            return voiceover
    
    # Need to add padding
    lines = voiceover.split("\n")
    
    # Get additional content for padding
    why_reasonable = packet.get("why_reasonable", "")
    sources = packet.get("sources", [])
    
    # Find insertion point (before CTA/disclaimer)
    insert_idx = len(lines) - 3  # Before engagement CTA
    if insert_idx < 0:
        insert_idx = len(lines)
    
    padding_lines = []
    
    # Add "think about it" explanation
    if why_reasonable:
        if is_chinese:
            padding_lines.append("")
            padding_lines.append(f"其實這樣想也合理，因為{why_reasonable[:100]}")
        else:
            padding_lines.append("")
            padding_lines.append(f"It's understandable why people think this, because {why_reasonable[:100].lower()}")
    
    # Add source reference for credibility
    if sources and len(sources) > 0:
        source = sources[0]
        source_name = source.get("title", source.get("source_name", ""))[:50]
        if source_name:
            if is_chinese:
                padding_lines.append(f"根據{source_name}的資料顯示...")
            else:
                padding_lines.append(f"According to {source_name}...")
    
    # Add a concrete example if still short
    topic = packet.get("topic", "")
    if is_chinese:
        example_map = {
            "health": "舉例來說，科學研究已經證實了這一點。",
            "science": "從科學角度來看，這個現象可以這樣解釋。",
            "history": "歷史記載告訴我們實際情況是這樣的。",
            "psychology": "心理學研究顯示，人們容易有這樣的誤解。",
        }
        padding_lines.append("")
        padding_lines.append(example_map.get(topic.lower(), "讓我們來看看事實是什麼。"))
    else:
        example_map = {
            "health": "For example, scientific research has consistently shown this.",
            "science": "From a scientific perspective, this phenomenon can be explained this way.",
            "history": "Historical records show us what actually happened.",
            "psychology": "Psychological research shows why people often have this misconception.",
        }
        padding_lines.append("")
        padding_lines.append(example_map.get(topic.lower(), "Let's look at what the facts actually tell us."))
    
    # Insert padding
    lines = lines[:insert_idx] + padding_lines + lines[insert_idx:]
    
    return "\n".join(lines)


def generate_shotlist_csv(packet: dict[str, Any]) -> str:
    """
    Generate shotlist CSV from packet shorts_script.
    
    Creates a scene-by-scene breakdown for video editors.
    Columns: time_start, time_end, scene, voice_line, on_screen_text, visual_suggestion, sfx
    
    Args:
        packet: Packet dictionary with shorts_script
    
    Returns:
        CSV content as string
    """
    shorts_script = packet.get("shorts_script", {})
    language = packet.get("language", "en")
    segments = shorts_script.get("segments", [])
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=SHOTLIST_CSV_COLUMNS,
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    
    for i, segment in enumerate(segments, start=1):
        time_start = segment.get("time_start", 0)
        time_end = segment.get("time_end", time_start + 5)
        narration = segment.get("narration", "")
        on_screen = segment.get("on_screen_text", "")
        visual = segment.get("visual_suggestion", "")
        
        # Truncate on-screen text
        on_screen_truncated = truncate_onscreen_text(on_screen, language)
        
        # Determine SFX based on scene type
        sfx = ""
        if i == 1:
            sfx = "whoosh"
        elif "verdict" in visual.lower() or "reveal" in visual.lower():
            sfx = "reveal_ding"
        
        writer.writerow({
            "time_start": time_start,
            "time_end": time_end,
            "scene": i,
            "voice_line": narration[:200] if narration else "",
            "on_screen_text": on_screen_truncated,
            "visual_suggestion": visual,
            "sfx": sfx,
        })
    
    return output.getvalue()


def generate_captions_srt(packet: dict[str, Any]) -> str:
    """
    Generate SRT subtitle file from shorts_script.
    
    Creates properly formatted SRT with sequential IDs and timestamps.
    Includes all narration segments plus CTA.
    
    Args:
        packet: Packet dictionary with shorts_script
    
    Returns:
        SRT content as string
    """
    shorts_script = packet.get("shorts_script", {})
    segments = shorts_script.get("segments", [])
    cta = shorts_script.get("cta", "")
    total_duration = shorts_script.get("total_duration", 35)
    language = packet.get("language", "en")
    
    srt_blocks = []
    block_num = 0
    last_end_time = 0
    
    # Process each segment
    for segment in segments:
        time_start = segment.get("time_start", last_end_time)
        time_end = segment.get("time_end", time_start + 5)
        narration = segment.get("narration", "")
        
        if not narration:
            continue
        
        block_num += 1
        last_end_time = time_end
        
        # Format timestamps: HH:MM:SS,mmm
        start_ts = _seconds_to_srt_time(time_start)
        end_ts = _seconds_to_srt_time(time_end)
        
        # Split long narration - may need multiple subtitle blocks for very long text
        subtitle_lines = _split_subtitle_text(narration, max_lines=3)
        
        srt_block = f"{block_num}\n{start_ts} --> {end_ts}\n{subtitle_lines}\n"
        srt_blocks.append(srt_block)
    
    # Add CTA as additional subtitle block
    if cta:
        block_num += 1
        cta_start = last_end_time
        cta_end = min(cta_start + 4, total_duration)  # 4 seconds for CTA
        
        start_ts = _seconds_to_srt_time(cta_start)
        end_ts = _seconds_to_srt_time(cta_end)
        
        subtitle_lines = _split_subtitle_text(cta, max_lines=2)
        srt_block = f"{block_num}\n{start_ts} --> {end_ts}\n{subtitle_lines}\n"
        srt_blocks.append(srt_block)
        last_end_time = cta_end
    
    # Add engagement prompt ("Comment below...")
    engagement_text = "留言告訴我：你還聽過哪些迷思？" if language.startswith("zh") else "Comment below: What other myths have you heard?"
    block_num += 1
    engage_start = last_end_time
    engage_end = min(engage_start + 3, total_duration + 5)  # 3 seconds for engagement
    
    start_ts = _seconds_to_srt_time(engage_start)
    end_ts = _seconds_to_srt_time(engage_end)
    
    subtitle_lines = _split_subtitle_text(engagement_text, max_lines=2)
    srt_block = f"{block_num}\n{start_ts} --> {end_ts}\n{subtitle_lines}\n"
    srt_blocks.append(srt_block)
    
    return "\n".join(srt_blocks)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_subtitle_text(text: str, max_line_length: int = 42, max_lines: int = 3) -> str:
    """
    Split text into subtitle-friendly lines.
    
    Args:
        text: Text to split
        max_line_length: Maximum characters per line
        max_lines: Maximum lines per subtitle block (default 3)
    
    Returns:
        Text formatted with newlines for subtitle display
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_line_length and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(" ".join(current_line))
    
    # Return all lines up to max_lines (don't truncate content)
    return "\n".join(lines[:max_lines]) if len(lines) > max_lines else "\n".join(lines)


def generate_metadata_json(packet: dict[str, Any]) -> dict[str, Any]:
    """
    Generate metadata JSON for YouTube upload.
    
    Includes title, description, hashtags, and other metadata.
    
    Args:
        packet: Packet dictionary
    
    Returns:
        Metadata dictionary
    """
    titles = packet.get("titles", [])
    topic = packet.get("topic", "unknown")
    language = packet.get("language", "en")
    
    # Pick best title (shortest that still has hook appeal)
    # Strategy: prefer titles with "?" or "!" as they tend to be more engaging
    primary_title = ""
    alternative_titles = []
    
    if titles:
        # Sort by length, prefer shorter titles
        sorted_titles = sorted(titles, key=len)
        
        # Find one with hook indicators (?, !, DEBUNKED, etc.)
        hook_indicators = ["?", "!", "DEBUNKED", "BUSTED", "Truth", "Actually"]
        for title in sorted_titles:
            if any(indicator in title for indicator in hook_indicators):
                primary_title = title
                break
        
        if not primary_title:
            primary_title = sorted_titles[0] if sorted_titles else "Myth Busted"
        
        alternative_titles = [t for t in titles if t != primary_title]
    
    # Generate hashtags
    hashtags = pick_hashtags(topic, count=10)
    
    # Build description with sources
    description = packet.get("description", "")
    if not description:
        claim = packet.get("claim", "")[:100]
        verdict = packet.get("verdict", "Unknown")
        description = f"Fact-checking: {claim}...\n\nVerdict: {verdict}"
    
    # Get topic-specific visual style suggestions
    template = TOPIC_TEMPLATES.get(topic.lower(), DEFAULT_TOPIC_TEMPLATE)
    visual_style = template.get("visual_style", "clean graphics and text overlays")
    
    metadata = {
        "title": primary_title,
        "alternatives": alternative_titles[:5],
        "description": description,
        "hashtags": hashtags,
        "claim_id": packet.get("claim_id"),
        "topic": topic,
        "verdict": packet.get("verdict"),
        "confidence": packet.get("confidence"),
        "language": language,
        "created_at": packet.get("created_at", datetime.now().isoformat()),
        # Topic-specific production hints
        "style_hints": {
            "visual_style": visual_style,
            "tone": template.get("tone", "informative and engaging"),
            "structure": template.get("structure", "Hook, evidence, conclusion"),
        },
    }
    
    # Add cartoon myth prompts and shorts topic
    from pipeline.cartoon_myth_generator import add_cartoon_myth_to_metadata
    metadata = add_cartoon_myth_to_metadata(metadata, packet)
    
    return metadata


def generate_sources_md(packet: dict[str, Any]) -> str:
    """
    Generate sources markdown with citations.
    
    Lists all sources with URLs and explains citation mapping.
    
    Args:
        packet: Packet dictionary with sources and citation_map
    
    Returns:
        Markdown content as string
    """
    sources = packet.get("sources", [])
    citation_map = packet.get("citation_map", {})
    claim = packet.get("claim", "")[:100]
    
    lines = [
        "# Sources",
        "",
        f"**Claim:** {claim}...",
        "",
        "## References",
        "",
    ]
    
    # Build source lookup by evidence_id
    source_lookup = {}
    for src in sources:
        eid = src.get("evidence_id")
        if eid:
            source_lookup[eid] = src
    
    # List all sources
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "Unknown Source")
        url = src.get("url", "")
        source_type = src.get("source_type", "unknown")
        credibility = src.get("credibility_score", 0)
        
        lines.append(f"{i}. **{title}**")
        if url:
            lines.append(f"   - URL: {url}")
        lines.append(f"   - Type: {source_type} (credibility: {credibility})")
        lines.append("")
    
    # Add citation map explanation
    if citation_map:
        lines.extend([
            "## Citation Map",
            "",
            "Which conclusions are supported by which sources:",
            "",
        ])
        
        for conclusion, evidence_ids in citation_map.items():
            if not evidence_ids:
                continue
            
            # Get source titles for these evidence IDs
            cited_sources = []
            for eid in evidence_ids[:3]:  # Limit to 3 citations per conclusion
                src = source_lookup.get(eid)
                if src:
                    cited_sources.append(src.get("title", f"Source #{eid}"))
            
            if cited_sources:
                sources_text = ", ".join(cited_sources)
                lines.append(f"- **{conclusion}**: {sources_text}")
        
        lines.append("")
    
    lines.extend([
        "---",
        f"*Generated: {datetime.now().isoformat()}*",
    ])
    
    return "\n".join(lines)


def generate_assets_needed_md(packet: dict[str, Any]) -> str:
    """
    Generate assets checklist markdown.
    
    Categorizes visual requirements by type and availability.
    
    Args:
        packet: Packet dictionary with visuals and shorts_script
    
    Returns:
        Markdown content as string
    """
    visuals = packet.get("visuals", [])
    shorts_script = packet.get("shorts_script", {})
    segments = shorts_script.get("segments", [])
    
    # Collect all visual suggestions
    all_visuals = list(visuals)
    for segment in segments:
        visual = segment.get("visual_suggestion", "")
        if visual and visual not in all_visuals:
            all_visuals.append(visual)
    
    # Categorize visuals
    categories = {
        "B-roll / Stock Footage": [],
        "Charts / Infographics": [],
        "Text Animation": [],
        "Self-made / Original": [],
        "Need Filming": [],
    }
    
    # Keywords for categorization
    broll_keywords = ["footage", "b-roll", "stock", "clip", "video", "image"]
    chart_keywords = ["chart", "graph", "data", "infographic", "diagram", "visualization"]
    text_keywords = ["text", "overlay", "title", "animation", "reveal"]
    filming_keywords = ["interview", "person", "street", "filming", "shoot"]
    
    for visual in all_visuals:
        visual_lower = visual.lower()
        
        if any(kw in visual_lower for kw in chart_keywords):
            categories["Charts / Infographics"].append(visual)
        elif any(kw in visual_lower for kw in filming_keywords):
            categories["Need Filming"].append(visual)
        elif any(kw in visual_lower for kw in text_keywords):
            categories["Text Animation"].append(visual)
        elif any(kw in visual_lower for kw in broll_keywords):
            categories["B-roll / Stock Footage"].append(visual)
        else:
            categories["Self-made / Original"].append(visual)
    
    # Build markdown
    lines = [
        "# Assets Needed",
        "",
        "Visual assets required for this Shorts video.",
        "",
    ]
    
    for category, items in categories.items():
        if not items:
            continue
        
        # Add availability note
        availability = ""
        if category == "B-roll / Stock Footage":
            availability = " (CC0/Public Domain available on Pexels, Pixabay)"
        elif category == "Charts / Infographics":
            availability = " (Create using Canva, After Effects, or similar)"
        elif category == "Text Animation":
            availability = " (Create in CapCut/剪映)"
        elif category == "Need Filming":
            availability = " (Requires original filming)"
        
        lines.append(f"## {category}{availability}")
        lines.append("")
        
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. {item}")
        
        lines.append("")
    
    # Add notes
    lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- Prefer royalty-free assets from: Pexels, Pixabay, Unsplash",
        "- For charts: Use accurate data from sources listed in sources.md",
        "- For text animations: Keep text short and readable on mobile",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Export Orchestration
# ============================================================================


async def export_shorts_pack(
    packet: dict[str, Any],
    output_dir: Path,
    overwrite: bool = False,
    use_llm: bool = False,
) -> Optional[Path]:
    """
    Export a single packet to a shorts production folder.
    
    Creates outputs/shorts/{claim_id}/ with all 6 files.
    
    Args:
        packet: Packet dictionary
        output_dir: Base output directory for shorts
        overwrite: Whether to overwrite existing files
        use_llm: Use LLM to generate enhanced, topic-specific voiceover
    
    Returns:
        Path to created folder, or None if skipped
    """
    claim_id = packet.get("claim_id")
    
    if claim_id is None:
        logger.warning("Packet missing claim_id, skipping")
        return None
    
    # Create folder
    folder = output_dir / str(claim_id)
    
    if folder.exists() and not overwrite:
        # Check if all files exist
        required_files = [
            "voiceover.txt",
            "shotlist.csv",
            "captions.srt",
            "metadata.json",
            "sources.md",
            "assets_needed.md",
        ]
        all_exist = all((folder / f).exists() for f in required_files)
        
        if all_exist:
            logger.debug(f"Folder {folder} already complete, skipping (use --overwrite)")
            return None
    
    folder.mkdir(parents=True, exist_ok=True)
    
    # Generate and write files
    try:
        # 1. voiceover.txt - Use LLM-enhanced version if requested
        if use_llm:
            voiceover = await generate_enhanced_voiceover(packet)
        else:
            voiceover = generate_voiceover(packet)
        (folder / "voiceover.txt").write_text(voiceover, encoding="utf-8")
        
        # 2. shotlist.csv
        shotlist = generate_shotlist_csv(packet)
        (folder / "shotlist.csv").write_text(shotlist, encoding="utf-8")
        
        # 3. captions.srt
        captions = generate_captions_srt(packet)
        (folder / "captions.srt").write_text(captions, encoding="utf-8")
        
        # 4. metadata.json
        metadata = generate_metadata_json(packet)
        (folder / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # 5. sources.md
        sources = generate_sources_md(packet)
        (folder / "sources.md").write_text(sources, encoding="utf-8")
        
        # 6. assets_needed.md
        assets = generate_assets_needed_md(packet)
        (folder / "assets_needed.md").write_text(assets, encoding="utf-8")
        
        logger.debug(f"Exported shorts pack: {folder}")
        return folder
        
    except Exception as e:
        logger.error(f"Failed to export shorts pack for claim {claim_id}: {e}")
        return None


async def process_packets_for_shorts(
    conn=None,
    limit: int = 100,
    topic: Optional[str] = None,
    from_files: bool = False,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    min_confidence: Optional[float] = None,
    use_llm: bool = False,
) -> int:
    """
    Process packets and export shorts production folders.
    
    Args:
        conn: Database connection (optional, for DB mode)
        limit: Maximum packets to process
        topic: Filter by topic
        from_files: If True, load from files instead of DB
        output_dir: Output directory for shorts folders
        overwrite: Whether to overwrite existing files
        min_confidence: Minimum confidence threshold
        use_llm: Use LLM to generate enhanced, topic-specific voiceover
    
    Returns:
        Number of shorts folders exported
    """
    # Determine output directory
    if output_dir is None:
        output_dir = Path("outputs/shorts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load packets
    if from_files:
        logger.info("Loading packets from files")
        packets = load_packets_from_files(
            limit=limit,
            topic=topic,
            min_confidence=min_confidence,
        )
    else:
        # Try database first, fallback to files
        logger.info("Loading packets from database")
        packets = load_packets_from_db(
            limit=limit,
            topic=topic,
            min_confidence=min_confidence,
        )
        
        if not packets:
            logger.info("No packets in database, falling back to files")
            packets = load_packets_from_files(
                limit=limit,
                topic=topic,
                min_confidence=min_confidence,
            )
    
    if not packets:
        logger.warning("No packets to export")
        return 0
    
    # Export each packet
    exported = 0
    for packet in packets:
        folder = await export_shorts_pack(packet, output_dir, overwrite=overwrite, use_llm=use_llm)
        if folder:
            exported += 1
    
    logger.info(f"Exported {exported} shorts folders to {output_dir}")
    return exported


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def export(
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of packets to export",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Filter by topic (health, science, history, etc.)",
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Output directory (default: outputs/shorts)",
    ),
    from_db: bool = typer.Option(
        False,
        "--from-db",
        help="Load packets from database",
    ),
    from_files: bool = typer.Option(
        False,
        "--from-files",
        help="Load packets from JSON files",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing shorts folders",
    ),
    min_confidence: Optional[float] = typer.Option(
        None,
        "--min-confidence",
        help="Minimum confidence threshold (0.0-1.0)",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Filter by language (en, zh)",
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Use LLM to generate enhanced, topic-specific voiceover",
    ),
) -> None:
    """
    Export packets to YouTube Shorts production folders.
    
    Creates outputs/shorts/{claim_id}/ with voiceover, shotlist, captions,
    metadata, sources, and assets_needed files.
    
    Use --use-llm for enhanced, topic-specific content with:
    - Specific counter-examples (not generic)
    - Topic-appropriate tone and structure
    - Educational facts that make viewers feel they learned something
    """
    import asyncio
    
    # Validate mutually exclusive flags
    if from_db and from_files:
        console.print("[red]Error: Cannot specify both --from-db and --from-files[/red]")
        raise typer.Exit(1)
    
    # Determine source
    use_files = from_files
    if not from_db and not from_files:
        # Default: try DB first
        use_files = False
    
    # Determine output directory
    output_dir = Path(out_dir) if out_dir else Path("outputs/shorts")
    
    console.print(f"[bold cyan]=== Shorts Export ===[/bold cyan]\n")
    console.print(f"Source: {'Files' if use_files else 'Database (fallback to files)'}")
    console.print(f"Output: {output_dir}")
    if topic:
        console.print(f"Topic filter: {topic}")
    console.print("")
    
    # Show LLM status
    if use_llm:
        console.print("[yellow]LLM-enhanced voiceover enabled[/yellow]")
    
    # Run export
    exported = asyncio.run(process_packets_for_shorts(
        limit=limit,
        topic=topic,
        from_files=use_files,
        output_dir=output_dir,
        overwrite=overwrite,
        min_confidence=min_confidence,
        use_llm=use_llm,
    ))
    
    # Print summary
    console.print(f"\n[green]OK[/green] Exported {exported} shorts folders")
    
    if exported > 0:
        console.print(f"\nOutput location: {output_dir.absolute()}")
        
        # List exported folders
        folders = sorted(output_dir.iterdir())[:5]
        if folders:
            console.print("\nExported folders:")
            for folder in folders:
                if folder.is_dir():
                    console.print(f"  • {folder.name}/")
            if len(list(output_dir.iterdir())) > 5:
                console.print(f"  ... and {len(list(output_dir.iterdir())) - 5} more")


@app.command()
def show(
    claim_id: int = typer.Argument(..., help="Claim ID to show"),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Shorts output directory",
    ),
) -> None:
    """
    Show contents of an exported shorts folder.
    """
    output_dir = Path(out_dir) if out_dir else Path("outputs/shorts")
    folder = output_dir / str(claim_id)
    
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]=== Shorts Pack: {claim_id} ===[/bold cyan]\n")
    
    files = [
        ("voiceover.txt", "Voiceover Script"),
        ("shotlist.csv", "Shot List"),
        ("captions.srt", "Captions"),
        ("metadata.json", "Metadata"),
        ("sources.md", "Sources"),
        ("assets_needed.md", "Assets"),
    ]
    
    for filename, label in files:
        filepath = folder / filename
        if filepath.exists():
            size = filepath.stat().st_size
            console.print(f"[green]OK[/green] {label}: {filename} ({size} bytes)")
        else:
            console.print(f"[red]X[/red] {label}: {filename} (missing)")
    
    # Show voiceover preview
    voiceover_path = folder / "voiceover.txt"
    if voiceover_path.exists():
        content = voiceover_path.read_text(encoding="utf-8")
        preview = content[:200] + "..." if len(content) > 200 else content
        console.print(f"\n[bold]Voiceover Preview:[/bold]")
        console.print(preview)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
