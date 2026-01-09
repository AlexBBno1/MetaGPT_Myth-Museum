"""
Myth Museum - Shorts Queue Selection

Select and prioritize packets for daily Shorts production.
Outputs a queue CSV with ranked items for video editing.
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from core.config import get_db_path, get_output_path
from core.constants import (
    QUEUE_CSV_COLUMNS,
    QUEUE_DEFAULT_MIN_CONFIDENCE,
    QUEUE_DEFAULT_SIMILARITY_THRESHOLD,
    QUEUE_SAFETY_GATES,
    QUEUE_TOPIC_MIX_TARGETS,
    ShortsStatus,
    VERDICT_WEIGHTS,
    determine_shorts_status,
)
from core.logging import get_logger
from core.textnorm import normalize_text, similarity_score

logger = get_logger(__name__)
console = Console()

# Typer CLI app
app = typer.Typer(
    name="select-for-shorts",
    help="Select and prioritize packets for daily Shorts queue",
    add_completion=False,
)


# ============================================================================
# Data Loading
# ============================================================================


def load_packets_from_db(limit: int = 100) -> list[dict[str, Any]]:
    """
    Load packets from SQLite database.
    
    Args:
        limit: Maximum number of packets to load
    
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
                packets.append(packet.packet_json)
        
        logger.info(f"Loaded {len(packets)} packets from database")
        
    except Exception as e:
        logger.error(f"Failed to load packets from database: {e}")
    
    return packets


def load_packets_from_files(
    input_dir: Optional[Path] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Load packets from JSON files in outputs/packets/.
    
    Args:
        input_dir: Directory containing packet JSON files
        limit: Maximum number of packets to load
    
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
    
    for json_path in json_files[:limit]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                packet = json.load(f)
                packets.append(packet)
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    
    logger.info(f"Loaded {len(packets)} packets from files in {input_dir}")
    return packets


# ============================================================================
# Scoring and Filtering
# ============================================================================


def score_packet(packet: dict[str, Any]) -> float:
    """
    Calculate selection score for a packet.
    
    Score = verdict_weight * confidence * evidence_bonus
    
    Args:
        packet: Packet dictionary
    
    Returns:
        Score value (higher = better candidate)
    """
    verdict = packet.get("verdict", "Unverified")
    confidence = packet.get("confidence", 0.5)
    sources = packet.get("sources", [])
    
    # Base score from verdict weight
    verdict_weight = VERDICT_WEIGHTS.get(verdict, 0)
    
    if verdict_weight == 0:
        return 0  # Unverified excluded
    
    # Evidence diversity bonus (more source types = better)
    source_types = set(s.get("source_type", "") for s in sources)
    evidence_bonus = 1.0 + (len(source_types) * 0.1)  # Up to 1.5x
    
    # Final score
    score = verdict_weight * confidence * evidence_bonus
    
    return round(score, 4)


def passes_safety_gate(packet: dict[str, Any]) -> bool:
    """
    Check if packet passes safety gate for sensitive topics.
    
    Args:
        packet: Packet dictionary
    
    Returns:
        True if safe to include, False if should be excluded
    """
    topic = packet.get("topic", "unknown").lower()
    
    if topic not in QUEUE_SAFETY_GATES:
        return True  # No gate for this topic
    
    gate = QUEUE_SAFETY_GATES[topic]
    min_conf = gate.get("min_confidence", 0.7)
    min_evidence = gate.get("min_evidence_types", 2)
    
    # Check confidence
    confidence = packet.get("confidence", 0)
    if confidence < min_conf:
        logger.debug(f"Safety gate: {topic} confidence {confidence} < {min_conf}")
        return False
    
    # Check evidence diversity
    sources = packet.get("sources", [])
    source_types = set(s.get("source_type", "") for s in sources)
    if len(source_types) < min_evidence:
        logger.debug(f"Safety gate: {topic} evidence types {len(source_types)} < {min_evidence}")
        return False
    
    return True


def filter_packets(
    packets: list[dict[str, Any]],
    min_confidence: float = QUEUE_DEFAULT_MIN_CONFIDENCE,
    exclude_unverified: bool = True,
) -> list[dict[str, Any]]:
    """
    Filter packets based on criteria.
    
    Args:
        packets: List of packets
        min_confidence: Minimum confidence threshold
        exclude_unverified: Whether to exclude Unverified verdicts
    
    Returns:
        Filtered list of packets
    """
    filtered = []
    
    for packet in packets:
        # Confidence filter
        confidence = packet.get("confidence", 0)
        if confidence < min_confidence:
            continue
        
        # Verdict filter
        verdict = packet.get("verdict", "Unverified")
        if exclude_unverified and verdict == "Unverified":
            continue
        
        # Safety gate
        if not passes_safety_gate(packet):
            continue
        
        filtered.append(packet)
    
    logger.info(f"Filtered to {len(filtered)} packets (from {len(packets)})")
    return filtered


def deduplicate_packets(
    packets: list[dict[str, Any]],
    threshold: float = QUEUE_DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Remove duplicate/similar packets.
    
    Uses textnorm similarity to detect near-duplicates.
    Keeps the one with higher score.
    
    Args:
        packets: List of packets (should be pre-scored and sorted)
        threshold: Similarity threshold (0-100)
    
    Returns:
        Deduplicated list of packets
    """
    if not packets:
        return []
    
    # Sort by score descending for deterministic selection
    sorted_packets = sorted(
        packets,
        key=lambda p: (score_packet(p), p.get("claim_id", 0)),
        reverse=True,
    )
    
    kept = []
    kept_texts = []
    
    for packet in sorted_packets:
        claim_text = packet.get("claim", "")
        normalized = normalize_text(claim_text)
        
        # Check similarity against kept packets
        is_duplicate = False
        for kept_text in kept_texts:
            sim = similarity_score(normalized, kept_text)
            if sim > threshold:
                is_duplicate = True
                logger.debug(f"Duplicate detected: similarity {sim:.1f}%")
                break
        
        if not is_duplicate:
            kept.append(packet)
            kept_texts.append(normalized)
    
    logger.info(f"Deduplicated to {len(kept)} packets (from {len(packets)})")
    return kept


def ensure_topic_mix(
    packets: list[dict[str, Any]],
    limit: int,
    targets: list[str] = QUEUE_TOPIC_MIX_TARGETS,
) -> list[dict[str, Any]]:
    """
    Ensure topic diversity in selection.
    
    Tries to include at least 1 from each target topic.
    
    Args:
        packets: List of packets (should be pre-scored)
        limit: Maximum items to select
        targets: List of topics to ensure representation
    
    Returns:
        Selected list with topic diversity
    """
    if not packets or limit <= 0:
        return []
    
    # Sort by score for deterministic selection
    sorted_packets = sorted(
        packets,
        key=lambda p: (score_packet(p), p.get("claim_id", 0)),
        reverse=True,
    )
    
    selected = []
    selected_ids = set()
    topic_counts = {t: 0 for t in targets}
    
    # First pass: ensure at least 1 from each target topic
    for target_topic in targets:
        if len(selected) >= limit:
            break
        
        for packet in sorted_packets:
            if packet.get("claim_id") in selected_ids:
                continue
            
            topic = packet.get("topic", "unknown").lower()
            
            # Map non-target topics to "unknown"
            if topic not in targets:
                topic = "unknown"
            
            if topic == target_topic:
                selected.append(packet)
                selected_ids.add(packet.get("claim_id"))
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                break
    
    # Second pass: fill remaining slots by score
    for packet in sorted_packets:
        if len(selected) >= limit:
            break
        
        if packet.get("claim_id") in selected_ids:
            continue
        
        selected.append(packet)
        selected_ids.add(packet.get("claim_id"))
        
        topic = packet.get("topic", "unknown").lower()
        if topic not in targets:
            topic = "unknown"
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Log topic distribution
    logger.info(f"Topic mix: {dict(topic_counts)}")
    
    return selected


# ============================================================================
# Queue Export
# ============================================================================


def get_shorts_folder_path(claim_id: int, shorts_dir: Path) -> Path:
    """
    Get the folder path for a shorts claim.
    
    Args:
        claim_id: Claim ID
        shorts_dir: Base shorts directory
    
    Returns:
        Path to the shorts folder
    """
    return shorts_dir / str(claim_id)


def get_relative_folder_path(claim_id: int) -> str:
    """
    Get a consistent relative path for queue CSV.
    
    Args:
        claim_id: Claim ID
    
    Returns:
        Relative path string (always uses forward slashes)
    """
    return f"outputs/shorts/{claim_id}"


def build_queue_row(
    packet: dict[str, Any],
    rank: int,
    shorts_dir: Path,
) -> dict[str, Any]:
    """
    Build a queue row from a packet.
    
    Uses the 4-state status machine:
    - needs_export: Folder missing or < 6 files
    - needs_tts: Has 6 files but no voiceover.mp3
    - ready: All files present, can render
    - rendered: Has final.mp4
    
    Args:
        packet: Packet dictionary
        rank: Position in queue
        shorts_dir: Base shorts directory
    
    Returns:
        Dictionary matching QUEUE_CSV_COLUMNS
    """
    claim_id = packet.get("claim_id", 0)
    folder_path = get_shorts_folder_path(claim_id, shorts_dir)
    
    # Determine status using the state machine
    status = determine_shorts_status(folder_path)
    
    # Get hook from shorts_script
    shorts_script = packet.get("shorts_script", {})
    hook = shorts_script.get("hook", "")[:100]
    estimated_seconds = shorts_script.get("total_duration", 35)
    
    # Get best title
    titles = packet.get("titles", [])
    title = titles[0] if titles else packet.get("claim", "")[:60]
    
    # Use consistent relative path format for CSV
    relative_path = get_relative_folder_path(claim_id)
    
    return {
        "rank": rank,
        "claim_id": claim_id,
        "topic": packet.get("topic", "unknown"),
        "verdict": packet.get("verdict", "Unknown"),
        "confidence": round(packet.get("confidence", 0), 2),
        "title": title[:80],
        "hook": hook,
        "estimated_seconds": estimated_seconds,
        "folder_path": relative_path,
        "status": status.value,
    }


def export_queue_csv(
    selected: list[dict[str, Any]],
    queue_date: str,
    output_dir: Path,
    shorts_dir: Path,
) -> Path:
    """
    Export queue to CSV file.
    
    Args:
        selected: List of selected packets
        queue_date: Date string for filename
        output_dir: Output directory for queue files
        shorts_dir: Shorts folder directory
    
    Returns:
        Path to created CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"queue_{queue_date}.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_CSV_COLUMNS)
        writer.writeheader()
        
        for i, packet in enumerate(selected, start=1):
            row = build_queue_row(packet, i, shorts_dir)
            writer.writerow(row)
    
    logger.info(f"Exported queue CSV: {csv_path}")
    return csv_path


def export_queue_md(
    selected: list[dict[str, Any]],
    queue_date: str,
    output_dir: Path,
    shorts_dir: Path,
) -> Path:
    """
    Export queue to Markdown file (human-readable).
    
    Args:
        selected: List of selected packets
        queue_date: Date string for filename
        output_dir: Output directory for queue files
        shorts_dir: Shorts folder directory
    
    Returns:
        Path to created Markdown file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"queue_{queue_date}.md"
    
    # Calculate topic and status counts
    topic_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {
        ShortsStatus.NEEDS_EXPORT.value: 0,
        ShortsStatus.NEEDS_TTS.value: 0,
        ShortsStatus.READY.value: 0,
        ShortsStatus.RENDERED.value: 0,
    }
    
    for packet in selected:
        topic = packet.get("topic", "unknown").lower()
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        claim_id = packet.get("claim_id", 0)
        folder_path = get_shorts_folder_path(claim_id, shorts_dir)
        status = determine_shorts_status(folder_path)
        status_counts[status.value] = status_counts.get(status.value, 0) + 1
    
    lines = [
        f"# Shorts Queue: {queue_date}",
        "",
        f"**Total Items:** {len(selected)}",
        "",
        "## Status Breakdown",
        "",
        f"- **Rendered:** {status_counts[ShortsStatus.RENDERED.value]}",
        f"- **Ready:** {status_counts[ShortsStatus.READY.value]}",
        f"- **Needs TTS:** {status_counts[ShortsStatus.NEEDS_TTS.value]}",
        f"- **Needs Export:** {status_counts[ShortsStatus.NEEDS_EXPORT.value]}",
        "",
        "## Topic Distribution",
        "",
    ]
    
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{topic}**: {count}")
    
    lines.extend([
        "",
        "## Queue",
        "",
        "| Rank | ID | Topic | Verdict | Confidence | Title | Status |",
        "|------|----|-------|---------|------------|-------|--------|",
    ])
    
    for i, packet in enumerate(selected, start=1):
        claim_id = packet.get("claim_id", 0)
        topic = packet.get("topic", "unknown")
        verdict = packet.get("verdict", "Unknown")
        confidence = round(packet.get("confidence", 0), 2)
        titles = packet.get("titles", [])
        title = (titles[0] if titles else packet.get("claim", ""))[:40]
        
        folder_path = get_shorts_folder_path(claim_id, shorts_dir)
        status = determine_shorts_status(folder_path)
        
        lines.append(f"| {i} | {claim_id} | {topic} | {verdict} | {confidence} | {title}... | {status.value} |")
    
    lines.extend([
        "",
        "---",
        f"*Generated: {datetime.now().isoformat()}*",
    ])
    
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Exported queue Markdown: {md_path}")
    return md_path


# ============================================================================
# Main Selection Function
# ============================================================================


def create_daily_queue(
    queue_date: Optional[str] = None,
    limit: int = 10,
    min_confidence: float = QUEUE_DEFAULT_MIN_CONFIDENCE,
    topic_mix: bool = True,
    from_files: bool = False,
    output_dir: Optional[Path] = None,
    shorts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Create daily shorts queue.
    
    Args:
        queue_date: Date for queue (default: today)
        limit: Maximum items in queue
        min_confidence: Minimum confidence threshold
        topic_mix: Whether to ensure topic diversity
        from_files: Load from files instead of DB
        output_dir: Output directory for queue files
        shorts_dir: Shorts folder directory
    
    Returns:
        Dict with queue stats: count, csv_path, md_path, topic_stats
    """
    # Set defaults
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    if output_dir is None:
        output_dir = Path("outputs/shorts_queue")
    
    if shorts_dir is None:
        shorts_dir = Path("outputs/shorts")
    
    # Load packets
    if from_files:
        packets = load_packets_from_files(limit=limit * 3)
    else:
        packets = load_packets_from_db(limit=limit * 3)
        if not packets:
            logger.info("No packets in DB, falling back to files")
            packets = load_packets_from_files(limit=limit * 3)
    
    if not packets:
        logger.warning("No packets available for queue")
        return {"count": 0, "csv_path": None, "md_path": None, "topic_stats": {}}
    
    # Filter
    filtered = filter_packets(packets, min_confidence=min_confidence)
    
    # Deduplicate
    deduped = deduplicate_packets(filtered)
    
    # Select with topic mix
    if topic_mix:
        selected = ensure_topic_mix(deduped, limit=limit)
    else:
        # Just take top N by score
        sorted_packets = sorted(
            deduped,
            key=lambda p: (score_packet(p), p.get("claim_id", 0)),
            reverse=True,
        )
        selected = sorted_packets[:limit]
    
    if not selected:
        logger.warning("No packets selected for queue")
        return {"count": 0, "csv_path": None, "md_path": None, "topic_stats": {}}
    
    # Calculate topic stats
    topic_stats: dict[str, int] = {}
    for packet in selected:
        topic = packet.get("topic", "unknown").lower()
        topic_stats[topic] = topic_stats.get(topic, 0) + 1
    
    # Export
    csv_path = export_queue_csv(selected, queue_date, output_dir, shorts_dir)
    md_path = export_queue_md(selected, queue_date, output_dir, shorts_dir)
    
    return {
        "count": len(selected),
        "csv_path": csv_path,
        "md_path": md_path,
        "topic_stats": topic_stats,
    }


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def select(
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of items in queue",
    ),
    queue_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Queue date (default: today, format: YYYY-MM-DD)",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        "-c",
        help="Minimum confidence threshold (0.0-1.0)",
    ),
    topic_mix: bool = typer.Option(
        False,
        "--topic-mix",
        help="Ensure topic diversity in queue",
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
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Output directory for queue files",
    ),
) -> None:
    """
    Select packets for daily Shorts queue.
    
    Creates queue CSV and Markdown files with ranked items.
    """
    # Validate mutually exclusive flags
    if from_db and from_files:
        console.print("[red]Error: Cannot specify both --from-db and --from-files[/red]")
        raise typer.Exit(1)
    
    # Set date
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    # Set output directory
    output_dir = Path(out_dir) if out_dir else Path("outputs/shorts_queue")
    shorts_dir = Path("outputs/shorts")
    
    console.print(f"[bold cyan]=== Shorts Queue Selection ===[/bold cyan]\n")
    console.print(f"Date: {queue_date}")
    console.print(f"Limit: {limit}")
    console.print(f"Min confidence: {min_confidence}")
    console.print(f"Topic mix: {topic_mix}")
    console.print("")
    
    # Create queue
    result = create_daily_queue(
        queue_date=queue_date,
        limit=limit,
        min_confidence=min_confidence,
        topic_mix=topic_mix,
        from_files=from_files or not from_db,
        output_dir=output_dir,
        shorts_dir=shorts_dir,
    )
    
    # Print results
    if result["count"] == 0:
        console.print("[yellow]No packets selected for queue[/yellow]")
        return
    
    console.print(f"[green]OK[/green] Created queue with {result['count']} items\n")
    
    # Show topic distribution
    console.print("[bold]Topic Distribution:[/bold]")
    for topic, count in sorted(result["topic_stats"].items(), key=lambda x: -x[1]):
        console.print(f"  - {topic}: {count}")
    
    # Show output paths
    console.print(f"\n[bold]Output Files:[/bold]")
    console.print(f"  CSV: {result['csv_path']}")
    console.print(f"  Markdown: {result['md_path']}")


@app.command()
def show(
    queue_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Queue date to show (default: today)",
    ),
    queue_dir: Optional[str] = typer.Option(
        None,
        "--dir",
        help="Queue directory",
    ),
) -> None:
    """
    Show contents of a queue file.
    """
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    base_dir = Path(queue_dir) if queue_dir else Path("outputs/shorts_queue")
    csv_path = base_dir / f"queue_{queue_date}.csv"
    
    if not csv_path.exists():
        console.print(f"[red]Queue not found: {csv_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]=== Queue: {queue_date} ===[/bold cyan]\n")
    
    # Read and display
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Create table
    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Topic")
    table.add_column("Verdict")
    table.add_column("Conf", justify="right")
    table.add_column("Title")
    table.add_column("Status")
    
    for row in rows:
        table.add_row(
            row.get("rank", ""),
            row.get("claim_id", ""),
            row.get("topic", ""),
            row.get("verdict", ""),
            row.get("confidence", ""),
            row.get("title", "")[:40] + "...",
            row.get("status", ""),
        )
    
    console.print(table)
    console.print(f"\nTotal: {len(rows)} items")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
