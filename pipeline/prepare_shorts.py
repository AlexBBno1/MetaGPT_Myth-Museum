"""
Myth Museum - Prepare Shorts for Production

Auto-complete missing steps in shorts folders to make them ready for rendering.
Reads queue CSV and ensures all required files exist with valid content.
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from core.config import get_output_path
from core.constants import (
    SHORTS_REQUIRED_FILES,
    ShortsStatus,
    SRT_TIMING_ADJUSTMENT_THRESHOLD,
    VOICEOVER_MIN_ZH_CHARS,
    VOICEOVER_MIN_EN_WORDS,
    VOICEOVER_MIN_DURATION_SECONDS,
    determine_shorts_status,
)
from core.logging import get_logger

logger = get_logger(__name__)
console = Console()


# ============================================================================
# Voiceover Validation
# ============================================================================


def check_voiceover_text_length(voiceover_path: Path) -> tuple[bool, str, int]:
    """
    Check if voiceover.txt meets minimum length requirements.
    
    Args:
        voiceover_path: Path to voiceover.txt
    
    Returns:
        Tuple of (is_valid, reason, actual_length)
    """
    if not voiceover_path.exists():
        return False, "voiceover.txt does not exist", 0
    
    try:
        text = voiceover_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, f"Cannot read voiceover.txt: {e}", 0
    
    if not text:
        return False, "voiceover.txt is empty", 0
    
    # Detect language by checking for Chinese characters
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    is_chinese = chinese_chars > total_chars * 0.3
    
    if is_chinese:
        # Check Chinese character count
        if chinese_chars < VOICEOVER_MIN_ZH_CHARS:
            return (
                False,
                f"Voiceover too short: {chinese_chars} Chinese chars < {VOICEOVER_MIN_ZH_CHARS} minimum. "
                f"Need longer script for 30-45s video.",
                chinese_chars
            )
        return True, "", chinese_chars
    else:
        # Check English word count
        words = text.split()
        word_count = len(words)
        if word_count < VOICEOVER_MIN_EN_WORDS:
            return (
                False,
                f"Voiceover too short: {word_count} words < {VOICEOVER_MIN_EN_WORDS} minimum. "
                f"Need longer script for 30-45s video.",
                word_count
            )
        return True, "", word_count


def check_audio_duration(audio_path: Path) -> tuple[bool, str, float]:
    """
    Check if voiceover.mp3 meets minimum duration requirements.
    
    Args:
        audio_path: Path to voiceover.mp3
    
    Returns:
        Tuple of (is_valid, reason, actual_duration)
    """
    if not audio_path.exists():
        return False, "voiceover.mp3 does not exist", 0.0
    
    # Import here to avoid circular imports
    try:
        from pipeline.render_basic_short import probe_audio_file
        audio_info = probe_audio_file(audio_path)
    except ImportError:
        # Fallback to tts module
        from pipeline.tts import get_audio_duration
        duration = get_audio_duration(audio_path)
        audio_info = type('obj', (object,), {'duration': duration, 'error': None})()
    
    if audio_info.error:
        return False, f"Cannot probe audio: {audio_info.error}", 0.0
    
    duration = audio_info.duration
    
    if duration < VOICEOVER_MIN_DURATION_SECONDS:
        return (
            False,
            f"Audio too short: {duration:.1f}s < {VOICEOVER_MIN_DURATION_SECONDS}s minimum. "
            f"TTS output is too brief. Consider lengthening voiceover.txt.",
            duration
        )
    
    return True, "", duration

# Typer CLI app
app = typer.Typer(
    name="prepare-shorts",
    help="Prepare shorts folders for video rendering",
    add_completion=False,
)


# ============================================================================
# Queue Loading
# ============================================================================


def load_queue_csv(queue_date: str, queue_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """
    Load queue from CSV file.
    
    Args:
        queue_date: Date string for queue file
        queue_dir: Directory containing queue files
    
    Returns:
        List of queue row dictionaries
    """
    if queue_dir is None:
        queue_dir = Path("outputs/shorts_queue")
    
    csv_path = queue_dir / f"queue_{queue_date}.csv"
    
    if not csv_path.exists():
        logger.error(f"Queue file not found: {csv_path}")
        return []
    
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    logger.info(f"Loaded {len(rows)} items from queue: {csv_path}")
    return rows


def load_packet_for_claim(claim_id: int, packets_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """
    Load packet JSON for a claim.
    
    Args:
        claim_id: Claim ID
        packets_dir: Directory containing packet JSON files
    
    Returns:
        Packet dictionary or None
    """
    if packets_dir is None:
        packets_dir = get_output_path()
    
    json_path = packets_dir / f"{claim_id}.json"
    
    if not json_path.exists():
        logger.warning(f"Packet file not found: {json_path}")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load packet {json_path}: {e}")
        return None


# ============================================================================
# SRT Validation and Repair
# ============================================================================


def parse_srt_timestamp(ts: str) -> float:
    """Parse SRT timestamp to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {ts}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def validate_srt(srt_path: Path) -> tuple[bool, list[str]]:
    """
    Validate SRT file format.
    
    Checks:
    - File is readable
    - Timestamps are valid
    - Timestamps are monotonically increasing
    - Start < End for each segment
    
    Args:
        srt_path: Path to SRT file
    
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    if not srt_path.exists():
        return False, ["File does not exist"]
    
    try:
        content = srt_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, [f"Cannot read file: {e}"]
    
    if not content.strip():
        return False, ["File is empty"]
    
    # Parse entries
    import re
    pattern = r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(.*?)(?=\n\n|\n*$)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        return False, ["No valid SRT entries found"]
    
    prev_end = 0.0
    
    for match in matches:
        try:
            start = parse_srt_timestamp(match.group(2))
            end = parse_srt_timestamp(match.group(3))
            
            # Check start < end
            if start >= end:
                issues.append(f"Entry {match.group(1)}: start ({start:.2f}s) >= end ({end:.2f}s)")
            
            # Check monotonically increasing (allow small overlap)
            if start < prev_end - 0.1:
                issues.append(f"Entry {match.group(1)}: start ({start:.2f}s) < previous end ({prev_end:.2f}s)")
            
            prev_end = end
            
        except ValueError as e:
            issues.append(f"Entry {match.group(1)}: {e}")
    
    return len(issues) == 0, issues


def repair_srt(srt_path: Path) -> bool:
    """
    Repair SRT file to ensure valid timestamps.
    
    Fixes:
    - Ensures start < end
    - Ensures monotonically increasing timestamps
    
    Args:
        srt_path: Path to SRT file
    
    Returns:
        True if repairs were made
    """
    if not srt_path.exists():
        return False
    
    content = srt_path.read_text(encoding="utf-8")
    
    import re
    pattern = r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(.*?)(?=\n\n|\n*$)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        return False
    
    entries = []
    prev_end = 0.0
    repaired = False
    
    for match in matches:
        try:
            index = int(match.group(1))
            start = parse_srt_timestamp(match.group(2))
            end = parse_srt_timestamp(match.group(3))
            text = match.group(4).strip()
            
            # Fix: ensure start >= previous end
            if start < prev_end:
                start = prev_end + 0.001
                repaired = True
            
            # Fix: ensure end > start
            if end <= start:
                end = start + 1.0
                repaired = True
            
            entries.append({
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            })
            
            prev_end = end
            
        except ValueError:
            continue
    
    if repaired:
        lines = []
        for entry in entries:
            lines.append(str(entry["index"]))
            lines.append(f"{format_srt_timestamp(entry['start'])} --> {format_srt_timestamp(entry['end'])}")
            lines.append(entry["text"])
            lines.append("")
        
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Repaired SRT: {srt_path}")
    
    return repaired


# ============================================================================
# Preparation Steps
# ============================================================================


async def ensure_shorts_folder(
    claim_id: int,
    packet: dict[str, Any],
    shorts_dir: Path,
) -> ShortsStatus:
    """
    Ensure shorts folder exists with all required files.
    
    Args:
        claim_id: Claim ID
        packet: Packet dictionary
        shorts_dir: Base shorts directory
    
    Returns:
        New status after export
    """
    from pipeline.export_shorts_pack import export_shorts_pack
    
    folder_path = shorts_dir / str(claim_id)
    
    # Check current status
    status = determine_shorts_status(folder_path)
    
    if status != ShortsStatus.NEEDS_EXPORT:
        return status
    
    # Ensure packet has claim_id
    if isinstance(packet, dict):
        packet["claim_id"] = claim_id
    
    # Export the folder (sync function)
    try:
        result = export_shorts_pack(
            packet=packet,
            output_dir=shorts_dir,
            overwrite=True,  # Overwrite to complete partial exports
        )
        
        if result:
            logger.info(f"Exported shorts folder for claim {claim_id}")
            return determine_shorts_status(folder_path)
        else:
            logger.error(f"Failed to export shorts folder for claim {claim_id}")
            return ShortsStatus.NEEDS_EXPORT
            
    except Exception as e:
        logger.error(f"Error exporting shorts for claim {claim_id}: {e}")
        return ShortsStatus.NEEDS_EXPORT


async def ensure_voiceover(
    claim_id: int,
    shorts_dir: Path,
    provider: str = "edge",
) -> tuple[ShortsStatus, str]:
    """
    Ensure voiceover.mp3 exists with valid duration.
    
    Performs validation:
    1. Check voiceover.txt length before TTS
    2. Generate TTS
    3. Check audio duration after TTS
    
    Args:
        claim_id: Claim ID
        shorts_dir: Base shorts directory
        provider: TTS provider ("edge" or "http")
    
    Returns:
        Tuple of (new_status, reason_if_failed)
    """
    from pipeline.tts import generate_voiceover_for_folder
    
    folder_path = shorts_dir / str(claim_id)
    voiceover_txt_path = folder_path / "voiceover.txt"
    voiceover_mp3_path = folder_path / "voiceover.mp3"
    
    # Check current status
    status = determine_shorts_status(folder_path)
    
    if status not in (ShortsStatus.NEEDS_TTS, ShortsStatus.TOO_SHORT_VOICEOVER, 
                      ShortsStatus.TOO_SHORT_AUDIO, ShortsStatus.TTS_FAILED):
        return status, ""
    
    # ========== 1. CHECK VOICEOVER TEXT LENGTH ==========
    is_valid, reason, length = check_voiceover_text_length(voiceover_txt_path)
    if not is_valid:
        logger.warning(f"Claim {claim_id}: {reason}")
        return ShortsStatus.TOO_SHORT_VOICEOVER, reason
    
    logger.info(f"Claim {claim_id}: voiceover.txt length OK ({length})")
    
    # ========== 2. GENERATE TTS ==========
    try:
        result = await generate_voiceover_for_folder(
            folder_path=folder_path,
            adjust_captions=True,
            provider=provider,
        )
        
        if not result or not result.exists():
            logger.error(f"TTS failed for claim {claim_id}")
            return ShortsStatus.TTS_FAILED, "TTS provider returned no output"
            
    except ImportError as e:
        logger.error(f"TTS not available: {e}")
        return ShortsStatus.TTS_FAILED, f"TTS import error: {e}. Install with: pip install edge-tts"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"TTS error for claim {claim_id}: {error_msg}")
        
        # Provide helpful suggestions for common errors
        if "SSL" in error_msg or "certificate" in error_msg.lower():
            error_msg += " | Fix: Set MYTH_MUSEUM_SKIP_SSL=1 or use --provider http"
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            error_msg += " | Fix: Check internet connection or use --provider http"
        
        return ShortsStatus.TTS_FAILED, error_msg
    
    # ========== 3. CHECK AUDIO DURATION ==========
    is_valid, reason, duration = check_audio_duration(voiceover_mp3_path)
    if not is_valid:
        logger.warning(f"Claim {claim_id}: {reason}")
        # Delete the too-short audio to allow re-generation
        try:
            voiceover_mp3_path.unlink()
        except Exception:
            pass
        return ShortsStatus.TOO_SHORT_AUDIO, reason
    
    logger.info(f"Claim {claim_id}: voiceover.mp3 duration OK ({duration:.1f}s)")
    
    return determine_shorts_status(folder_path), ""


def ensure_valid_srt(
    claim_id: int,
    shorts_dir: Path,
) -> bool:
    """
    Ensure captions.srt is valid and synced.
    
    Args:
        claim_id: Claim ID
        shorts_dir: Base shorts directory
    
    Returns:
        True if SRT is valid (or was repaired)
    """
    from pipeline.tts import get_audio_duration, adjust_srt_timing
    
    folder_path = shorts_dir / str(claim_id)
    srt_path = folder_path / "captions.srt"
    mp3_path = folder_path / "voiceover.mp3"
    
    if not srt_path.exists():
        logger.warning(f"No captions.srt for claim {claim_id}")
        return False
    
    # Validate and repair if needed
    is_valid, issues = validate_srt(srt_path)
    
    if not is_valid:
        logger.warning(f"SRT issues for claim {claim_id}: {issues}")
        repair_srt(srt_path)
    
    # Sync with audio if mp3 exists
    if mp3_path.exists():
        try:
            audio_duration = get_audio_duration(mp3_path)
            adjust_srt_timing(
                srt_path,
                audio_duration,
                threshold=SRT_TIMING_ADJUSTMENT_THRESHOLD,
            )
        except Exception as e:
            logger.warning(f"Could not sync SRT with audio for claim {claim_id}: {e}")
    
    return True


# ============================================================================
# Main Preparation Function
# ============================================================================


async def prepare_single(
    claim_id: int,
    packet: Optional[dict[str, Any]],
    shorts_dir: Path,
    packets_dir: Path,
    provider: str = "edge",
) -> dict[str, Any]:
    """
    Prepare a single shorts folder with validation gates.
    
    Validation gates that block rendering:
    - too_short_voiceover: voiceover.txt below minimum length
    - too_short_audio: voiceover.mp3 duration < 20s
    - tts_failed: TTS provider error
    
    Args:
        claim_id: Claim ID
        packet: Packet dictionary (optional, will load if None)
        shorts_dir: Base shorts directory
        packets_dir: Directory containing packet JSON files
        provider: TTS provider ("edge" or "http")
    
    Returns:
        Dict with status changes, actions taken, and failure reasons
    """
    result = {
        "claim_id": claim_id,
        "initial_status": None,
        "final_status": None,
        "actions": [],
        "errors": [],
        "reason": "",  # Detailed reason for failures
    }
    
    folder_path = shorts_dir / str(claim_id)
    
    # Get initial status
    initial_status = determine_shorts_status(folder_path)
    result["initial_status"] = initial_status.value
    
    # Skip if already rendered
    if initial_status == ShortsStatus.RENDERED:
        result["final_status"] = initial_status.value
        return result
    
    # Load packet if needed
    if packet is None:
        packet = load_packet_for_claim(claim_id, packets_dir)
    
    if packet is None and initial_status == ShortsStatus.NEEDS_EXPORT:
        result["errors"].append("Cannot load packet for export")
        result["reason"] = "Packet JSON file not found"
        result["final_status"] = initial_status.value
        return result
    
    # Step 1: Ensure folder exists with all files
    if initial_status == ShortsStatus.NEEDS_EXPORT:
        result["actions"].append("export_folder")
        new_status = await ensure_shorts_folder(claim_id, packet, shorts_dir)
        
        if new_status == ShortsStatus.NEEDS_EXPORT:
            result["errors"].append("Export failed")
            result["reason"] = "Failed to export shorts folder"
            result["final_status"] = new_status.value
            return result
        
        initial_status = new_status
    
    # Step 2: Ensure voiceover exists (with validation gates)
    if initial_status in (ShortsStatus.NEEDS_TTS, ShortsStatus.TOO_SHORT_VOICEOVER,
                          ShortsStatus.TOO_SHORT_AUDIO, ShortsStatus.TTS_FAILED):
        result["actions"].append("generate_tts")
        new_status, reason = await ensure_voiceover(claim_id, shorts_dir, provider)
        
        # Check for validation failures (these block rendering)
        if new_status == ShortsStatus.TOO_SHORT_VOICEOVER:
            result["errors"].append("voiceover.txt too short")
            result["reason"] = reason
            result["final_status"] = new_status.value
            logger.warning(f"Claim {claim_id} blocked: {reason}")
            return result
        
        if new_status == ShortsStatus.TOO_SHORT_AUDIO:
            result["errors"].append("voiceover.mp3 too short")
            result["reason"] = reason
            result["final_status"] = new_status.value
            logger.warning(f"Claim {claim_id} blocked: {reason}")
            return result
        
        if new_status == ShortsStatus.TTS_FAILED:
            result["errors"].append("TTS failed")
            result["reason"] = reason
            result["final_status"] = new_status.value
            logger.warning(f"Claim {claim_id} blocked: {reason}")
            return result
        
        if new_status == ShortsStatus.NEEDS_TTS:
            result["errors"].append("TTS generation failed")
            result["reason"] = reason or "Unknown TTS error"
            result["final_status"] = new_status.value
            return result
        
        initial_status = new_status
    
    # Step 3: Validate and repair SRT
    if initial_status in (ShortsStatus.READY, ShortsStatus.RENDERED):
        if ensure_valid_srt(claim_id, shorts_dir):
            result["actions"].append("validate_srt")
    
    # Get final status
    final_status = determine_shorts_status(folder_path)
    result["final_status"] = final_status.value
    
    return result


async def prepare_queue(
    queue_date: str,
    limit: int = 100,
    queue_dir: Optional[Path] = None,
    shorts_dir: Optional[Path] = None,
    packets_dir: Optional[Path] = None,
    provider: str = "edge",
) -> dict[str, Any]:
    """
    Prepare all items in a queue with validation gates.
    
    Items that fail validation (too_short_voiceover, too_short_audio, tts_failed)
    will be marked with their status and not proceed to rendering.
    
    Args:
        queue_date: Date string for queue file
        limit: Maximum items to process
        queue_dir: Directory containing queue files
        shorts_dir: Base shorts directory
        packets_dir: Directory containing packet JSON files
        provider: TTS provider ("edge" or "http")
    
    Returns:
        Dict with stats, results, and blocked items
    """
    if queue_dir is None:
        queue_dir = Path("outputs/shorts_queue")
    
    if shorts_dir is None:
        shorts_dir = Path("outputs/shorts")
    
    if packets_dir is None:
        packets_dir = get_output_path()
    
    # Load queue
    queue_items = load_queue_csv(queue_date, queue_dir)
    
    if not queue_items:
        return {
            "total": 0,
            "processed": 0,
            "ready": 0,
            "blocked": 0,
            "errors": 0,
            "results": [],
        }
    
    # Process items
    results = []
    processed = 0
    ready_count = 0
    blocked_count = 0
    errors = 0
    
    for item in queue_items[:limit]:
        claim_id = int(item.get("claim_id", 0))
        
        if claim_id == 0:
            continue
        
        console.print(f"  Processing claim {claim_id}...", end=" ")
        
        result = await prepare_single(
            claim_id=claim_id,
            packet=None,
            shorts_dir=shorts_dir,
            packets_dir=packets_dir,
            provider=provider,
        )
        
        results.append(result)
        processed += 1
        
        final_status = result.get("final_status", "")
        
        if final_status == "ready":
            ready_count += 1
            console.print("[green]ready[/green]")
        elif final_status in ("too_short_voiceover", "too_short_audio", "tts_failed"):
            blocked_count += 1
            console.print(f"[yellow]{final_status}[/yellow]")
        elif result["errors"]:
            errors += 1
            console.print(f"[red]error[/red]")
        else:
            console.print(f"[dim]{final_status}[/dim]")
    
    # Update queue CSV with new statuses
    update_queue_csv(queue_date, results, queue_dir, shorts_dir)
    
    # Generate report with detailed reasons
    report_path = generate_prepare_report(queue_date, results, queue_dir)
    
    return {
        "total": len(queue_items),
        "processed": processed,
        "ready": ready_count,
        "blocked": blocked_count,
        "errors": errors,
        "results": results,
        "report_path": report_path,
    }


def update_queue_csv(
    queue_date: str,
    results: list[dict[str, Any]],
    queue_dir: Path,
    shorts_dir: Path,
) -> Path:
    """
    Update queue CSV with new statuses.
    
    Args:
        queue_date: Date string for queue file
        results: List of prepare results
        queue_dir: Directory containing queue files
        shorts_dir: Base shorts directory
    
    Returns:
        Path to updated CSV
    """
    csv_path = queue_dir / f"queue_{queue_date}.csv"
    
    if not csv_path.exists():
        return csv_path
    
    # Read current queue
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Build status map from results
    status_map = {r["claim_id"]: r["final_status"] for r in results}
    
    # Update statuses
    for row in rows:
        claim_id = int(row.get("claim_id", 0))
        if claim_id in status_map:
            row["status"] = status_map[claim_id]
        else:
            # Re-check status for items not in results
            folder_path = shorts_dir / str(claim_id)
            row["status"] = determine_shorts_status(folder_path).value
    
    # Write updated CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Updated queue CSV: {csv_path}")
    return csv_path


def generate_prepare_report(
    queue_date: str,
    results: list[dict[str, Any]],
    queue_dir: Path,
) -> Path:
    """
    Generate a prepare report in Markdown with detailed failure reasons.
    
    Args:
        queue_date: Date string
        results: List of prepare results
        queue_dir: Directory for report output
    
    Returns:
        Path to report file
    """
    report_path = queue_dir / f"prepare_report_{queue_date}.md"
    
    # Calculate stats
    status_counts = {
        "ready": 0,
        "rendered": 0,
        "too_short_voiceover": 0,
        "too_short_audio": 0,
        "tts_failed": 0,
        "needs_export": 0,
        "needs_tts": 0,
        "other_errors": 0,
    }
    
    blocked_items = []
    
    for result in results:
        final = result.get("final_status", "")
        
        if final in status_counts:
            status_counts[final] += 1
        elif result.get("errors"):
            status_counts["other_errors"] += 1
        
        # Track blocked items
        if final in ("too_short_voiceover", "too_short_audio", "tts_failed"):
            blocked_items.append({
                "claim_id": result.get("claim_id"),
                "status": final,
                "reason": result.get("reason", "Unknown"),
            })
    
    lines = [
        f"# Prepare Report: {queue_date}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Total Processed:** {len(results)}",
        "",
        "## Summary",
        "",
        f"- **Ready for rendering:** {status_counts['ready']}",
        f"- **Already rendered:** {status_counts['rendered']}",
        f"- **Blocked (validation failed):** {len(blocked_items)}",
        "",
    ]
    
    # Blocked items section
    if blocked_items:
        lines.extend([
            "## Blocked Items (Cannot Render)",
            "",
            "These items failed validation and will not proceed to rendering.",
            "Fix the issues and re-run prepare to unblock them.",
            "",
            "| Claim ID | Status | Reason |",
            "|----------|--------|--------|",
        ])
        
        for item in blocked_items:
            # Truncate reason for table
            reason = item["reason"][:80] + "..." if len(item.get("reason", "")) > 80 else item.get("reason", "")
            lines.append(f"| {item['claim_id']} | {item['status']} | {reason} |")
        
        lines.extend([
            "",
            "### How to Fix",
            "",
            "**too_short_voiceover:**",
            "- The voiceover.txt is below minimum length (Chinese: 120 chars, English: 80 words)",
            "- Solution: Edit voiceover.txt to add more content (hook + clarification + example + CTA)",
            "- Target length: 30-45 seconds of speech",
            "",
            "**too_short_audio:**",
            "- The generated voiceover.mp3 is under 20 seconds",
            "- Solution: Lengthen voiceover.txt and regenerate TTS",
            "",
            "**tts_failed:**",
            "- TTS provider error (network, SSL, or API issue)",
            "- Solutions:",
            "  - Check internet connection",
            "  - For SSL errors: Set `MYTH_MUSEUM_SKIP_SSL=1`",
            "  - Use HTTP provider: `--provider http` with `OPENAI_API_KEY` set",
            "",
        ])
    
    # Status breakdown
    lines.extend([
        "## Status Breakdown",
        "",
    ])
    
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"- **{status}:** {count}")
    
    # Details table
    lines.extend([
        "",
        "## All Items",
        "",
        "| Claim ID | Initial | Final | Actions | Issues |",
        "|----------|---------|-------|---------|--------|",
    ])
    
    for result in results:
        claim_id = result.get("claim_id", "?")
        initial = result.get("initial_status", "?")
        final = result.get("final_status", "?")
        actions = ", ".join(result.get("actions", [])) or "-"
        errors_list = result.get("errors", [])
        errors_str = ", ".join(errors_list) if errors_list else "-"
        
        lines.append(f"| {claim_id} | {initial} | {final} | {actions} | {errors_str} |")
    
    lines.extend([
        "",
        "---",
        "*Prepare report generated by myth-museum*",
    ])
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated prepare report: {report_path}")
    
    return report_path


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def prepare(
    queue_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Queue date (default: today, format: YYYY-MM-DD)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum items to process",
    ),
    queue_dir: Optional[str] = typer.Option(
        None,
        "--queue-dir",
        help="Queue directory",
    ),
    shorts_dir: Optional[str] = typer.Option(
        None,
        "--shorts-dir",
        help="Shorts output directory",
    ),
    provider: str = typer.Option(
        "edge",
        "--provider",
        "-p",
        help="TTS provider: 'edge' (default, free) or 'http' (OpenAI-compatible)",
    ),
) -> None:
    """
    Prepare shorts folders for rendering with validation gates.
    
    Reads queue CSV and auto-completes missing steps for each item.
    Items that fail validation (too short voiceover/audio, TTS errors)
    will be blocked from rendering with detailed error messages.
    """
    import asyncio
    
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    q_dir = Path(queue_dir) if queue_dir else Path("outputs/shorts_queue")
    s_dir = Path(shorts_dir) if shorts_dir else Path("outputs/shorts")
    
    console.print(f"[bold cyan]=== Prepare Shorts ===[/bold cyan]\n")
    console.print(f"Queue date: {queue_date}")
    console.print(f"Limit: {limit}")
    console.print(f"TTS provider: {provider}")
    console.print("")
    
    # Check queue exists
    csv_path = q_dir / f"queue_{queue_date}.csv"
    if not csv_path.exists():
        console.print(f"[red]Queue not found: {csv_path}[/red]")
        raise typer.Exit(1)
    
    # Run prepare
    try:
        result = asyncio.run(prepare_queue(
            queue_date=queue_date,
            limit=limit,
            queue_dir=q_dir,
            shorts_dir=s_dir,
            provider=provider,
        ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Print results
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Total in queue: {result['total']}")
    console.print(f"  Processed: {result['processed']}")
    console.print(f"  [green]Ready for render: {result.get('ready', 0)}[/green]")
    
    blocked = result.get('blocked', 0)
    if blocked > 0:
        console.print(f"  [yellow]Blocked (validation failed): {blocked}[/yellow]")
    
    if result.get('errors', 0) > 0:
        console.print(f"  [red]Errors: {result['errors']}[/red]")
    
    # Show status breakdown
    status_counts = {}
    for r in result.get("results", []):
        final = r.get("final_status", "unknown")
        status_counts[final] = status_counts.get(final, 0) + 1
    
    if status_counts:
        console.print(f"\n[bold]Final Statuses:[/bold]")
        for status, count in sorted(status_counts.items()):
            color = "green" if status == "ready" else "yellow" if "too_short" in status or "failed" in status else "dim"
            console.print(f"  [{color}]{status}: {count}[/{color}]")
    
    if result.get("report_path"):
        console.print(f"\n[bold]Report:[/bold] {result['report_path']}")
    
    console.print(f"\n[green]OK[/green] Prepare complete")


@app.command()
def status(
    queue_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Queue date (default: today)",
    ),
    queue_dir: Optional[str] = typer.Option(
        None,
        "--dir",
        help="Queue directory",
    ),
) -> None:
    """
    Show current status of queue items.
    """
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    q_dir = Path(queue_dir) if queue_dir else Path("outputs/shorts_queue")
    s_dir = Path("outputs/shorts")
    
    # Load queue
    queue_items = load_queue_csv(queue_date, q_dir)
    
    if not queue_items:
        console.print(f"[yellow]No queue found for {queue_date}[/yellow]")
        return
    
    console.print(f"[bold cyan]=== Queue Status: {queue_date} ===[/bold cyan]\n")
    
    # Get current statuses
    status_counts = {}
    
    table = Table()
    table.add_column("ID", justify="right")
    table.add_column("CSV Status")
    table.add_column("Actual Status")
    table.add_column("Topic")
    
    for item in queue_items:
        claim_id = int(item.get("claim_id", 0))
        csv_status = item.get("status", "?")
        
        folder_path = s_dir / str(claim_id)
        actual_status = determine_shorts_status(folder_path).value
        
        status_counts[actual_status] = status_counts.get(actual_status, 0) + 1
        
        # Highlight if status changed
        if csv_status != actual_status:
            actual_display = f"[yellow]{actual_status}[/yellow]"
        else:
            actual_display = actual_status
        
        table.add_row(
            str(claim_id),
            csv_status,
            actual_display,
            item.get("topic", "?"),
        )
    
    console.print(table)
    
    console.print(f"\n[bold]Status Counts:[/bold]")
    for status, count in sorted(status_counts.items()):
        console.print(f"  - {status}: {count}")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
