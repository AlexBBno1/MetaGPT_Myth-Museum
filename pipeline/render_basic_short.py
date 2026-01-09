"""
Myth Museum - Basic Shorts Video Renderer

Render ready shorts folders to final.mp4 using ffmpeg.
Combines voiceover audio with burned-in subtitles on a solid background.

Audio is explicitly mapped and normalized to ensure audible output.
"""

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from core.constants import (
    RENDER_AUDIO_CODEC,
    RENDER_DEFAULT_BG_COLOR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_SUBTITLE_STYLE,
    RENDER_VIDEO_CODEC,
    RENDER_WIDTH,
    ShortsStatus,
    VOICEOVER_MIN_DURATION_SECONDS,
    determine_shorts_status,
)
from core.logging import get_logger
from pipeline.prepare_shorts import (
    load_queue_csv,
    repair_srt,
    validate_srt,
)

logger = get_logger(__name__)
console = Console()

# Typer CLI app
app = typer.Typer(
    name="render-basic-short",
    help="Render shorts folders to final.mp4 video files",
    add_completion=False,
)


# ============================================================================
# FFmpeg Utilities and Validation
# ============================================================================


class FFmpegNotFoundError(Exception):
    """Raised when ffmpeg is not available."""
    pass


class RenderValidationError(Exception):
    """Raised when pre/post render validation fails."""
    pass


@dataclass
class AudioInfo:
    """Audio file information from ffprobe."""
    exists: bool
    duration: float
    has_audio_stream: bool
    codec: str
    error: Optional[str] = None


@dataclass
class VideoInfo:
    """Video file information from ffprobe."""
    exists: bool
    duration: float
    has_video_stream: bool
    has_audio_stream: bool
    video_codec: str
    audio_codec: str
    error: Optional[str] = None


def check_ffmpeg() -> str:
    """
    Check if ffmpeg is available.
    
    Returns:
        Version string if available
    
    Raises:
        FFmpegNotFoundError: If ffmpeg is not found
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            return first_line
        
        raise FFmpegNotFoundError("ffmpeg returned error")
        
    except FileNotFoundError:
        raise FFmpegNotFoundError(
            "ffmpeg not found. Please install ffmpeg and add it to PATH.\n\n"
            "Windows installation:\n"
            "  1. Download from https://www.ffmpeg.org/download.html\n"
            "  2. Extract to C:\\ffmpeg\n"
            "  3. Add C:\\ffmpeg\\bin to system PATH\n"
            "  4. Restart terminal/IDE\n\n"
            "Or use: winget install ffmpeg\n"
            "Or use: choco install ffmpeg"
        )
    except subprocess.TimeoutExpired:
        raise FFmpegNotFoundError("ffmpeg timeout - check not responding")


def probe_audio_file(audio_path: Path) -> AudioInfo:
    """
    Get detailed audio file information using ffprobe.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        AudioInfo with duration, codec, and stream info
    """
    if not audio_path.exists():
        return AudioInfo(
            exists=False,
            duration=0.0,
            has_audio_stream=False,
            codec="",
            error="File does not exist"
        )
    
    try:
        # Get format info (duration)
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return AudioInfo(
                exists=True,
                duration=0.0,
                has_audio_stream=False,
                codec="",
                error=f"ffprobe failed: {result.stderr}"
            )
        
        data = json.loads(result.stdout)
        
        # Extract duration
        duration = float(data.get("format", {}).get("duration", 0))
        
        # Check for audio stream
        streams = data.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        has_audio = len(audio_streams) > 0
        codec = audio_streams[0].get("codec_name", "") if audio_streams else ""
        
        return AudioInfo(
            exists=True,
            duration=duration,
            has_audio_stream=has_audio,
            codec=codec,
            error=None
        )
        
    except subprocess.TimeoutExpired:
        return AudioInfo(
            exists=True,
            duration=0.0,
            has_audio_stream=False,
            codec="",
            error="ffprobe timeout"
        )
    except (json.JSONDecodeError, ValueError) as e:
        return AudioInfo(
            exists=True,
            duration=0.0,
            has_audio_stream=False,
            codec="",
            error=f"Parse error: {e}"
        )
    except FileNotFoundError:
        return AudioInfo(
            exists=True,
            duration=0.0,
            has_audio_stream=False,
            codec="",
            error="ffprobe not found"
        )


def probe_video_file(video_path: Path) -> VideoInfo:
    """
    Get detailed video file information using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        VideoInfo with duration, codecs, and stream info
    """
    if not video_path.exists():
        return VideoInfo(
            exists=False,
            duration=0.0,
            has_video_stream=False,
            has_audio_stream=False,
            video_codec="",
            audio_codec="",
            error="File does not exist"
        )
    
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return VideoInfo(
                exists=True,
                duration=0.0,
                has_video_stream=False,
                has_audio_stream=False,
                video_codec="",
                audio_codec="",
                error=f"ffprobe failed: {result.stderr}"
            )
        
        data = json.loads(result.stdout)
        
        # Extract duration
        duration = float(data.get("format", {}).get("duration", 0))
        
        # Check streams
        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        
        has_video = len(video_streams) > 0
        has_audio = len(audio_streams) > 0
        video_codec = video_streams[0].get("codec_name", "") if video_streams else ""
        audio_codec = audio_streams[0].get("codec_name", "") if audio_streams else ""
        
        return VideoInfo(
            exists=True,
            duration=duration,
            has_video_stream=has_video,
            has_audio_stream=has_audio,
            video_codec=video_codec,
            audio_codec=audio_codec,
            error=None
        )
        
    except subprocess.TimeoutExpired:
        return VideoInfo(
            exists=True,
            duration=0.0,
            has_video_stream=False,
            has_audio_stream=False,
            video_codec="",
            audio_codec="",
            error="ffprobe timeout"
        )
    except (json.JSONDecodeError, ValueError) as e:
        return VideoInfo(
            exists=True,
            duration=0.0,
            has_video_stream=False,
            has_audio_stream=False,
            video_codec="",
            audio_codec="",
            error=f"Parse error: {e}"
        )
    except FileNotFoundError:
        return VideoInfo(
            exists=True,
            duration=0.0,
            has_video_stream=False,
            has_audio_stream=False,
            video_codec="",
            audio_codec="",
            error="ffprobe not found"
        )


def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration using ffprobe.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    info = probe_audio_file(audio_path)
    if info.duration > 0:
        return info.duration
    
    # Fallback: estimate from file size (assumes ~128kbps)
    if audio_path.exists():
        return audio_path.stat().st_size / 16000
    
    return 35.0  # Default estimate


def escape_ffmpeg_path(path: Path) -> str:
    """
    Escape path for use in ffmpeg filters (Windows compatible).
    
    Args:
        path: Path to escape
    
    Returns:
        Escaped path string
    """
    path_str = str(path.absolute()).replace("\\", "/")
    path_str = path_str.replace(":", "\\:")
    return path_str


def copy_srt_to_temp(srt_path: Path) -> Path:
    """
    Copy SRT file to temp directory with simple filename.
    
    Args:
        srt_path: Original SRT path
    
    Returns:
        Path to temp SRT file
    """
    temp_dir = Path(tempfile.gettempdir()) / "myth_museum_render"
    temp_dir.mkdir(exist_ok=True)
    
    temp_srt = temp_dir / f"subs_{hash(str(srt_path)) % 100000}.srt"
    shutil.copy2(srt_path, temp_srt)
    
    return temp_srt


def build_ffmpeg_command(
    audio_path: Path,
    srt_path: Path,
    output_path: Path,
    duration: float,
    background_path: Optional[Path] = None,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
    fps: int = RENDER_FPS,
    audio_gain_db: float = 6.0,
    use_loudnorm: bool = True,
) -> list[str]:
    """
    Build ffmpeg command for video rendering with explicit audio mapping.
    
    IMPORTANT: Uses explicit -map to ensure audio is included in output.
    
    Args:
        audio_path: Path to voiceover.mp3
        srt_path: Path to captions.srt (should be temp copy)
        output_path: Path to output final.mp4
        duration: Audio duration in seconds
        background_path: Optional background image
        width: Video width (default: 1080)
        height: Video height (default: 1920)
        fps: Frame rate (default: 30)
        audio_gain_db: Audio gain in dB (default: 6.0)
        use_loudnorm: Whether to use loudnorm filter (default: True)
    
    Returns:
        List of command arguments
    """
    cmd = ["ffmpeg", "-y"]  # -y to overwrite
    
    # Input 0: Video source (color background or image)
    if background_path and background_path.exists():
        cmd.extend([
            "-loop", "1",
            "-i", str(background_path),
        ])
    else:
        # Generate solid color background with explicit duration
        cmd.extend([
            "-f", "lavfi",
            "-i", f"color=c={RENDER_DEFAULT_BG_COLOR}:s={width}x{height}:d={duration + 1}:r={fps}",
        ])
    
    # Input 1: Audio source (voiceover.mp3)
    cmd.extend(["-i", str(audio_path)])
    
    # EXPLICIT STREAM MAPPING - Critical for audio inclusion
    # Map video from input 0, audio from input 1
    cmd.extend([
        "-map", "0:v:0",  # Video from first input
        "-map", "1:a:0",  # Audio from second input (voiceover)
    ])
    
    # Build video filter for subtitles
    escaped_srt = escape_ffmpeg_path(srt_path)
    vf = f"subtitles='{escaped_srt}':force_style='{RENDER_SUBTITLE_STYLE}'"
    
    # If using background image, add scale filter
    if background_path and background_path.exists():
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2," + vf
    
    cmd.extend(["-vf", vf])
    
    # AUDIO FILTER - Normalize volume for audibility
    if use_loudnorm:
        # EBU R128 loudness normalization
        # I=-16: target integrated loudness (YouTube standard)
        # TP=-1.5: true peak limit
        # LRA=11: loudness range
        af = "loudnorm=I=-16:TP=-1.5:LRA=11"
    else:
        # Simple volume boost
        af = f"volume={audio_gain_db}dB"
    
    cmd.extend(["-af", af])
    
    # Video codec settings
    cmd.extend([
        "-c:v", RENDER_VIDEO_CODEC,
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ])
    
    # Audio codec settings
    cmd.extend([
        "-c:a", RENDER_AUDIO_CODEC,
        "-b:a", "192k",
    ])
    
    # Use shortest stream (audio) as reference for duration
    cmd.extend(["-shortest"])
    
    # Output
    cmd.append(str(output_path))
    
    return cmd


# ============================================================================
# Pre/Post Render Validation
# ============================================================================


def validate_before_render(
    folder_path: Path,
    min_duration: float = VOICEOVER_MIN_DURATION_SECONDS,
) -> tuple[bool, str, AudioInfo]:
    """
    Validate shorts folder before rendering.
    
    Checks:
    - voiceover.mp3 exists
    - voiceover.mp3 duration >= min_duration
    - voiceover.mp3 has valid audio stream
    
    Args:
        folder_path: Path to shorts folder
        min_duration: Minimum audio duration in seconds
    
    Returns:
        Tuple of (is_valid, error_message, audio_info)
    """
    audio_path = folder_path / "voiceover.mp3"
    
    if not audio_path.exists():
        return False, "voiceover.mp3 does not exist", AudioInfo(
            exists=False, duration=0, has_audio_stream=False, codec="", error="Missing"
        )
    
    # Probe audio file
    audio_info = probe_audio_file(audio_path)
    
    if audio_info.error:
        return False, f"Audio probe failed: {audio_info.error}", audio_info
    
    if not audio_info.has_audio_stream:
        return False, "voiceover.mp3 has no audio stream (corrupt file?)", audio_info
    
    if audio_info.duration < min_duration:
        return False, f"Audio too short: {audio_info.duration:.1f}s < {min_duration}s minimum", audio_info
    
    # Log diagnostic info
    logger.info(f"Pre-render check: duration={audio_info.duration:.1f}s, codec={audio_info.codec}")
    
    return True, "", audio_info


def validate_after_render(
    output_path: Path,
) -> tuple[bool, str, VideoInfo]:
    """
    Validate rendered video after ffmpeg completes.
    
    Checks:
    - final.mp4 exists
    - final.mp4 has video stream
    - final.mp4 has audio stream (CRITICAL)
    - final.mp4 duration > 0
    
    Args:
        output_path: Path to final.mp4
    
    Returns:
        Tuple of (is_valid, error_message, video_info)
    """
    if not output_path.exists():
        return False, "Output file was not created", VideoInfo(
            exists=False, duration=0, has_video_stream=False, has_audio_stream=False,
            video_codec="", audio_codec="", error="Missing"
        )
    
    # Probe video file
    video_info = probe_video_file(output_path)
    
    if video_info.error:
        return False, f"Video probe failed: {video_info.error}", video_info
    
    if not video_info.has_video_stream:
        return False, "Output has no video stream", video_info
    
    if not video_info.has_audio_stream:
        return False, "OUTPUT HAS NO AUDIO STREAM - render failed to include audio", video_info
    
    if video_info.duration <= 0:
        return False, "Output has zero duration", video_info
    
    # Log diagnostic info
    logger.info(
        f"Post-render check: duration={video_info.duration:.1f}s, "
        f"video={video_info.video_codec}, audio={video_info.audio_codec}"
    )
    console.print(
        f"  [green]OK[/green] Video: {video_info.duration:.1f}s, "
        f"codecs: {video_info.video_codec}/{video_info.audio_codec}"
    )
    
    return True, "", video_info


# ============================================================================
# Rendering Functions
# ============================================================================


def render_single(
    folder_path: Path,
    background_path: Optional[Path] = None,
    validate_before: bool = True,
    audio_gain_db: float = 6.0,
    use_loudnorm: bool = True,
) -> Optional[Path]:
    """
    Render a single shorts folder to final.mp4.
    
    Args:
        folder_path: Path to shorts folder
        background_path: Optional background image
        validate_before: Whether to validate/repair SRT first
        audio_gain_db: Audio gain in dB when not using loudnorm
        use_loudnorm: Whether to use loudnorm filter
    
    Returns:
        Path to final.mp4 or None if failed
    """
    # Check status
    status = determine_shorts_status(folder_path)
    
    if status == ShortsStatus.RENDERED:
        logger.info(f"Already rendered: {folder_path}")
        return folder_path / "final.mp4"
    
    if status != ShortsStatus.READY:
        logger.warning(f"Not ready to render (status={status.value}): {folder_path}")
        return None
    
    audio_path = folder_path / "voiceover.mp3"
    srt_path = folder_path / "captions.srt"
    output_path = folder_path / "final.mp4"
    
    # ========== PRE-RENDER VALIDATION ==========
    is_valid, error_msg, audio_info = validate_before_render(folder_path)
    
    if not is_valid:
        logger.error(f"Pre-render validation failed: {error_msg}")
        console.print(f"  [red]FAIL[/red] Pre-render check failed: {error_msg}")
        # Write error to status file for debugging
        _write_render_status(folder_path, "render_failed", error_msg)
        return None
    
    console.print(f"  [green]OK[/green] Audio: {audio_info.duration:.1f}s, codec: {audio_info.codec}")
    
    # Validate required files
    if not srt_path.exists():
        logger.error(f"Missing captions.srt: {folder_path}")
        return None
    
    # Validate and repair SRT
    if validate_before:
        is_valid_srt, issues = validate_srt(srt_path)
        if not is_valid_srt:
            logger.warning(f"SRT issues: {issues}")
            repair_srt(srt_path)
    
    # Get audio duration
    duration = audio_info.duration
    
    # Copy SRT to temp (avoid Windows path issues)
    temp_srt = copy_srt_to_temp(srt_path)
    
    try:
        # Build command with explicit audio mapping
        cmd = build_ffmpeg_command(
            audio_path=audio_path,
            srt_path=temp_srt,
            output_path=output_path,
            duration=duration,
            background_path=background_path,
            audio_gain_db=audio_gain_db,
            use_loudnorm=use_loudnorm,
        )
        
        logger.info(f"Running ffmpeg for {folder_path.name}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            _write_render_status(folder_path, "render_failed", f"ffmpeg error: {result.stderr[:200]}")
            return None
        
        # ========== POST-RENDER VALIDATION ==========
        is_valid, error_msg, video_info = validate_after_render(output_path)
        
        if not is_valid:
            logger.error(f"Post-render validation failed: {error_msg}")
            console.print(f"  [red]FAIL[/red] Post-render check failed: {error_msg}")
            # Delete invalid output
            if output_path.exists():
                output_path.unlink()
            _write_render_status(folder_path, "render_failed", error_msg)
            return None
        
        logger.info(f"Rendered: {output_path}")
        _write_render_status(folder_path, "rendered", "Success")
        return output_path
            
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out for {folder_path}")
        _write_render_status(folder_path, "render_failed", "ffmpeg timeout")
        return None
    except Exception as e:
        logger.error(f"Render error: {e}")
        _write_render_status(folder_path, "render_failed", str(e))
        return None
    finally:
        # Clean up temp SRT
        if temp_srt.exists():
            try:
                temp_srt.unlink()
            except Exception:
                pass


def _write_render_status(folder_path: Path, status: str, message: str) -> None:
    """Write render status to a JSON file for debugging."""
    status_file = folder_path / "render_status.json"
    try:
        status_data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        status_file.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
    except Exception:
        pass


def render_from_queue(
    queue_date: str,
    limit: int = 100,
    queue_dir: Optional[Path] = None,
    shorts_dir: Optional[Path] = None,
    background_path: Optional[Path] = None,
    audio_gain_db: float = 6.0,
    use_loudnorm: bool = True,
) -> dict[str, Any]:
    """
    Render all ready items from a queue.
    
    Args:
        queue_date: Date string for queue file
        limit: Maximum items to render
        queue_dir: Directory containing queue files
        shorts_dir: Base shorts directory
        background_path: Optional background image
        audio_gain_db: Audio gain in dB when not using loudnorm
        use_loudnorm: Whether to use loudnorm filter
    
    Returns:
        Dict with stats
    """
    if queue_dir is None:
        queue_dir = Path("outputs/shorts_queue")
    
    if shorts_dir is None:
        shorts_dir = Path("outputs/shorts")
    
    # Load queue
    queue_items = load_queue_csv(queue_date, queue_dir)
    
    if not queue_items:
        return {
            "total": 0,
            "rendered": 0,
            "skipped": 0,
            "failed": 0,
            "paths": [],
        }
    
    rendered = 0
    skipped = 0
    failed = 0
    paths = []
    
    for item in queue_items[:limit]:
        claim_id = int(item.get("claim_id", 0))
        
        if claim_id == 0:
            continue
        
        folder_path = shorts_dir / str(claim_id)
        status = determine_shorts_status(folder_path)
        
        # Only render READY items
        if status == ShortsStatus.RENDERED:
            logger.debug(f"Skipping already rendered: {claim_id}")
            skipped += 1
            continue
        
        if status != ShortsStatus.READY:
            logger.debug(f"Skipping non-ready (status={status.value}): {claim_id}")
            skipped += 1
            continue
        
        console.print(f"\n[bold]Rendering claim {claim_id}...[/bold]")
        
        # Render with explicit audio mapping
        result = render_single(
            folder_path=folder_path,
            background_path=background_path,
            audio_gain_db=audio_gain_db,
            use_loudnorm=use_loudnorm,
        )
        
        if result:
            rendered += 1
            paths.append(result)
        else:
            failed += 1
    
    # Update queue CSV with new statuses
    update_queue_after_render(queue_date, queue_dir, shorts_dir)
    
    return {
        "total": len(queue_items),
        "rendered": rendered,
        "skipped": skipped,
        "failed": failed,
        "paths": paths,
    }


def update_queue_after_render(
    queue_date: str,
    queue_dir: Path,
    shorts_dir: Path,
) -> None:
    """Update queue CSV with rendered statuses."""
    csv_path = queue_dir / f"queue_{queue_date}.csv"
    
    if not csv_path.exists():
        return
    
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    for row in rows:
        claim_id = int(row.get("claim_id", 0))
        folder_path = shorts_dir / str(claim_id)
        row["status"] = determine_shorts_status(folder_path).value
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Updated queue CSV after render: {csv_path}")


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def render(
    queue_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Queue date (default: today)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum items to render",
    ),
    background: Optional[str] = typer.Option(
        None,
        "--background",
        "-b",
        help="Background image path",
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
    audio_gain_db: float = typer.Option(
        6.0,
        "--audio-gain-db",
        help="Audio gain in dB when not using loudnorm (default: 6.0)",
    ),
    loudnorm: bool = typer.Option(
        True,
        "--loudnorm/--no-loudnorm",
        help="Use EBU R128 loudness normalization (default: True)",
    ),
) -> None:
    """
    Render ready shorts from queue to final.mp4.
    
    Uses explicit audio mapping and loudness normalization to ensure
    audible audio in the output video.
    """
    # Check ffmpeg first
    try:
        version = check_ffmpeg()
        console.print(f"[dim]ffmpeg: {version.split()[2] if len(version.split()) > 2 else 'OK'}[/dim]")
    except FFmpegNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    if queue_date is None:
        queue_date = date.today().isoformat()
    
    q_dir = Path(queue_dir) if queue_dir else Path("outputs/shorts_queue")
    s_dir = Path(shorts_dir) if shorts_dir else Path("outputs/shorts")
    bg_path = Path(background) if background else None
    
    # Check for default background
    if bg_path is None:
        default_bg = Path("assets/bg.png")
        if default_bg.exists():
            bg_path = default_bg
            console.print(f"[dim]Using background: {bg_path}[/dim]")
    
    console.print(f"[bold cyan]=== Render Shorts ===[/bold cyan]\n")
    console.print(f"Queue date: {queue_date}")
    console.print(f"Limit: {limit}")
    console.print(f"Audio: {'loudnorm (I=-16)' if loudnorm else f'gain +{audio_gain_db}dB'}")
    console.print("")
    
    # Check queue exists
    csv_path = q_dir / f"queue_{queue_date}.csv"
    if not csv_path.exists():
        console.print(f"[red]Queue not found: {csv_path}[/red]")
        raise typer.Exit(1)
    
    # Render
    result = render_from_queue(
        queue_date=queue_date,
        limit=limit,
        queue_dir=q_dir,
        shorts_dir=s_dir,
        background_path=bg_path,
        audio_gain_db=audio_gain_db,
        use_loudnorm=loudnorm,
    )
    
    # Print results
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Total in queue: {result['total']}")
    console.print(f"  Rendered: {result['rendered']}")
    console.print(f"  Skipped: {result['skipped']}")
    console.print(f"  Failed: {result['failed']}")
    
    if result["paths"]:
        console.print(f"\n[bold]Rendered files:[/bold]")
        for p in result["paths"][:5]:
            console.print(f"  - {p}")
        if len(result["paths"]) > 5:
            console.print(f"  ... and {len(result['paths']) - 5} more")
    
    console.print(f"\n[green]OK[/green] Render complete")


@app.command()
def single(
    claim_id: int = typer.Argument(..., help="Claim ID to render"),
    shorts_dir: Optional[str] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Shorts output directory",
    ),
    background: Optional[str] = typer.Option(
        None,
        "--background",
        "-b",
        help="Background image path",
    ),
    audio_gain_db: float = typer.Option(
        6.0,
        "--audio-gain-db",
        help="Audio gain in dB when not using loudnorm",
    ),
    loudnorm: bool = typer.Option(
        True,
        "--loudnorm/--no-loudnorm",
        help="Use EBU R128 loudness normalization",
    ),
) -> None:
    """
    Render a single shorts folder to final.mp4.
    """
    # Check ffmpeg first
    try:
        version = check_ffmpeg()
        console.print(f"[dim]ffmpeg: {version.split()[2] if len(version.split()) > 2 else 'OK'}[/dim]")
    except FFmpegNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    s_dir = Path(shorts_dir) if shorts_dir else Path("outputs/shorts")
    bg_path = Path(background) if background else None
    
    folder_path = s_dir / str(claim_id)
    
    if not folder_path.exists():
        console.print(f"[red]Folder not found: {folder_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]=== Render Single Short ===[/bold cyan]\n")
    console.print(f"Claim ID: {claim_id}")
    console.print(f"Folder: {folder_path}")
    console.print(f"Audio: {'loudnorm' if loudnorm else f'gain +{audio_gain_db}dB'}")
    console.print("")
    
    # Check status
    status = determine_shorts_status(folder_path)
    console.print(f"Current status: {status.value}")
    
    if status == ShortsStatus.RENDERED:
        console.print("[yellow]Already rendered[/yellow]")
        return
    
    if status != ShortsStatus.READY:
        console.print(f"[red]Not ready to render (status={status.value})[/red]")
        console.print("Run prepare-shorts first to complete missing steps.")
        raise typer.Exit(1)
    
    # Render
    result = render_single(
        folder_path=folder_path,
        background_path=bg_path,
        audio_gain_db=audio_gain_db,
        use_loudnorm=loudnorm,
    )
    
    if result:
        console.print(f"\n[green]OK[/green] Rendered: {result}")
        console.print(f"New status: {determine_shorts_status(folder_path).value}")
    else:
        console.print(f"\n[red]Render failed[/red]")
        raise typer.Exit(1)


@app.command()
def check() -> None:
    """
    Check ffmpeg availability and version.
    """
    console.print("[bold cyan]=== FFmpeg Check ===[/bold cyan]\n")
    
    try:
        version = check_ffmpeg()
        console.print(f"[green]OK[/green] ffmpeg is available")
        console.print(f"\n{version}")
        
        # Also check ffprobe
        try:
            probe_result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe_result.returncode == 0:
                console.print(f"\n[green]OK[/green] ffprobe is available")
        except Exception:
            console.print(f"\n[yellow]Warning: ffprobe not found[/yellow]")
        
    except FFmpegNotFoundError as e:
        console.print(f"[red]Error: ffmpeg not found[/red]")
        console.print(f"\n{e}")
        raise typer.Exit(1)


@app.command()
def verify(
    claim_id: int = typer.Argument(..., help="Claim ID to verify"),
    shorts_dir: Optional[str] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Shorts output directory",
    ),
) -> None:
    """
    Verify a rendered video has audio using ffprobe.
    """
    s_dir = Path(shorts_dir) if shorts_dir else Path("outputs/shorts")
    folder_path = s_dir / str(claim_id)
    video_path = folder_path / "final.mp4"
    
    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]=== Verify Video ===[/bold cyan]\n")
    
    video_info = probe_video_file(video_path)
    
    console.print(f"File: {video_path}")
    console.print(f"Duration: {video_info.duration:.1f}s")
    console.print(f"Video stream: {video_info.has_video_stream} ({video_info.video_codec})")
    console.print(f"Audio stream: {video_info.has_audio_stream} ({video_info.audio_codec})")
    
    if video_info.has_audio_stream:
        console.print(f"\n[green]OK - AUDIO PRESENT[/green]")
    else:
        console.print(f"\n[red]FAIL - NO AUDIO STREAM[/red]")
        raise typer.Exit(1)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
