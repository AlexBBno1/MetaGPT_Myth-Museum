"""
Myth Museum - Video Short Composer

Compose final short video from B-roll clips, voiceover, and captions.

Input:
    - broll/*.mp4 (video clips from Veo)
    - voiceover.mp3 (TTS audio)
    - captions.srt or captions.ass (subtitles)

Output:
    - background.mp4 (concatenated B-roll)
    - final.mp4 (complete video with audio and subtitles)

Specifications:
    - Resolution: 1080x1920 (9:16 vertical)
    - Frame rate: 30fps
    - Video codec: h264
    - Audio codec: aac
    - Audio normalization: loudnorm
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from core.logging import get_logger
from pipeline.tts import get_audio_duration

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="compose",
    help="Compose video shorts from B-roll clips",
    add_completion=False,
)


# ============================================================================
# Constants
# ============================================================================

# Target video specs for Shorts
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
CRF = 23  # Quality (lower = better, 18-28 typical)

# Fallback solid color (dark blue) for missing B-roll
FALLBACK_COLOR = "0x1a1a2e"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ComposeResult:
    """Result of video composition."""
    success: bool = False
    background_path: Optional[Path] = None
    final_path: Optional[Path] = None
    duration: float = 0.0
    error: Optional[str] = None
    clips_used: int = 0
    fallback_used: bool = False
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "background_path": str(self.background_path) if self.background_path else None,
            "final_path": str(self.final_path) if self.final_path else None,
            "duration": self.duration,
            "error": self.error,
            "clips_used": self.clips_used,
            "fallback_used": self.fallback_used,
        }


# ============================================================================
# FFmpeg Utilities
# ============================================================================

def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using FFprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}")
    
    return 0.0


def escape_ffmpeg_path(path: Path) -> str:
    """
    Escape path for FFmpeg filter (especially for Windows).
    
    FFmpeg on Windows requires special escaping for paths with
    backslashes and colons in filter arguments.
    """
    path_str = str(path.absolute())
    
    # Convert backslashes to forward slashes
    path_str = path_str.replace("\\", "/")
    
    # Escape colons (Windows drive letters)
    path_str = path_str.replace(":", r"\:")
    
    return path_str


# ============================================================================
# Concatenation
# ============================================================================

def concat_broll_clips(
    clips: list[Path],
    output_path: Path,
    target_duration: float,
) -> bool:
    """
    Concatenate B-roll clips to match target duration.
    
    If clips are shorter than target, they will be looped.
    If clips are longer, they will be trimmed.
    
    Args:
        clips: List of video clip paths
        output_path: Output video path
        target_duration: Target duration in seconds
    
    Returns:
        True if successful
    """
    if not clips:
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total clip duration
    clip_durations = [get_video_duration(c) for c in clips]
    total_clip_duration = sum(clip_durations)
    
    logger.info(f"Total B-roll duration: {total_clip_duration:.1f}s, target: {target_duration:.1f}s")
    
    # Create concat file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        concat_file = Path(f.name)
        
        # If clips are shorter than target, repeat them
        if total_clip_duration < target_duration:
            repeats = int(target_duration / total_clip_duration) + 1
            for _ in range(repeats):
                for clip in clips:
                    f.write(f"file '{clip.absolute()}'\n")
        else:
            for clip in clips:
                f.write(f"file '{clip.absolute()}'\n")
    
    try:
        # Concatenate and trim to target duration
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-t", str(target_duration),
            "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}",
            "-c:v", VIDEO_CODEC,
            "-preset", "medium",
            "-crf", str(CRF),
            "-an",  # No audio in background
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg concat failed: {result.stderr[:500]}")
            return False
        
        return True
        
    finally:
        # Clean up concat file
        concat_file.unlink(missing_ok=True)


def create_fallback_background(
    output_path: Path,
    duration: float,
    color: str = FALLBACK_COLOR,
) -> bool:
    """
    Create solid color background video as fallback.
    
    Args:
        output_path: Output video path
        duration: Duration in seconds
        color: Hex color (e.g., "0x1a1a2e")
    
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:d={duration}:r={TARGET_FPS}",
        "-c:v", VIDEO_CODEC,
        "-preset", "fast",
        "-crf", str(CRF),
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg fallback failed: {result.stderr[:500]}")
        return False
    
    return True


# ============================================================================
# Final Composition
# ============================================================================

def compose_final_video(
    background_path: Path,
    audio_path: Path,
    subtitles_path: Optional[Path],
    output_path: Path,
) -> bool:
    """
    Compose final video with audio and subtitles.
    
    Args:
        background_path: Background video (B-roll concat)
        audio_path: Audio file (voiceover)
        subtitles_path: Subtitles file (.srt or .ass)
        output_path: Output video path
    
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", str(background_path),
        "-i", str(audio_path),
    ]
    
    # Add subtitle filter if available
    if subtitles_path and subtitles_path.exists():
        sub_ext = subtitles_path.suffix.lower()
        sub_path_escaped = escape_ffmpeg_path(subtitles_path)
        
        if sub_ext == ".ass":
            # ASS subtitles (styled)
            video_filter = f"ass='{sub_path_escaped}'"
        else:
            # SRT subtitles (plain)
            video_filter = f"subtitles='{sub_path_escaped}':force_style='FontSize=48,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=5'"
        
        cmd.extend(["-vf", video_filter])
    
    # Output settings with audio normalization
    cmd.extend([
        "-c:v", VIDEO_CODEC,
        "-preset", "medium",
        "-crf", str(CRF),
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg compose failed: {result.stderr[:500]}")
        return False
    
    return True


# ============================================================================
# Main Composition Pipeline
# ============================================================================

def compose_video_short(
    folder: Path,
    voiceover: Optional[Path] = None,
    captions: Optional[Path] = None,
) -> ComposeResult:
    """
    Compose a complete video short from folder assets.
    
    Expects folder structure:
        folder/
            broll/
                broll_01.mp4
                broll_02.mp4
                ...
            voiceover.mp3
            captions.srt (or captions.ass)
    
    Args:
        folder: Shorts folder path
        voiceover: Override voiceover path
        captions: Override captions path
    
    Returns:
        ComposeResult
    """
    result = ComposeResult()
    folder = Path(folder)
    
    if not folder.exists():
        result.error = f"Folder not found: {folder}"
        return result
    
    # Find assets
    voiceover_path = voiceover or folder / "voiceover.mp3"
    if not voiceover_path.exists():
        result.error = f"Voiceover not found: {voiceover_path}"
        return result
    
    # Get target duration from audio
    duration = get_audio_duration(voiceover_path)
    if duration <= 0:
        result.error = "Could not determine audio duration"
        return result
    
    result.duration = duration
    
    # Find B-roll clips
    broll_dir = folder / "broll"
    broll_clips = []
    
    if broll_dir.exists():
        broll_clips = sorted(broll_dir.glob("broll_*.mp4"))
    
    result.clips_used = len(broll_clips)
    
    # Create background video
    background_path = folder / "background.mp4"
    
    if broll_clips:
        logger.info(f"Concatenating {len(broll_clips)} B-roll clips...")
        success = concat_broll_clips(broll_clips, background_path, duration)
        
        if not success:
            logger.warning("B-roll concat failed, using fallback")
            result.fallback_used = True
            success = create_fallback_background(background_path, duration)
            
            if not success:
                result.error = "Failed to create background video"
                return result
    else:
        logger.warning("No B-roll clips found, using fallback background")
        result.fallback_used = True
        success = create_fallback_background(background_path, duration)
        
        if not success:
            result.error = "Failed to create fallback background"
            return result
    
    result.background_path = background_path
    
    # Find captions
    captions_path = captions
    if not captions_path:
        # Prefer ASS over SRT
        ass_path = folder / "captions.ass"
        srt_path = folder / "captions.srt"
        
        if ass_path.exists():
            captions_path = ass_path
        elif srt_path.exists():
            captions_path = srt_path
    
    # Compose final video
    final_path = folder / "final.mp4"
    logger.info("Composing final video...")
    
    success = compose_final_video(
        background_path=background_path,
        audio_path=voiceover_path,
        subtitles_path=captions_path,
        output_path=final_path,
    )
    
    if not success:
        result.error = "Failed to compose final video"
        return result
    
    result.success = True
    result.final_path = final_path
    
    logger.info(f"Composed video: {final_path} ({duration:.1f}s)")
    return result


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def compose(
    folder: Path = typer.Argument(..., help="Shorts folder with B-roll clips"),
    voiceover: Optional[Path] = typer.Option(
        None, "--voiceover", "-a",
        help="Override voiceover path",
    ),
    captions: Optional[Path] = typer.Option(
        None, "--captions", "-c",
        help="Override captions path",
    ),
) -> None:
    """
    Compose final video from folder assets.
    
    Expects broll/*.mp4, voiceover.mp3, and captions.srt/ass.
    """
    console.print("[bold cyan]=== Video Short Composer ===[/bold cyan]\n")
    
    if not check_ffmpeg():
        console.print("[red]FFmpeg not found! Please install FFmpeg.[/red]")
        raise typer.Exit(1)
    
    folder = Path(folder)
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Folder: {folder}")
    
    # List available assets
    broll_dir = folder / "broll"
    if broll_dir.exists():
        clips = list(broll_dir.glob("broll_*.mp4"))
        console.print(f"B-roll clips: {len(clips)}")
    else:
        console.print(f"B-roll clips: [yellow]None found[/yellow]")
    
    voiceover_path = voiceover or folder / "voiceover.mp3"
    if voiceover_path.exists():
        duration = get_audio_duration(voiceover_path)
        console.print(f"Voiceover: {voiceover_path} ({duration:.1f}s)")
    else:
        console.print(f"Voiceover: [red]Not found[/red]")
        raise typer.Exit(1)
    
    # Find captions
    captions_path = captions
    if not captions_path:
        for ext in [".ass", ".srt"]:
            path = folder / f"captions{ext}"
            if path.exists():
                captions_path = path
                break
    
    if captions_path and captions_path.exists():
        console.print(f"Captions: {captions_path}")
    else:
        console.print(f"Captions: [yellow]None found[/yellow]")
    
    # Compose
    console.print("\n[bold]Composing video...[/bold]")
    
    result = compose_video_short(
        folder=folder,
        voiceover=voiceover,
        captions=captions,
    )
    
    if result.success:
        console.print(f"\n[green]Success![/green]")
        console.print(f"Background: {result.background_path}")
        console.print(f"Final: {result.final_path}")
        console.print(f"Duration: {result.duration:.1f}s")
        console.print(f"B-roll clips used: {result.clips_used}")
        
        if result.fallback_used:
            console.print(f"[yellow]Note: Fallback background used[/yellow]")
        
        # Save result
        result_path = folder / "compose_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
    else:
        console.print(f"\n[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@app.command()
def check() -> None:
    """
    Check FFmpeg availability and version.
    """
    console.print("[bold cyan]=== FFmpeg Check ===[/bold cyan]\n")
    
    if not check_ffmpeg():
        console.print("[red]FFmpeg not found![/red]")
        console.print("\nPlease install FFmpeg:")
        console.print("  Windows: choco install ffmpeg")
        console.print("  macOS: brew install ffmpeg")
        console.print("  Linux: apt install ffmpeg")
        raise typer.Exit(1)
    
    # Get version info
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
    )
    
    version_line = result.stdout.split('\n')[0]
    console.print(f"[green]FFmpeg found![/green]")
    console.print(f"Version: {version_line}")


@app.command()
def preview_concat(
    folder: Path = typer.Argument(..., help="Folder with broll/ clips"),
    output: Path = typer.Option(
        Path("preview_concat.mp4"), "--output", "-o",
        help="Output path",
    ),
    duration: float = typer.Option(
        60.0, "--duration", "-d",
        help="Target duration in seconds",
    ),
) -> None:
    """
    Preview B-roll concatenation without audio.
    """
    console.print("[bold cyan]=== B-roll Concat Preview ===[/bold cyan]\n")
    
    if not check_ffmpeg():
        console.print("[red]FFmpeg not found![/red]")
        raise typer.Exit(1)
    
    folder = Path(folder)
    broll_dir = folder / "broll"
    
    if not broll_dir.exists():
        console.print(f"[red]B-roll directory not found: {broll_dir}[/red]")
        raise typer.Exit(1)
    
    clips = sorted(broll_dir.glob("broll_*.mp4"))
    if not clips:
        console.print(f"[red]No B-roll clips found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(clips)} clips")
    console.print(f"Target duration: {duration}s")
    
    success = concat_broll_clips(clips, output, duration)
    
    if success:
        console.print(f"\n[green]Success![/green]")
        console.print(f"Output: {output}")
    else:
        console.print(f"\n[red]Failed to concatenate clips[/red]")
        raise typer.Exit(1)


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
