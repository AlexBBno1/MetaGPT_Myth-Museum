"""
Myth Museum - Series Overlay Generator

Generates ASS subtitle overlays for series markers (e.g., "Greek Myths #2").
Displays in top-right corner throughout the video.
"""

import re
from pathlib import Path
from typing import Optional

from core.logging import get_logger
from pipeline.shorts_optimizer import SeriesInfo, ShortsOptimizer

logger = get_logger(__name__)


# ============================================================================
# ASS Template for Series Marker
# ============================================================================

# Video resolution (YouTube Shorts 9:16)
PLAY_RES_X = 1080
PLAY_RES_Y = 1920

# Series marker style settings
SERIES_MARKER_STYLE = """Style: SeriesMarker,Arial,32,&H99FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,9,30,30,30,1"""

# Alignment 9 = top-right
# FontSize 32 = small but readable
# PrimaryColour &H99FFFFFF = 60% white (semi-transparent)
# BorderStyle 1 = outline + shadow
# MarginR 30 = 30px from right edge


def generate_series_marker_ass(
    series_info: SeriesInfo,
    duration: float,
) -> str:
    """
    Generate ASS content for series marker overlay.
    
    Args:
        series_info: Series information
        duration: Video duration in seconds
    
    Returns:
        ASS format string for the series marker
    """
    # Format duration as ASS timestamp
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    centiseconds = int((duration % 1) * 100)
    end_time = f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    ass_content = f"""[Script Info]
Title: Series Marker Overlay
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{SERIES_MARKER_STYLE}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,{end_time},SeriesMarker,,0,0,0,,{series_info.display_text}
"""
    
    return ass_content


def inject_series_marker_into_ass(
    ass_path: Path,
    series_info: SeriesInfo,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Inject series marker into existing ASS file.
    
    Adds:
    1. SeriesMarker style to V4+ Styles section
    2. Dialogue line for series marker spanning full video
    
    Args:
        ass_path: Path to existing ASS file
        series_info: Series information
        output_path: Optional output path (defaults to overwriting input)
    
    Returns:
        Path to modified ASS file
    """
    output_path = output_path or ass_path
    
    # Read existing ASS
    ass_content = ass_path.read_text(encoding="utf-8")
    
    # Check if SeriesMarker style already exists
    if "SeriesMarker" in ass_content:
        logger.info("Series marker already present in ASS")
        return ass_path
    
    # Inject style into V4+ Styles section
    styles_match = re.search(r'(\[V4\+ Styles\].*?Format:.*?\n)', ass_content, re.DOTALL)
    if styles_match:
        insert_pos = styles_match.end()
        ass_content = (
            ass_content[:insert_pos] + 
            SERIES_MARKER_STYLE + "\n" +
            ass_content[insert_pos:]
        )
    
    # Find video duration from last dialogue line
    dialogue_times = re.findall(r'Dialogue:.*?,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+)', ass_content)
    if dialogue_times:
        # Get the latest end time
        end_time = max(dialogue_times, key=lambda x: x[1])[1]
    else:
        end_time = "0:01:00.00"  # Default 1 minute
    
    # Add series marker dialogue at end
    series_dialogue = f"\nDialogue: 1,0:00:00.00,{end_time},SeriesMarker,,0,0,0,,{series_info.display_text}"
    
    # Find Events section and append
    if "[Events]" in ass_content:
        ass_content = ass_content.rstrip() + series_dialogue + "\n"
    
    # Write output
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info(f"Injected series marker '{series_info.display_text}' into {output_path}")
    
    return output_path


def create_series_marker_overlay(
    topic: str,
    duration: float,
    output_path: Path,
) -> tuple[Path, SeriesInfo]:
    """
    Create standalone series marker overlay ASS file.
    
    Args:
        topic: Video topic (used to detect series)
        duration: Video duration in seconds
        output_path: Path to save ASS file
    
    Returns:
        Tuple of (output_path, series_info)
    """
    # Get series info
    optimizer = ShortsOptimizer()
    series_info = optimizer.get_series_info(topic, increment=True)
    
    # Generate ASS content
    ass_content = generate_series_marker_ass(series_info, duration)
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass_content, encoding="utf-8")
    
    logger.info(f"Created series marker overlay: {output_path}")
    return output_path, series_info


def add_series_marker_to_video(
    video_path: Path,
    ass_path: Path,
    topic: str,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Add series marker to existing ASS and re-render video.
    
    This is a convenience function that:
    1. Injects series marker into ASS
    2. Re-renders video with updated subtitles
    
    Args:
        video_path: Path to video (without subtitles)
        ass_path: Path to ASS subtitle file
        topic: Video topic
        output_path: Optional output path
    
    Returns:
        Path to output video
    """
    import subprocess
    
    # Get series info
    optimizer = ShortsOptimizer()
    series_info = optimizer.get_series_info(topic, increment=True)
    
    # Inject into ASS
    inject_series_marker_into_ass(ass_path, series_info)
    
    # Determine output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_series{video_path.suffix}"
    
    # Re-render with FFmpeg
    ass_path_str = str(ass_path).replace('\\', '/')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f"ass='{ass_path_str}'",
        '-c:a', 'copy',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to render video with series marker")
    
    logger.info(f"Created video with series marker: {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Series Marker Overlay Generator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create overlay
    create_parser = subparsers.add_parser("create", help="Create series marker overlay")
    create_parser.add_argument("--topic", "-t", required=True, help="Video topic")
    create_parser.add_argument("--duration", "-d", type=float, required=True, help="Video duration")
    create_parser.add_argument("--output", "-o", required=True, help="Output ASS path")
    
    # Inject into existing ASS
    inject_parser = subparsers.add_parser("inject", help="Inject marker into existing ASS")
    inject_parser.add_argument("--ass", "-a", required=True, help="Existing ASS file")
    inject_parser.add_argument("--topic", "-t", required=True, help="Video topic")
    inject_parser.add_argument("--output", "-o", help="Output path (default: overwrite)")
    
    args = parser.parse_args()
    
    if args.command == "create":
        output_path, series_info = create_series_marker_overlay(
            topic=args.topic,
            duration=args.duration,
            output_path=Path(args.output),
        )
        print(f"Created: {output_path}")
        print(f"Series: {series_info.display_text}")
    
    elif args.command == "inject":
        optimizer = ShortsOptimizer()
        series_info = optimizer.get_series_info(args.topic, increment=True)
        
        output_path = Path(args.output) if args.output else None
        inject_series_marker_into_ass(
            ass_path=Path(args.ass),
            series_info=series_info,
            output_path=output_path,
        )
        print(f"Injected: {series_info.display_text}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
