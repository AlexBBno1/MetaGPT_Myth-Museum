"""
Myth Museum - Video Short Generator (Veo Pipeline)

Generate short videos using AI-generated B-roll clips from Google Veo.

This is the video-based pipeline counterpart to generate_short.py (image-based).
Output folders are distinguished by suffix: _video vs _image.

Usage:
    python -m pipeline.generate_video_short "Napoleon height myth"
    python -m pipeline.generate_video_short "Aztec civilization" --series "Lost Civs"
    python -m pipeline.generate_video_short "Hades god" --script-file script.txt
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.logging import get_logger
from pipeline.folder_naming import generate_folder_name
from pipeline.shorts_optimizer import ShortsOptimizer, SeriesInfo
from pipeline.shotlist_generator import (
    generate_shotlist_with_llm,
    write_shotlist_csv,
    ShotlistEntry,
)
from pipeline.veo import UnifiedVeoProvider, VeoProvider, VeoResult
from pipeline.compose_short import compose_video_short, check_ffmpeg, ComposeResult
from pipeline.tts import GoogleTTSProvider, get_audio_duration

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="generate-video-short",
    help="Generate video shorts using Veo B-roll",
    add_completion=False,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VideoGenerationConfig:
    """Configuration for video generation."""
    topic: str
    script: Optional[str] = None
    series_override: Optional[str] = None
    output_dir: Path = Path("outputs/shorts")
    voice: str = "en-US-Casual-K"
    num_shots: int = 4
    veo_provider: VeoProvider = VeoProvider.GEMINI
    
    # Derived
    folder_name: str = ""
    output_folder: Optional[Path] = None


@dataclass
class VideoGenerationResult:
    """Result of video generation."""
    success: bool = False
    output_folder: Optional[Path] = None
    final_video: Optional[Path] = None
    duration: float = 0.0
    series_info: Optional[SeriesInfo] = None
    error: Optional[str] = None
    
    # Pipeline results
    shotlist_generated: bool = False
    broll_results: list[VeoResult] = field(default_factory=list)
    compose_result: Optional[ComposeResult] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_folder": str(self.output_folder) if self.output_folder else None,
            "final_video": str(self.final_video) if self.final_video else None,
            "duration": self.duration,
            "series_info": self.series_info.to_dict() if self.series_info else None,
            "error": self.error,
            "shotlist_generated": self.shotlist_generated,
            "broll_count": len(self.broll_results),
            "broll_success": sum(1 for r in self.broll_results if r.success),
        }


# ============================================================================
# Main Generation Pipeline
# ============================================================================

class VideoShortGenerator:
    """
    Video short generator using Veo B-roll.
    
    Pipeline:
    1. Script optimization
    2. Shotlist generation (LLM)
    3. B-roll generation (Veo)
    4. TTS audio generation
    5. Subtitle creation
    6. Video composition (FFmpeg)
    """
    
    def __init__(self, veo_provider: VeoProvider = VeoProvider.GEMINI):
        self.optimizer = ShortsOptimizer()
        self.veo = UnifiedVeoProvider(preferred_provider=veo_provider)
        self.tts_provider = GoogleTTSProvider()
    
    async def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        Generate a complete video short with B-roll.
        
        Args:
            config: Generation configuration
        
        Returns:
            VideoGenerationResult
        """
        result = VideoGenerationResult()
        
        try:
            # Phase 1: Setup
            console.print(f"\n[bold cyan]Phase 1: Setup[/bold cyan]")
            
            # Get series info
            series_info = self.optimizer.get_series_info(
                topic=config.topic,
                script=config.script or "",
                series_override=config.series_override,
                increment=True,
            )
            result.series_info = series_info
            
            # Create folder with _video suffix
            base_name = generate_folder_name(series_info.series_name, config.topic)
            folder_name = f"{base_name}_video"
            output_folder = config.output_dir / folder_name
            output_folder.mkdir(parents=True, exist_ok=True)
            (output_folder / "broll").mkdir(exist_ok=True)
            
            result.output_folder = output_folder
            console.print(f"  Output: {output_folder}")
            console.print(f"  Series: {series_info.display_text}")
            
            # Phase 2: Script optimization
            console.print(f"\n[bold cyan]Phase 2: Script Optimization[/bold cyan]")
            
            if config.script:
                script = config.script
            else:
                script = f"This is a placeholder script about {config.topic}. Replace with actual content."
                console.print(f"  [yellow]Warning: No script provided, using placeholder[/yellow]")
            
            optimized = self.optimizer.optimize_script(
                script=script,
                topic=config.topic,
                series_override=config.series_override,
            )
            
            # Save script
            (output_folder / "voiceover.txt").write_text(
                optimized.optimized_script, encoding="utf-8"
            )
            
            console.print(f"  Retention hook: \"{optimized.retention_hook}\"")
            console.print(f"  Comment trigger: \"{optimized.comment_trigger}\"")
            
            # Save optimization report
            with open(output_folder / "optimization_report.json", "w", encoding="utf-8") as f:
                json.dump(optimized.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Phase 3: TTS generation (do this early to get duration)
            console.print(f"\n[bold cyan]Phase 3: TTS Generation[/bold cyan]")
            
            audio_path = output_folder / "voiceover.mp3"
            await self.tts_provider.synthesize(
                text=optimized.optimized_script,
                output_path=audio_path,
                voice=config.voice,
            )
            
            duration = get_audio_duration(audio_path)
            result.duration = duration
            console.print(f"  Audio: {audio_path}")
            console.print(f"  Duration: {duration:.1f}s")
            
            # Phase 4: Shotlist generation
            console.print(f"\n[bold cyan]Phase 4: Shotlist Generation[/bold cyan]")
            console.print("  Generating shotlist via LLM...")
            
            shotlist_entries = await generate_shotlist_with_llm(
                script=optimized.optimized_script,
                topic=config.topic,
                duration_hint=duration,
                num_shots=config.num_shots,
            )
            
            if not shotlist_entries:
                result.error = "Failed to generate shotlist"
                return result
            
            result.shotlist_generated = True
            
            # Save shotlist
            shotlist_path = output_folder / "shotlist.csv"
            write_shotlist_csv(shotlist_entries, shotlist_path)
            console.print(f"  Shotlist: {shotlist_path} ({len(shotlist_entries)} shots)")
            
            for entry in shotlist_entries:
                console.print(f"    [{entry.shot_id}] {entry.segment}: {entry.visual_description[:50]}...")
            
            # Phase 5: B-roll generation
            console.print(f"\n[bold cyan]Phase 5: B-roll Generation (Veo)[/bold cyan]")
            
            available_providers = self.veo.get_available_providers()
            if not available_providers:
                console.print(f"  [yellow]Warning: No Veo providers available[/yellow]")
                console.print(f"  [yellow]Skipping B-roll generation, will use fallback background[/yellow]")
            else:
                console.print(f"  Available providers: {', '.join(available_providers)}")
                
                broll_results = await self.veo.generate_broll_for_folder(
                    folder=output_folder,
                    shotlist_csv=shotlist_path,
                )
                
                result.broll_results = broll_results
                success_count = sum(1 for r in broll_results if r.success)
                console.print(f"  B-roll generated: {success_count}/{len(broll_results)}")
            
            # Phase 6: Subtitles
            console.print(f"\n[bold cyan]Phase 6: Subtitles[/bold cyan]")
            
            srt_path = output_folder / "captions.srt"
            ass_path = output_folder / "captions.ass"
            
            self._create_ass_with_series_marker(
                srt_path=srt_path,
                ass_path=ass_path,
                series_text=series_info.display_text,
                duration=duration,
            )
            console.print(f"  ASS: {ass_path}")
            
            # Phase 7: Video composition
            console.print(f"\n[bold cyan]Phase 7: Video Composition[/bold cyan]")
            
            if not check_ffmpeg():
                console.print(f"  [red]FFmpeg not found - skipping composition[/red]")
                result.error = "FFmpeg not available"
                return result
            
            compose_result = compose_video_short(
                folder=output_folder,
                voiceover=audio_path,
                captions=ass_path,
            )
            
            result.compose_result = compose_result
            
            if not compose_result.success:
                console.print(f"  [red]Composition failed: {compose_result.error}[/red]")
                result.error = compose_result.error
                return result
            
            if compose_result.fallback_used:
                console.print(f"  [yellow]Note: Using fallback background (no B-roll)[/yellow]")
            
            console.print(f"  Background: {compose_result.background_path}")
            console.print(f"  Final: {compose_result.final_path}")
            
            result.success = True
            result.final_video = compose_result.final_path
            
            # Save metadata
            metadata = {
                "topic": config.topic,
                "series": series_info.to_dict(),
                "duration": duration,
                "type": "video",
                "veo_provider": config.veo_provider.value,
                "shots_generated": len(shotlist_entries),
                "broll_success": sum(1 for r in result.broll_results if r.success),
                "fallback_used": compose_result.fallback_used,
                "voice": config.voice,
            }
            with open(output_folder / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Save series info
            with open(output_folder / "series_info.json", "w", encoding="utf-8") as f:
                json.dump(series_info.to_dict(), f, indent=2)
            
            console.print(f"\n[bold green]Complete![/bold green]")
            console.print(f"Output: {compose_result.final_path}")
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            result.error = str(e)
            console.print(f"\n[bold red]Error: {e}[/bold red]")
        
        return result
    
    def _create_ass_with_series_marker(
        self,
        srt_path: Path,
        ass_path: Path,
        series_text: str,
        duration: float,
    ) -> None:
        """Create ASS file with series marker overlay."""
        import re
        
        ass_content = f'''[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,4,0,5,80,80,300,1
Style: SeriesMarker,Arial,32,&H99FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,9,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
        
        # Parse SRT if it exists
        if srt_path.exists():
            srt_content = srt_path.read_text(encoding="utf-8")
            srt_blocks = re.split(r'\n\n+', srt_content.strip())
            
            for block in srt_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    time_match = re.match(
                        r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})',
                        lines[1]
                    )
                    if time_match:
                        sh, sm, ss, sms = time_match.groups()[:4]
                        eh, em, es, ems = time_match.groups()[4:]
                        start = f"{int(sh)}:{sm}:{ss}.{sms[:2]}"
                        end = f"{int(eh)}:{em}:{es}.{ems[:2]}"
                        text = ' '.join(lines[2:]).replace('\n', ' ')
                        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
        
        # Add series marker
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        centiseconds = int((duration % 1) * 100)
        end_time = f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
        ass_content += f"Dialogue: 1,0:00:00.00,{end_time},SeriesMarker,,0,0,0,,{series_text}\n"
        
        ass_path.write_text(ass_content, encoding="utf-8")


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def generate(
    topic: str = typer.Argument(..., help="Video topic"),
    script_file: Optional[Path] = typer.Option(
        None, "--script-file", "-f",
        help="Path to script file (voiceover.txt)",
    ),
    script: Optional[str] = typer.Option(
        None, "--script", "-s",
        help="Script text directly",
    ),
    series: Optional[str] = typer.Option(
        None, "--series",
        help="Override series name (e.g., 'Greek Myths', 'History Lies')",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/shorts"), "--output-dir", "-o",
        help="Output directory",
    ),
    voice: str = typer.Option(
        "en-US-Casual-K", "--voice", "-v",
        help="TTS voice name",
    ),
    num_shots: int = typer.Option(
        4, "--shots", "-n",
        help="Number of video shots to generate",
    ),
    provider: VeoProvider = typer.Option(
        VeoProvider.GEMINI, "--provider", "-p",
        help="Preferred Veo provider",
    ),
) -> None:
    """
    Generate a video short using Veo B-roll.
    
    This creates a video with AI-generated video clips instead of static images.
    
    Examples:
        python -m pipeline.generate_video_short "Napoleon height myth"
        python -m pipeline.generate_video_short "Aztec civilization" --series "Lost Civs"
        python -m pipeline.generate_video_short "Hades" -f my_script.txt
    """
    console.print("[bold cyan]=== Video Short Generator (Veo) ===[/bold cyan]\n")
    console.print(f"Topic: {topic}")
    
    # Load script from file if provided
    script_text = script
    if script_file and script_file.exists():
        script_text = script_file.read_text(encoding="utf-8")
        console.print(f"Script: {script_file}")
    elif script:
        console.print(f"Script: (provided inline)")
    else:
        console.print(f"Script: (will use placeholder)")
    
    if series:
        console.print(f"Series: {series} (override)")
    
    console.print(f"Provider: {provider.value}")
    
    # Create config
    config = VideoGenerationConfig(
        topic=topic,
        script=script_text,
        series_override=series,
        output_dir=output_dir,
        voice=voice,
        num_shots=num_shots,
        veo_provider=provider,
    )
    
    # Run generation
    generator = VideoShortGenerator(veo_provider=provider)
    result = asyncio.run(generator.generate(config))
    
    if result.success:
        console.print(f"\n[green]Success![/green]")
        console.print(f"Video: {result.final_video}")
        console.print(f"Duration: {result.duration:.1f}s")
        if result.series_info:
            console.print(f"Series: {result.series_info.display_text}")
        console.print(f"Type: Video (B-roll)")
    else:
        console.print(f"\n[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@app.command()
def shotlist(
    topic: str = typer.Argument(..., help="Video topic"),
    script_file: Optional[Path] = typer.Option(
        None, "--script-file", "-f",
        help="Path to script file",
    ),
    output: Path = typer.Option(
        Path("shotlist.csv"), "--output", "-o",
        help="Output CSV path",
    ),
    duration: float = typer.Option(
        60.0, "--duration", "-d",
        help="Expected video duration in seconds",
    ),
    num_shots: int = typer.Option(
        4, "--shots", "-n",
        help="Number of shots",
    ),
) -> None:
    """
    Generate just the shotlist (no video).
    
    Useful for previewing/editing shotlist before generating B-roll.
    """
    console.print("[bold cyan]=== Shotlist Generator ===[/bold cyan]\n")
    
    # Get script
    script_text = ""
    if script_file and script_file.exists():
        script_text = script_file.read_text(encoding='utf-8')
        console.print(f"Script: {script_file}")
    else:
        script_text = f"A video about {topic}."
        console.print(f"Script: (using topic only)")
    
    console.print(f"Topic: {topic}")
    console.print(f"Duration: {duration}s")
    
    # Generate shotlist
    from pipeline.shotlist_generator import generate_shotlist_with_llm, write_shotlist_csv
    
    entries = asyncio.run(generate_shotlist_with_llm(
        script=script_text,
        topic=topic,
        duration_hint=duration,
        num_shots=num_shots,
    ))
    
    if not entries:
        console.print("[red]Failed to generate shotlist[/red]")
        raise typer.Exit(1)
    
    write_shotlist_csv(entries, output)
    
    console.print(f"\n[green]Generated {len(entries)} shots[/green]")
    console.print(f"Output: {output}")
    
    for entry in entries:
        console.print(f"\n[bold]{entry.shot_id}. {entry.segment}[/bold] ({entry.start_time:.0f}s - {entry.end_time:.0f}s)")
        console.print(f"   {entry.visual_description}")


@app.command("list-series")
def list_series() -> None:
    """List all available series and their episode counts."""
    from pipeline.shorts_optimizer import SeriesRegistry
    
    console.print("[bold cyan]=== Series Registry ===[/bold cyan]\n")
    
    registry = SeriesRegistry()
    series = registry.get_all_series()
    
    if not series:
        console.print("[yellow]No series recorded yet.[/yellow]")
        return
    
    for name, count in sorted(series.items()):
        console.print(f"  {name}: {count} episodes")


@app.command()
def status() -> None:
    """Check Veo provider status."""
    from pipeline.veo import GeminiVeoProvider, VertexVeoProvider
    from pipeline.compose_short import check_ffmpeg
    
    console.print("[bold cyan]=== Video Pipeline Status ===[/bold cyan]\n")
    
    # FFmpeg
    console.print("[bold]FFmpeg:[/bold]")
    if check_ffmpeg():
        console.print(f"  Status: [green]Available[/green]")
    else:
        console.print(f"  Status: [red]Not found[/red]")
        console.print(f"  Install: choco install ffmpeg (Windows)")
    
    console.print()
    
    # Veo providers
    gemini = GeminiVeoProvider()
    vertex = VertexVeoProvider()
    
    console.print("[bold]Gemini API (Veo):[/bold]")
    if gemini.is_available():
        console.print(f"  Status: [green]Available[/green]")
        console.print(f"  API Key: {gemini.api_key[:10]}...")
    else:
        console.print(f"  Status: [yellow]Not configured[/yellow]")
        console.print(f"  Set GEMINI_API_KEY environment variable")
    
    console.print()
    
    console.print("[bold]Vertex AI (Veo):[/bold]")
    if vertex.is_available():
        console.print(f"  Status: [green]Available[/green]")
        console.print(f"  Project: {vertex.project_id}")
    else:
        console.print(f"  Status: [yellow]Not configured[/yellow]")
        console.print(f"  Set VERTEX_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS")


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
