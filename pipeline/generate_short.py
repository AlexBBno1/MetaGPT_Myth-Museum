"""
Myth Museum - Unified Short Video Generator

Single entry point to generate a complete short video from a topic.

Usage:
    python -m pipeline.generate_short "Napoleon height myth"
    python -m pipeline.generate_short "Aztec civilization" --series "Lost Civs"
    python -m pipeline.generate_short "Hades god" --script-file script.txt
    python -m pipeline.generate_short "Jordan flu game" --auto-prompts  # Uses LLM visual director
"""

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from core.logging import get_logger
from pipeline.shorts_optimizer import (
    ShortsOptimizer,
    OptimizedScript,
    SeriesInfo,
    validate_script_length,
    calculate_optimal_images,
    estimate_duration_from_script,
)
from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
from pipeline.tts import GoogleTTSProvider, get_audio_duration
from pipeline.folder_naming import generate_folder_name, slugify
from pipeline.visual_director import VisualDirector, generate_scene_prompts

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="generate-short",
    help="Generate a complete short video from a topic",
    add_completion=False,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    topic: str
    script: Optional[str] = None
    series_override: Optional[str] = None
    output_dir: Path = Path("outputs/shorts")
    voice: str = "en-US-Casual-K"
    num_images: int = 0  # 0 = auto-calculate based on script
    
    # Image generation
    image_prompts: list[dict] = field(default_factory=list)
    fallback_keywords: list[str] = field(default_factory=list)
    image_quality: str = "high"  # "high" (Imagen 3), "standard", or "fallback"
    
    # Visual director settings
    auto_prompts: bool = True  # Use LLM visual director for prompts
    
    # TTS settings
    tts_rate: str = "+0%"  # Speech rate adjustment
    
    # Video effects
    use_ken_burns: bool = True  # Slow zoom/pan effect
    use_crossfade: bool = True  # Crossfade between images


@dataclass
class GenerationResult:
    """Result of video generation."""
    success: bool = False
    output_folder: Optional[Path] = None
    final_video: Optional[Path] = None
    duration: float = 0.0
    series_info: Optional[SeriesInfo] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_folder": str(self.output_folder) if self.output_folder else None,
            "final_video": str(self.final_video) if self.final_video else None,
            "duration": self.duration,
            "series_info": self.series_info.to_dict() if self.series_info else None,
            "error": self.error,
        }


# ============================================================================
# Default Image Prompts
# ============================================================================

def generate_default_prompts(topic: str) -> list[dict]:
    """
    Generate default image prompts based on topic.
    
    Returns 4 prompts for: hook, context, turning point, resolution.
    """
    topic_clean = topic.lower().strip()
    
    return [
        {
            "segment": "bg_1",
            "prompt": (
                f"Extreme close-up Dutch angle, dramatic lighting, "
                f"related to {topic_clean}, high contrast chiaroscuro, "
                f"mysterious atmosphere, scroll-stopping composition, "
                f"cinematic 8k quality, no text"
            ),
            "fallback_keyword": topic_clean.split()[0] if topic_clean else "history",
        },
        {
            "segment": "bg_2",
            "prompt": (
                f"Wide establishing shot, {topic_clean} historical context, "
                f"atmospheric perspective, golden hour lighting, "
                f"documentary style, photorealistic, 8k quality, no text"
            ),
            "fallback_keyword": f"{topic_clean} historical",
        },
        {
            "segment": "bg_3",
            "prompt": (
                f"Medium shot showing conflict or contrast related to {topic_clean}, "
                f"dramatic lighting, tension in composition, "
                f"cinematic style, photorealistic, 8k, no text"
            ),
            "fallback_keyword": f"{topic_clean} dramatic",
        },
        {
            "segment": "bg_4",
            "prompt": (
                f"Resolution image for {topic_clean}, clarity and understanding, "
                f"warm lighting, balanced composition, "
                f"documentary photography style, 8k quality, no text"
            ),
            "fallback_keyword": topic_clean,
        },
    ]


# ============================================================================
# Main Generation Pipeline
# ============================================================================

class ShortVideoGenerator:
    """
    Unified short video generator.
    
    Handles the complete pipeline:
    1. Script optimization
    2. Image generation (with fallbacks) - now with LLM Visual Director
    3. TTS generation
    4. Subtitle creation
    5. Video rendering
    """
    
    def __init__(self):
        self.optimizer = ShortsOptimizer()
        self.image_provider = ImageProviderWithFallback()
        self.tts_provider = GoogleTTSProvider()
        self.visual_director = VisualDirector()
    
    async def generate(self, config: GenerationConfig) -> GenerationResult:
        """
        Generate a complete short video.
        
        Args:
            config: Generation configuration
        
        Returns:
            GenerationResult
        """
        result = GenerationResult()
        
        try:
            # Phase 1: Setup output folder
            console.print(f"\n[bold cyan]Phase 1: Setup[/bold cyan]")
            
            # Get series info
            series_info = self.optimizer.get_series_info(
                topic=config.topic,
                script=config.script or "",
                series_override=config.series_override,
                increment=True,
            )
            result.series_info = series_info
            
            # Create folder with standardized name
            folder_name = generate_folder_name(series_info.series_name, config.topic)
            output_folder = config.output_dir / folder_name
            output_folder.mkdir(parents=True, exist_ok=True)
            (output_folder / "backgrounds").mkdir(exist_ok=True)
            
            result.output_folder = output_folder
            console.print(f"  Output: {output_folder}")
            console.print(f"  Series: {series_info.display_text}")
            
            # Phase 2: Script optimization
            console.print(f"\n[bold cyan]Phase 2: Script Optimization[/bold cyan]")
            
            if config.script:
                script = config.script
            else:
                # Use a placeholder - in production, this would call LLM
                script = f"This is a placeholder script about {config.topic}. Replace with actual content."
                console.print(f"  [yellow]Warning: No script provided, using placeholder[/yellow]")
            
            # Validate script length for target duration (45-70s)
            validation = validate_script_length(script)
            console.print(f"  Words: {validation.word_count} (target: {validation.target_word_range[0]}-{validation.target_word_range[1]})")
            console.print(f"  Est. duration: {validation.estimated_duration:.1f}s")
            if not validation.is_valid:
                console.print(f"  [yellow]{validation.recommendation}[/yellow]")
            
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
            
            # Phase 3: Image generation
            console.print(f"\n[bold cyan]Phase 3: Image Generation[/bold cyan]")
            
            # Calculate optimal image count if not specified
            if config.num_images <= 0:
                num_images = calculate_optimal_images(validation.estimated_duration)
                console.print(f"  Auto images: {num_images} (based on {validation.estimated_duration:.1f}s duration)")
            else:
                num_images = config.num_images
            
            # Determine prompts source
            if config.image_prompts:
                # User provided custom prompts
                prompts = config.image_prompts
                console.print("  [dim]Using custom prompts[/dim]")
            elif config.auto_prompts and optimized.optimized_script:
                # Use LLM Visual Director for script-aware prompts
                console.print("  [dim]Using LLM Visual Director...[/dim]")
                try:
                    prompts = await self.visual_director.generate_prompts(
                        script=optimized.optimized_script,
                        topic=config.topic,
                        series_name=series_info.series_name,
                        num_images=num_images,
                    )
                    prompts = [p.to_dict() for p in prompts]
                    console.print(f"  [green]Generated {len(prompts)} scene-aware prompts[/green]")
                    
                    # Save visual briefs for debugging
                    briefs_path = output_folder / "visual_briefs.json"
                    with open(briefs_path, "w", encoding="utf-8") as f:
                        json.dump(prompts, f, indent=2, ensure_ascii=False)
                        
                except Exception as e:
                    logger.warning(f"Visual Director failed, using defaults: {e}")
                    console.print(f"  [yellow]Visual Director failed, using defaults[/yellow]")
                    prompts = generate_default_prompts(config.topic)
            else:
                # Fallback to default prompts
                prompts = generate_default_prompts(config.topic)
                console.print("  [dim]Using default prompts[/dim]")
            
            bg_dir = output_folder / "backgrounds"
            
            images_generated = 0
            for idx, p in enumerate(prompts[:num_images]):
                output_path = bg_dir / f"{p['segment']}.jpg"
                console.print(f"  [{idx+1}/{num_images}] {p['segment']}...")
                
                img_result = await self.image_provider.generate_image(
                    prompt=p["prompt"],
                    output_path=output_path,
                    fallback_keyword=p.get("fallback_keyword", config.topic),
                    aspect_ratio="9:16",
                    quality=config.image_quality,  # Use configured quality
                )
                
                if img_result.success:
                    console.print(f"       [green]OK[/green] ({img_result.source})")
                    images_generated += 1
                else:
                    console.print(f"       [red]FAILED[/red]: {img_result.error[:50] if img_result.error else 'Unknown'}")
            
            if images_generated == 0:
                result.error = "No images generated"
                return result
            
            console.print(f"  Total: {images_generated}/{num_images}")
            
            # Phase 4: TTS generation
            console.print(f"\n[bold cyan]Phase 4: TTS Generation[/bold cyan]")
            
            audio_path = output_folder / "voiceover.mp3"
            await self.tts_provider.synthesize(
                text=optimized.optimized_script,
                output_path=audio_path,
                voice=config.voice,
                rate=config.tts_rate,
            )
            if config.tts_rate != "+0%":
                console.print(f"  Rate: {config.tts_rate}")
            
            duration = get_audio_duration(audio_path)
            result.duration = duration
            console.print(f"  Audio: {audio_path}")
            console.print(f"  Duration: {duration:.1f}s")
            
            # Phase 5: Create ASS subtitles with series marker
            console.print(f"\n[bold cyan]Phase 5: Subtitles[/bold cyan]")
            
            srt_path = output_folder / "captions.srt"
            ass_path = output_folder / "captions.ass"
            
            self._create_ass_with_series_marker(
                srt_path=srt_path,
                ass_path=ass_path,
                series_text=series_info.display_text,
                duration=duration,
            )
            console.print(f"  ASS: {ass_path}")
            
            # Phase 6: Render video
            console.print(f"\n[bold cyan]Phase 6: Video Render[/bold cyan]")
            
            images = sorted(bg_dir.glob("bg_*.jpg"))
            slideshow_path = output_folder / "background.mp4"
            final_path = output_folder / "final.mp4"
            
            # Create slideshow with effects
            effects_info = []
            if config.use_ken_burns:
                effects_info.append("Ken Burns")
            if config.use_crossfade:
                effects_info.append("crossfade")
            effects_str = " + ".join(effects_info) if effects_info else "basic"
            console.print(f"  Effects: {effects_str}")
            
            self._create_slideshow(
                images, slideshow_path, duration,
                use_ken_burns=config.use_ken_burns,
                use_crossfade=config.use_crossfade,
            )
            console.print(f"  Slideshow: {slideshow_path}")
            
            # Final render with subtitles
            self._render_final(slideshow_path, audio_path, ass_path, final_path)
            console.print(f"  Final: {final_path}")
            
            result.success = True
            result.final_video = final_path
            
            # Save metadata
            metadata = {
                "topic": config.topic,
                "series": series_info.to_dict(),
                "duration": duration,
                "images_generated": images_generated,
                "voice": config.voice,
            }
            with open(output_folder / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Save series info
            with open(output_folder / "series_info.json", "w", encoding="utf-8") as f:
                json.dump(series_info.to_dict(), f, indent=2)
            
            console.print(f"\n[bold green]Complete![/bold green]")
            console.print(f"Output: {final_path}")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
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
    
    def _create_slideshow(
        self,
        images: list[Path],
        output_path: Path,
        duration: float,
        use_ken_burns: bool = True,
        use_crossfade: bool = True,
        crossfade_duration: float = 0.8,
    ) -> None:
        """
        Create slideshow video from images with Ken Burns effect and crossfade.
        
        Args:
            images: List of image paths
            output_path: Output video path
            duration: Total duration in seconds
            use_ken_burns: Apply slow zoom/pan effect
            use_crossfade: Apply crossfade between images
            crossfade_duration: Duration of crossfade in seconds
        """
        num_images = len(images)
        if num_images == 0:
            raise ValueError("No images to create slideshow")
        
        # Calculate time per image accounting for crossfade overlap
        if use_crossfade and num_images > 1:
            total_crossfade_time = crossfade_duration * (num_images - 1)
            time_per_image = (duration + total_crossfade_time) / num_images
        else:
            time_per_image = duration / num_images
        
        # Build filter complex with Ken Burns and crossfade
        filter_parts = []
        
        # Ken Burns parameters - alternating zoom directions for variety
        kb_configs = [
            "z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",  # Zoom in center
            "z='if(lte(zoom,1.0),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",  # Zoom out
            "z='min(zoom+0.0006,1.12)':x='0':y='ih/2-(ih/zoom/2)'",  # Zoom left to center
            "z='min(zoom+0.0006,1.12)':x='iw-(iw/zoom)':y='ih/2-(ih/zoom/2)'",  # Zoom right to center
        ]
        
        frames_per_image = int(time_per_image * 30)  # 30fps
        
        for i, img in enumerate(images):
            if use_ken_burns:
                kb = kb_configs[i % len(kb_configs)]
                filter_parts.append(
                    f"[{i}:v]scale=1200:2133:force_original_aspect_ratio=increase,"  # Slightly larger for zoom
                    f"zoompan={kb}:d={frames_per_image}:s=1080x1920:fps=30,"
                    f"setsar=1,format=yuv420p[v{i}];"
                )
            else:
                filter_parts.append(
                    f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}];"
                )
        
        # Build crossfade chain or simple concat
        if use_crossfade and num_images > 1:
            # Progressive crossfade chain
            # [v0][v1]xfade...[xf0]; [xf0][v2]xfade...[xf1]; etc
            current_label = "v0"
            for i in range(1, num_images):
                offset = time_per_image * i - crossfade_duration * i
                next_label = f"xf{i-1}" if i < num_images - 1 else "outv"
                filter_parts.append(
                    f"[{current_label}][v{i}]xfade=transition=fade:duration={crossfade_duration}:offset={offset:.2f}[{next_label}];"
                )
                current_label = next_label
            # Remove trailing semicolon
            filter_parts[-1] = filter_parts[-1].rstrip(';')
        else:
            concat_inputs = ''.join(f'[v{i}]' for i in range(num_images))
            filter_parts.append(f"{concat_inputs}concat=n={num_images}:v=1:a=0[outv]")
        
        filter_complex = ''.join(filter_parts)
        
        input_args = []
        for img in images:
            input_args.extend(['-loop', '1', '-t', str(time_per_image + 1), '-i', str(img)])
        
        cmd = [
            'ffmpeg', '-y',
            *input_args,
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-t', str(duration),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg slideshow failed: {result.stderr[:200]}")
    
    def _render_final(
        self,
        slideshow_path: Path,
        audio_path: Path,
        ass_path: Path,
        output_path: Path,
    ) -> None:
        """Render final video with audio and subtitles."""
        ass_path_str = str(ass_path).replace('\\', '/')
        
        # Force yuv420p for maximum compatibility (fixes High 4:4:4 issue)
        cmd = [
            'ffmpeg', '-y',
            '-i', str(slideshow_path),
            '-i', str(audio_path),
            '-vf', f"ass='{ass_path_str}',format=yuv420p",
            '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0',
            '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed: {result.stderr[:200]}")


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
    rate: str = typer.Option(
        "+0%", "--rate", "-r",
        help="TTS speech rate (-10% slower, +10% faster)",
    ),
    num_images: int = typer.Option(
        0, "--images", "-n",
        help="Number of images (0=auto based on script length)",
    ),
    auto_prompts: bool = typer.Option(
        True, "--auto-prompts/--no-auto-prompts",
        help="Use LLM Visual Director for script-aware image prompts",
    ),
    ken_burns: bool = typer.Option(
        True, "--ken-burns/--no-ken-burns",
        help="Apply Ken Burns zoom/pan effect",
    ),
    crossfade: bool = typer.Option(
        True, "--crossfade/--no-crossfade",
        help="Apply crossfade transitions between images",
    ),
    quality: str = typer.Option(
        "high", "--quality", "-q",
        help="Image quality: 'high' (Imagen 3), 'standard' (Imagen @006), or 'fallback' (stock photos)",
    ),
) -> None:
    """
    Generate a complete short video from a topic.
    
    Examples:
        python -m pipeline.generate_short "Napoleon height myth"
        python -m pipeline.generate_short "Aztec civilization" --series "Lost Civs"
        python -m pipeline.generate_short "Hades" -f my_script.txt
        python -m pipeline.generate_short "Jordan flu game" --auto-prompts
        python -m pipeline.generate_short "T-Rex myth" --rate "-5%" --ken-burns
        python -m pipeline.generate_short "Da Vinci" --quality high  # Use Imagen 3
    """
    console.print("[bold cyan]=== Short Video Generator ===[/bold cyan]\n")
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
    
    if auto_prompts:
        console.print(f"Visual Director: [green]enabled[/green]")
    else:
        console.print(f"Visual Director: [dim]disabled[/dim]")
    
    console.print(f"Image Quality: [cyan]{quality}[/cyan]")
    
    # Create config
    config = GenerationConfig(
        topic=topic,
        script=script_text,
        series_override=series,
        output_dir=output_dir,
        voice=voice,
        num_images=num_images,
        auto_prompts=auto_prompts,
        image_quality=quality,
        tts_rate=rate,
        use_ken_burns=ken_burns,
        use_crossfade=crossfade,
    )
    
    # Run generation
    generator = ShortVideoGenerator()
    result = asyncio.run(generator.generate(config))
    
    if result.success:
        console.print(f"\n[green]Success![/green]")
        console.print(f"Video: {result.final_video}")
        console.print(f"Duration: {result.duration:.1f}s")
        if result.series_info:
            console.print(f"Series: {result.series_info.display_text}")
    else:
        console.print(f"\n[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


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


@app.command("preview")
def preview_images(
    folder: Path = typer.Argument(..., help="Path to shorts output folder"),
) -> None:
    """
    Preview generated images and optionally regenerate.
    
    Shows all generated images and allows regenerating individual ones.
    
    Example:
        python -m pipeline.generate_short preview outputs/shorts/art-myths_davinci
    """
    console.print("[bold cyan]=== Image Preview ===[/bold cyan]\n")
    
    bg_dir = folder / "backgrounds"
    if not bg_dir.exists():
        console.print(f"[red]No backgrounds folder found in {folder}[/red]")
        raise typer.Exit(1)
    
    images = sorted(bg_dir.glob("bg_*.jpg"))
    if not images:
        console.print(f"[red]No images found in {bg_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(images)} images in {bg_dir}\n")
    
    for img in images:
        size_kb = img.stat().st_size / 1024
        console.print(f"  {img.name}: {size_kb:.1f} KB")
    
    # Check for visual briefs
    briefs_path = folder / "visual_briefs.json"
    if briefs_path.exists():
        console.print(f"\nVisual briefs: {briefs_path}")
    
    console.print("\n[dim]Use 'regenerate' command to regenerate specific images[/dim]")
    console.print(f"[dim]Example: python -m pipeline.generate_short regenerate {folder} 1[/dim]")


@app.command("regenerate")
def regenerate_image(
    folder: Path = typer.Argument(..., help="Path to shorts output folder"),
    image_num: int = typer.Argument(..., help="Image number to regenerate (1-6)"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p",
        help="Custom prompt (uses existing if not specified)",
    ),
    quality: str = typer.Option(
        "high", "--quality", "-q",
        help="Image quality: 'high' (Imagen 3), 'standard', or 'fallback'",
    ),
) -> None:
    """
    Regenerate a single image in an existing output folder.
    
    Example:
        python -m pipeline.generate_short regenerate outputs/shorts/art-myths_davinci 3
        python -m pipeline.generate_short regenerate outputs/shorts/art-myths_davinci 2 -p "New prompt here"
    """
    console.print(f"[bold cyan]=== Regenerate Image {image_num} ===[/bold cyan]\n")
    
    bg_dir = folder / "backgrounds"
    if not bg_dir.exists():
        console.print(f"[red]No backgrounds folder found in {folder}[/red]")
        raise typer.Exit(1)
    
    output_path = bg_dir / f"bg_{image_num}.jpg"
    
    # Get prompt
    generation_prompt = prompt
    fallback_keyword = ""
    
    if not generation_prompt:
        # Try to load from visual briefs
        briefs_path = folder / "visual_briefs.json"
        if briefs_path.exists():
            import json
            briefs = json.loads(briefs_path.read_text(encoding="utf-8"))
            for brief in briefs:
                if brief.get("segment") == f"bg_{image_num}":
                    generation_prompt = brief.get("prompt", "")
                    fallback_keyword = brief.get("fallback_keyword", "")
                    break
    
    if not generation_prompt:
        console.print("[red]No prompt found. Use --prompt to specify one.[/red]")
        raise typer.Exit(1)
    
    console.print(f"Prompt: {generation_prompt[:100]}...")
    console.print(f"Quality: {quality}")
    console.print(f"Output: {output_path}")
    
    # Backup existing image
    if output_path.exists():
        backup_path = bg_dir / f"bg_{image_num}_backup.jpg"
        import shutil
        shutil.copy(output_path, backup_path)
        console.print(f"[dim]Backed up existing image to {backup_path.name}[/dim]")
    
    # Generate new image
    console.print("\nGenerating...")
    
    async def do_regenerate():
        provider = ImageProviderWithFallback()
        result = await provider.generate_image(
            prompt=generation_prompt,
            output_path=output_path,
            fallback_keyword=fallback_keyword or f"image {image_num}",
            quality=quality,
        )
        return result
    
    result = asyncio.run(do_regenerate())
    
    if result.success:
        size_kb = output_path.stat().st_size / 1024
        console.print(f"\n[green]Success![/green]")
        console.print(f"  Source: {result.source}")
        console.print(f"  Size: {size_kb:.1f} KB")
        console.print(f"  Latency: {result.latency_ms}ms")
    else:
        console.print(f"\n[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@app.command("render")
def render_video(
    folder: Path = typer.Argument(..., help="Path to shorts output folder"),
) -> None:
    """
    Render final video from existing images and audio.
    
    Use this after regenerating images to create a new video.
    
    Example:
        python -m pipeline.generate_short render outputs/shorts/art-myths_davinci
    """
    import subprocess
    
    console.print("[bold cyan]=== Render Video ===[/bold cyan]\n")
    
    # Check required files
    bg_dir = folder / "backgrounds"
    audio_path = folder / "voiceover.mp3"
    ass_path = folder / "captions.ass"
    
    if not bg_dir.exists():
        console.print(f"[red]No backgrounds folder found[/red]")
        raise typer.Exit(1)
    
    if not audio_path.exists():
        console.print(f"[red]No voiceover.mp3 found[/red]")
        raise typer.Exit(1)
    
    if not ass_path.exists():
        console.print(f"[red]No captions.ass found[/red]")
        raise typer.Exit(1)
    
    images = sorted(bg_dir.glob("bg_*.jpg"))
    if not images:
        console.print(f"[red]No images found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Images: {len(images)}")
    console.print(f"Audio: {audio_path}")
    console.print(f"Captions: {ass_path}")
    
    # Get audio duration
    duration = get_audio_duration(audio_path)
    console.print(f"Duration: {duration:.1f}s")
    
    # Create slideshow
    slideshow_path = folder / "background.mp4"
    final_path = folder / "final.mp4"
    
    console.print("\nCreating slideshow...")
    
    generator = ShortVideoGenerator()
    generator._create_slideshow(
        images=images,
        output_path=slideshow_path,
        duration=duration,
        use_ken_burns=True,
        use_crossfade=True,
    )
    
    console.print("Rendering final video...")
    
    generator._render_final(
        slideshow_path=slideshow_path,
        audio_path=audio_path,
        ass_path=ass_path,
        output_path=final_path,
    )
    
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        console.print(f"\n[green]Success![/green]")
        console.print(f"Output: {final_path}")
        console.print(f"Size: {size_mb:.1f} MB")
    else:
        console.print(f"\n[red]Failed to create final video[/red]")
        raise typer.Exit(1)


@app.command("list-styles")
def list_styles() -> None:
    """List available visual styles for image generation."""
    from pipeline.storyboard_templates import STYLE_TEMPLATES
    
    console.print("[bold cyan]=== Available Visual Styles ===[/bold cyan]\n")
    
    for style_id, style in STYLE_TEMPLATES.items():
        console.print(f"[bold]{style_id}[/bold]: {style.name}")
        console.print(f"  {style.description}")
        console.print(f"  Prefix: {style.style_prefix[:50]}...")
        console.print()


@app.command("list-arcs")
def list_arcs() -> None:
    """List available narrative arcs for storyboard generation."""
    from pipeline.storyboard_templates import NARRATIVE_ARCS
    
    console.print("[bold cyan]=== Available Narrative Arcs ===[/bold cyan]\n")
    
    for arc_id, scenes in NARRATIVE_ARCS.items():
        console.print(f"[bold]{arc_id}[/bold]: {len(scenes)} scenes")
        for i, scene in enumerate(scenes, 1):
            console.print(f"  {i}. {scene.name} - {scene.description[:40]}...")
        console.print()


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
