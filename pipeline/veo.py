"""
Myth Museum - Google Veo Video Generation Provider

Generate B-roll video clips using Google Veo via Gemini API or Vertex AI.

Supports:
- Gemini API (default, uses GEMINI_API_KEY)
- Vertex AI (fallback, uses service account)

Workflow: submit prompt -> poll operation -> download video

Environment Variables:
    VEO_PROVIDER: gemini | vertex (default: gemini)
    GEMINI_API_KEY: API key for Gemini (required for gemini provider)
    VEO_MODEL: Model override (auto-detect by default)
    VERTEX_PROJECT_ID: GCP project (for vertex provider)
    GOOGLE_APPLICATION_CREDENTIALS: Service account JSON (for vertex provider)
"""

import asyncio
import csv
import json
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="veo",
    help="Google Veo video generation",
    add_completion=False,
)


# ============================================================================
# Data Classes
# ============================================================================

class VeoProvider(str, Enum):
    """Veo provider options."""
    GEMINI = "gemini"
    VERTEX = "vertex"


@dataclass
class VeoResult:
    """Result of video generation."""
    success: bool = False
    local_path: Optional[Path] = None
    prompt: str = ""
    duration_sec: float = 0.0
    error: Optional[str] = None
    latency_ms: int = 0
    provider: str = ""
    model: str = ""
    operation_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "local_path": str(self.local_path) if self.local_path else None,
            "prompt": self.prompt,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model": self.model,
            "operation_id": self.operation_id,
        }


@dataclass
class ShotlistEntry:
    """Single entry from shotlist.csv."""
    shot_id: int
    segment: str
    start_time: float
    end_time: float
    visual_description: str
    notes: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_veo_prompt(self, style: str = "cinematic documentary") -> str:
        """Convert to Veo-optimized prompt."""
        base = self.visual_description.strip()
        
        # Add Veo-specific modifiers
        modifiers = [
            "vertical video 9:16 aspect ratio",
            "smooth camera movement",
            "no text overlays",
            "no logos or watermarks",
            "no subtitles",
            style,
            "high quality 1080p",
        ]
        
        return f"{base}. {', '.join(modifiers)}."


# ============================================================================
# Gemini API Provider
# ============================================================================

class GeminiVeoProvider:
    """
    Veo video generation via Gemini API (Google AI Studio).
    
    Uses the google-generativeai SDK.
    """
    
    # Model priority for auto-detection
    MODEL_PRIORITY = [
        "veo-3.1",
        "veo-3",
        "veo-2",
    ]
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_override = os.getenv("VEO_MODEL", "")
        self._client = None
        self._available_model = None
    
    def is_available(self) -> bool:
        """Check if Gemini API is configured."""
        return bool(self.api_key)
    
    def _ensure_client(self):
        """Initialize Gemini client."""
        if self._client is not None:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
            logger.info("Gemini API initialized for Veo")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    def _detect_model(self) -> str:
        """Auto-detect available Veo model."""
        if self._available_model:
            return self._available_model
        
        if self.model_override:
            self._available_model = self.model_override
            return self._available_model
        
        self._ensure_client()
        
        # Try models in priority order
        for model_name in self.MODEL_PRIORITY:
            try:
                # Check if model exists by trying to get info
                models = list(self._client.list_models())
                model_names = [m.name for m in models]
                
                # Check for model support
                if any(model_name in name for name in model_names):
                    self._available_model = model_name
                    logger.info(f"Detected Veo model: {model_name}")
                    return model_name
            except Exception:
                continue
        
        # Default to veo-2 if auto-detect fails
        self._available_model = "veo-2"
        logger.warning(f"Auto-detect failed, using default: {self._available_model}")
        return self._available_model
    
    async def generate_video(
        self,
        prompt: str,
        output_path: Path,
        duration_sec: int = 8,
        resolution: str = "1080p",
        poll_interval: float = 5.0,
        max_wait: float = 300.0,
    ) -> VeoResult:
        """
        Generate video using Veo via Gemini API.
        
        Args:
            prompt: Video generation prompt
            output_path: Where to save the video
            duration_sec: Target duration (5-8 seconds)
            resolution: Resolution (1080p for Shorts)
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time in seconds
        
        Returns:
            VeoResult
        """
        result = VeoResult(
            success=False,
            prompt=prompt,
            provider="gemini",
        )
        
        if not self.is_available():
            result.error = "GEMINI_API_KEY not set"
            return result
        
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._ensure_client()
            model_name = self._detect_model()
            result.model = model_name
            
            # Create video generation request
            # Note: The exact API may vary based on model availability
            # This is the expected structure for Veo via Gemini API
            
            import google.generativeai as genai
            
            # Try to use the video generation capability
            # Veo models support generate_content with video output
            model = genai.GenerativeModel(model_name)
            
            # Build the generation config for video
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 8192,
            }
            
            # For video generation, we use a specific prompt format
            video_prompt = f"""Generate a {duration_sec} second video clip:

{prompt}

Requirements:
- Vertical format (9:16 aspect ratio) for mobile viewing
- Resolution: {resolution}
- No text, logos, or watermarks
- Smooth, professional camera movement
- Cinematic documentary style
"""
            
            logger.info(f"Submitting Veo generation request: {prompt[:100]}...")
            
            # Submit the generation request
            # The response handling depends on the Veo API structure
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    video_prompt,
                    generation_config=generation_config,
                )
            )
            
            # Check for video in response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Look for video data in the response
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        # Handle video blob
                        if hasattr(part, 'inline_data') and part.inline_data:
                            video_data = part.inline_data.data
                            mime_type = part.inline_data.mime_type
                            
                            if 'video' in mime_type:
                                output_path.write_bytes(video_data)
                                result.success = True
                                result.local_path = output_path
                                result.duration_sec = duration_sec
                                result.latency_ms = int((time.time() - start_time) * 1000)
                                logger.info(f"Generated video: {output_path} ({result.latency_ms}ms)")
                                return result
            
            # If direct video generation isn't available, try operation-based approach
            # This is the async pattern where we get an operation ID and poll
            if hasattr(response, 'operation'):
                operation_id = response.operation.name
                result.operation_id = operation_id
                logger.info(f"Video generation started, operation: {operation_id}")
                
                # Poll for completion
                elapsed = 0.0
                while elapsed < max_wait:
                    await asyncio.sleep(poll_interval)
                    elapsed = time.time() - start_time
                    
                    # Check operation status
                    status = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._client.get_operation(operation_id)
                    )
                    
                    if status.done:
                        if hasattr(status, 'result') and status.result:
                            # Download the video
                            video_url = status.result.get('video_uri')
                            if video_url:
                                await self._download_video(video_url, output_path)
                                result.success = True
                                result.local_path = output_path
                                result.duration_sec = duration_sec
                                result.latency_ms = int((time.time() - start_time) * 1000)
                                return result
                        
                        if hasattr(status, 'error') and status.error:
                            result.error = status.error.message
                            return result
                    
                    logger.debug(f"Polling... {elapsed:.0f}s / {max_wait:.0f}s")
                
                result.error = f"Timeout after {max_wait}s"
                return result
            
            # Fallback: If no video in response, check for text that might indicate an error
            if hasattr(response, 'text') and response.text:
                result.error = f"Unexpected response: {response.text[:200]}"
            else:
                result.error = "No video generated - model may not support video output"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Veo generation error: {error_msg}")
            result.error = error_msg
        
        return result
    
    async def _download_video(self, url: str, output_path: Path) -> None:
        """Download video from URL."""
        ctx = ssl.create_default_context()
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(request, context=ctx, timeout=60) as response:
            output_path.write_bytes(response.read())


# ============================================================================
# Vertex AI Provider
# ============================================================================

class VertexVeoProvider:
    """
    Veo video generation via Vertex AI.
    
    Uses service account authentication.
    """
    
    def __init__(self):
        self.project_id = os.getenv("VERTEX_PROJECT_ID", "")
        self.location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.model_override = os.getenv("VEO_MODEL", "")
        self._initialized = False
    
    def is_available(self) -> bool:
        """Check if Vertex AI is configured."""
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        return bool(self.project_id and creds_path)
    
    def _ensure_initialized(self):
        """Initialize Vertex AI."""
        if self._initialized:
            return
        
        try:
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            self._initialized = True
            logger.info(f"Vertex AI initialized: project={self.project_id}, location={self.location}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    async def generate_video(
        self,
        prompt: str,
        output_path: Path,
        duration_sec: int = 8,
        resolution: str = "1080p",
        poll_interval: float = 5.0,
        max_wait: float = 300.0,
    ) -> VeoResult:
        """
        Generate video using Veo via Vertex AI.
        
        Args:
            prompt: Video generation prompt
            output_path: Where to save the video
            duration_sec: Target duration (5-8 seconds)
            resolution: Resolution
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time
        
        Returns:
            VeoResult
        """
        result = VeoResult(
            success=False,
            prompt=prompt,
            provider="vertex",
        )
        
        if not self.is_available():
            result.error = "Vertex AI not configured (missing VERTEX_PROJECT_ID or credentials)"
            return result
        
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._ensure_initialized()
            
            # Import Vertex AI video generation
            from vertexai.preview.vision_models import VideoGenerationModel
            
            model_name = self.model_override or "veo-001"
            result.model = model_name
            
            model = VideoGenerationModel.from_pretrained(model_name)
            
            # Build video prompt with modifiers
            full_prompt = f"{prompt}. Vertical 9:16 aspect ratio, {resolution}, no text or logos, cinematic documentary style."
            
            logger.info(f"Submitting Vertex Veo request: {prompt[:100]}...")
            
            # Generate video
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_videos(
                    prompt=full_prompt,
                    number_of_videos=1,
                    duration_seconds=duration_sec,
                )
            )
            
            # Handle response
            if hasattr(response, 'videos') and response.videos:
                video = response.videos[0]
                video.save(str(output_path))
                
                result.success = True
                result.local_path = output_path
                result.duration_sec = duration_sec
                result.latency_ms = int((time.time() - start_time) * 1000)
                logger.info(f"Generated video: {output_path} ({result.latency_ms}ms)")
                return result
            
            result.error = "No video in response"
            
        except ImportError:
            result.error = "Vertex AI video generation not available in SDK"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Vertex Veo error: {error_msg}")
            result.error = error_msg
        
        return result


# ============================================================================
# Unified Veo Provider
# ============================================================================

class UnifiedVeoProvider:
    """
    Unified Veo provider with automatic fallback.
    
    Tries providers in order:
    1. Gemini API (if GEMINI_API_KEY set)
    2. Vertex AI (if credentials configured)
    """
    
    def __init__(self, preferred_provider: VeoProvider = VeoProvider.GEMINI):
        self.preferred = preferred_provider
        self.gemini = GeminiVeoProvider()
        self.vertex = VertexVeoProvider()
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        available = []
        if self.gemini.is_available():
            available.append("gemini")
        if self.vertex.is_available():
            available.append("vertex")
        return available
    
    async def generate_broll(
        self,
        prompt: str,
        output_path: Path,
        duration_sec: int = 8,
        resolution: str = "1080p",
    ) -> VeoResult:
        """
        Generate B-roll video clip.
        
        Args:
            prompt: Video generation prompt
            output_path: Where to save the video
            duration_sec: Target duration (5-8 seconds)
            resolution: Resolution (1080p for Shorts)
        
        Returns:
            VeoResult
        """
        # Determine provider order based on preference
        if self.preferred == VeoProvider.GEMINI:
            providers = [
                ("gemini", self.gemini),
                ("vertex", self.vertex),
            ]
        else:
            providers = [
                ("vertex", self.vertex),
                ("gemini", self.gemini),
            ]
        
        # Try each provider
        for name, provider in providers:
            if not provider.is_available():
                logger.debug(f"Provider {name} not available, skipping")
                continue
            
            logger.info(f"Trying Veo provider: {name}")
            result = await provider.generate_video(
                prompt=prompt,
                output_path=output_path,
                duration_sec=duration_sec,
                resolution=resolution,
            )
            
            if result.success:
                return result
            
            logger.warning(f"Provider {name} failed: {result.error}")
        
        # All providers failed
        return VeoResult(
            success=False,
            prompt=prompt,
            error="All Veo providers failed or unavailable",
        )
    
    async def generate_broll_for_folder(
        self,
        folder: Path,
        shotlist_csv: Optional[Path] = None,
    ) -> list[VeoResult]:
        """
        Generate B-roll clips for a shorts folder from shotlist.
        
        Args:
            folder: Output folder path
            shotlist_csv: Path to shotlist.csv (default: folder/shotlist.csv)
        
        Returns:
            List of VeoResult for each shot
        """
        folder = Path(folder)
        if shotlist_csv is None:
            shotlist_csv = folder / "shotlist.csv"
        
        if not shotlist_csv.exists():
            raise FileNotFoundError(f"Shotlist not found: {shotlist_csv}")
        
        # Parse shotlist
        entries = parse_shotlist(shotlist_csv)
        if not entries:
            raise ValueError("Empty shotlist")
        
        # Create broll directory
        broll_dir = folder / "broll"
        broll_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate videos
        results = []
        for idx, entry in enumerate(entries, start=1):
            output_path = broll_dir / f"broll_{idx:02d}.mp4"
            prompt = entry.to_veo_prompt()
            
            # Clamp duration to 5-8 seconds
            duration = max(5, min(8, int(entry.duration)))
            
            console.print(f"  [{idx}/{len(entries)}] {entry.segment}: {entry.visual_description[:50]}...")
            
            result = await self.generate_broll(
                prompt=prompt,
                output_path=output_path,
                duration_sec=duration,
            )
            
            results.append(result)
            
            if result.success:
                console.print(f"       [green]OK[/green] ({result.provider}, {result.latency_ms}ms)")
            else:
                console.print(f"       [red]FAILED[/red]: {result.error[:50] if result.error else 'Unknown'}")
        
        return results


# ============================================================================
# Shotlist Parser
# ============================================================================

def parse_shotlist(csv_path: Path) -> list[ShotlistEntry]:
    """
    Parse shotlist.csv into ShotlistEntry objects.
    
    Expected columns: shot_id, segment, start_time, end_time, visual_description, notes
    
    Args:
        csv_path: Path to shotlist.csv
    
    Returns:
        List of ShotlistEntry
    """
    entries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                entry = ShotlistEntry(
                    shot_id=int(row.get('shot_id', 0)),
                    segment=row.get('segment', ''),
                    start_time=float(row.get('start_time', 0)),
                    end_time=float(row.get('end_time', 0)),
                    visual_description=row.get('visual_description', ''),
                    notes=row.get('notes', ''),
                )
                entries.append(entry)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue
    
    return entries


def write_shotlist(entries: list[ShotlistEntry], csv_path: Path) -> None:
    """
    Write shotlist entries to CSV.
    
    Args:
        entries: List of ShotlistEntry
        csv_path: Output path
    """
    fieldnames = ['shot_id', 'segment', 'start_time', 'end_time', 'visual_description', 'notes']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in entries:
            writer.writerow({
                'shot_id': entry.shot_id,
                'segment': entry.segment,
                'start_time': entry.start_time,
                'end_time': entry.end_time,
                'visual_description': entry.visual_description,
                'notes': entry.notes,
            })


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def generate(
    folder: Path = typer.Argument(..., help="Shorts folder with shotlist.csv"),
    shotlist: Optional[Path] = typer.Option(
        None, "--shotlist", "-s",
        help="Path to shotlist.csv (default: folder/shotlist.csv)",
    ),
    provider: VeoProvider = typer.Option(
        VeoProvider.GEMINI, "--provider", "-p",
        help="Preferred Veo provider",
    ),
) -> None:
    """
    Generate B-roll videos from shotlist.
    
    Reads shotlist.csv and generates video clips for each shot.
    """
    console.print("[bold cyan]=== Veo B-roll Generator ===[/bold cyan]\n")
    
    folder = Path(folder)
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)
    
    shotlist_path = shotlist or folder / "shotlist.csv"
    if not shotlist_path.exists():
        console.print(f"[red]Shotlist not found: {shotlist_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Folder: {folder}")
    console.print(f"Shotlist: {shotlist_path}")
    console.print(f"Provider: {provider.value}")
    
    # Check available providers
    veo = UnifiedVeoProvider(preferred_provider=provider)
    available = veo.get_available_providers()
    
    if not available:
        console.print("\n[red]No Veo providers available![/red]")
        console.print("Configure one of:")
        console.print("  - GEMINI_API_KEY for Gemini API")
        console.print("  - VERTEX_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS for Vertex AI")
        raise typer.Exit(1)
    
    console.print(f"Available providers: {', '.join(available)}")
    
    # Generate B-roll
    console.print("\n[bold]Generating B-roll clips...[/bold]")
    
    results = asyncio.run(veo.generate_broll_for_folder(folder, shotlist_path))
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Summary:[/bold] {success_count}/{len(results)} clips generated")
    
    # Save results
    results_path = folder / "veo_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    console.print(f"Results saved: {results_path}")


@app.command()
def test_prompt(
    prompt: str = typer.Argument(..., help="Video generation prompt"),
    output: Path = typer.Option(
        Path("test_video.mp4"), "--output", "-o",
        help="Output path",
    ),
    duration: int = typer.Option(
        5, "--duration", "-d",
        help="Video duration in seconds",
    ),
    provider: VeoProvider = typer.Option(
        VeoProvider.GEMINI, "--provider", "-p",
        help="Veo provider to use",
    ),
) -> None:
    """
    Test a single video generation prompt.
    """
    console.print("[bold cyan]=== Veo Test ===[/bold cyan]\n")
    console.print(f"Prompt: {prompt}")
    console.print(f"Output: {output}")
    console.print(f"Duration: {duration}s")
    console.print(f"Provider: {provider.value}")
    
    veo = UnifiedVeoProvider(preferred_provider=provider)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating video...", total=None)
        result = asyncio.run(veo.generate_broll(
            prompt=prompt,
            output_path=output,
            duration_sec=duration,
        ))
    
    if result.success:
        console.print(f"\n[green]Success![/green]")
        console.print(f"Output: {result.local_path}")
        console.print(f"Provider: {result.provider}")
        console.print(f"Model: {result.model}")
        console.print(f"Latency: {result.latency_ms}ms")
    else:
        console.print(f"\n[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """
    Check Veo provider status.
    """
    console.print("[bold cyan]=== Veo Provider Status ===[/bold cyan]\n")
    
    gemini = GeminiVeoProvider()
    vertex = VertexVeoProvider()
    
    # Check Gemini
    console.print("[bold]Gemini API:[/bold]")
    if gemini.is_available():
        console.print(f"  Status: [green]Available[/green]")
        console.print(f"  API Key: {gemini.api_key[:10]}...")
    else:
        console.print(f"  Status: [yellow]Not configured[/yellow]")
        console.print(f"  Set GEMINI_API_KEY environment variable")
    
    console.print()
    
    # Check Vertex
    console.print("[bold]Vertex AI:[/bold]")
    if vertex.is_available():
        console.print(f"  Status: [green]Available[/green]")
        console.print(f"  Project: {vertex.project_id}")
        console.print(f"  Location: {vertex.location}")
    else:
        console.print(f"  Status: [yellow]Not configured[/yellow]")
        console.print(f"  Set VERTEX_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS")
    
    console.print()
    
    # Environment variables
    console.print("[bold]Environment:[/bold]")
    console.print(f"  VEO_PROVIDER: {os.getenv('VEO_PROVIDER', '(not set, default: gemini)')}")
    console.print(f"  VEO_MODEL: {os.getenv('VEO_MODEL', '(not set, auto-detect)')}")


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
