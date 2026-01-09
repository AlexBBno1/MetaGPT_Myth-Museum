"""
Myth Museum - Shotlist Generator

Generate shotlist.csv from script using LLM.

The shotlist defines video segments with timing and visual descriptions
that can be used for Veo B-roll generation.

Output format:
    shot_id,segment,start_time,end_time,visual_description,notes
    1,hook,0,5,"Golden man emerging from lake at dawn",Opening hook
"""

import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from core.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="shotlist",
    help="Generate shotlist from script",
    add_completion=False,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ShotlistEntry:
    """Single entry in shotlist."""
    shot_id: int
    segment: str
    start_time: float
    end_time: float
    visual_description: str
    notes: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "segment": self.segment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "visual_description": self.visual_description,
            "notes": self.notes,
        }


# ============================================================================
# Prompt Template
# ============================================================================

SHOTLIST_PROMPT = """You are a visual director for short documentary videos.

Given a script and topic, create a shotlist with 4 video segments.
Each segment should be 5-8 seconds long.

IMPORTANT VIDEO REQUIREMENTS:
- Vertical format (9:16 aspect ratio) for mobile viewing
- NO text overlays, titles, or captions
- NO logos or watermarks
- NO human faces (use silhouettes, hands, objects instead)
- Cinematic documentary style
- Smooth, professional camera movements
- High contrast, visually striking compositions

SEGMENT STRUCTURE:
1. hook (0-5s): Attention-grabbing opening visual
2. context (5-20s): Establish the setting/background
3. contrast (20-40s): Show the conflict/turning point
4. resolution (40-60s): Conclude with impact

For each segment, describe the EXACT visual scene that would work for AI video generation.
Be specific about:
- Camera angle and movement
- Lighting and atmosphere
- Key visual elements
- Color palette

Topic: {topic}

Script:
{script}

Total duration hint: {duration_hint} seconds

Respond in valid JSON format:
{{
  "shots": [
    {{
      "shot_id": 1,
      "segment": "hook",
      "start_time": 0,
      "end_time": 5,
      "visual_description": "Extreme close-up of golden dust particles floating in sunlight, slow motion, shallow depth of field",
      "notes": "Opening hook"
    }},
    ...
  ]
}}

Generate the shotlist now:"""


# ============================================================================
# LLM Integration
# ============================================================================

async def generate_shotlist_with_llm(
    script: str,
    topic: str,
    duration_hint: float = 60.0,
    num_shots: int = 4,
) -> list[ShotlistEntry]:
    """
    Generate shotlist using LLM.
    
    Args:
        script: Video script/voiceover text
        topic: Video topic
        duration_hint: Expected total duration in seconds
        num_shots: Number of shots to generate
    
    Returns:
        List of ShotlistEntry
    """
    # Try to use available LLM
    entries = []
    
    # Try Google AI Studio first (since we have GEMINI_API_KEY for Veo)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        entries = await _generate_with_gemini(script, topic, duration_hint, num_shots)
        if entries:
            return entries
    
    # Try OpenAI
    try:
        from core.llm import LLMClient
        client = LLMClient.from_config()
        
        if client.is_configured():
            entries = await _generate_with_openai(client, script, topic, duration_hint, num_shots)
            if entries:
                return entries
    except Exception as e:
        logger.warning(f"OpenAI LLM failed: {e}")
    
    # Fallback: Generate template-based shotlist
    logger.info("Using template-based shotlist generation")
    return _generate_template_shotlist(script, topic, duration_hint, num_shots)


async def _generate_with_gemini(
    script: str,
    topic: str,
    duration_hint: float,
    num_shots: int,
) -> list[ShotlistEntry]:
    """Generate shotlist using Gemini API."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return []
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = SHOTLIST_PROMPT.format(
            topic=topic,
            script=script,
            duration_hint=int(duration_hint),
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                )
            )
        )
        
        if response.text:
            return _parse_llm_response(response.text)
        
    except Exception as e:
        logger.warning(f"Gemini shotlist generation failed: {e}")
    
    return []


async def _generate_with_openai(
    client,
    script: str,
    topic: str,
    duration_hint: float,
    num_shots: int,
) -> list[ShotlistEntry]:
    """Generate shotlist using OpenAI API."""
    try:
        prompt = SHOTLIST_PROMPT.format(
            topic=topic,
            script=script,
            duration_hint=int(duration_hint),
        )
        
        response = await client.chat([
            {"role": "system", "content": "You are a visual director for documentary shorts. Always respond with valid JSON."},
            {"role": "user", "content": prompt},
        ])
        
        if response:
            return _parse_llm_response(response)
        
    except Exception as e:
        logger.warning(f"OpenAI shotlist generation failed: {e}")
    
    return []


def _parse_llm_response(response: str) -> list[ShotlistEntry]:
    """Parse LLM response into ShotlistEntry objects."""
    entries = []
    
    try:
        # Clean up response
        text = response.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```json?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        # Parse JSON
        data = json.loads(text)
        
        shots = data.get("shots", [])
        if not shots and isinstance(data, list):
            shots = data
        
        for shot in shots:
            entry = ShotlistEntry(
                shot_id=int(shot.get("shot_id", 0)),
                segment=shot.get("segment", ""),
                start_time=float(shot.get("start_time", 0)),
                end_time=float(shot.get("end_time", 0)),
                visual_description=shot.get("visual_description", ""),
                notes=shot.get("notes", ""),
            )
            entries.append(entry)
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
    
    return entries


def _generate_template_shotlist(
    script: str,
    topic: str,
    duration_hint: float,
    num_shots: int,
) -> list[ShotlistEntry]:
    """
    Generate template-based shotlist when LLM is unavailable.
    
    Uses topic keywords to create relevant visual descriptions.
    """
    # Extract keywords from topic
    topic_lower = topic.lower()
    keywords = [w for w in topic_lower.split() if len(w) > 3]
    main_keyword = keywords[0] if keywords else "subject"
    
    # Calculate segment durations
    segment_duration = duration_hint / num_shots
    
    # Template visual descriptions based on topic type
    visual_templates = _get_visual_templates(topic_lower, main_keyword)
    
    entries = []
    segments = ["hook", "context", "contrast", "resolution"]
    
    for i in range(num_shots):
        segment = segments[i] if i < len(segments) else f"segment_{i+1}"
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        visual_desc = visual_templates.get(segment, f"Visual related to {main_keyword}")
        
        entry = ShotlistEntry(
            shot_id=i + 1,
            segment=segment,
            start_time=start_time,
            end_time=end_time,
            visual_description=visual_desc,
            notes=f"Segment {i + 1}",
        )
        entries.append(entry)
    
    return entries


def _get_visual_templates(topic_lower: str, main_keyword: str) -> dict[str, str]:
    """Get visual templates based on topic type."""
    
    # Historical topics
    if any(w in topic_lower for w in ['history', 'ancient', 'civilization', 'empire', 'war']):
        return {
            "hook": f"Dramatic extreme close-up of ancient artifact or ruins related to {main_keyword}, dust particles in golden light, cinematic shallow depth of field",
            "context": f"Wide aerial shot of historical landscape, ancient architecture silhouettes at sunset, atmospheric haze, documentary style",
            "contrast": f"Side-by-side visual metaphor showing contrast - light and shadow, old and new, truth and myth, dramatic lighting",
            "resolution": f"Slow pullback from detailed artifact to reveal full context, warm lighting, sense of discovery and understanding",
        }
    
    # Mythology topics
    if any(w in topic_lower for w in ['myth', 'god', 'legend', 'greek', 'roman', 'norse']):
        return {
            "hook": f"Dramatic statue silhouette against stormy sky, lightning flashes, extreme low angle, epic scale",
            "context": f"Ancient temple columns in golden hour light, ethereal mist, slow dolly movement through ruins",
            "contrast": f"Split composition showing mythological imagery merging with historical evidence, dramatic shadows",
            "resolution": f"Sunrise over ancient landscape, hope and clarity, warm colors breaking through clouds",
        }
    
    # Science topics
    if any(w in topic_lower for w in ['science', 'physics', 'biology', 'space', 'nature']):
        return {
            "hook": f"Macro shot of natural phenomenon related to {main_keyword}, vivid colors, abstract patterns",
            "context": f"Time-lapse of natural process, stars moving, plants growing, waves crashing, cyclical patterns",
            "contrast": f"Split screen comparing common misconception vs scientific reality, clean visual demonstration",
            "resolution": f"Elegant scientific visualization, mathematical patterns in nature, satisfying reveal",
        }
    
    # Art/culture topics
    if any(w in topic_lower for w in ['art', 'paint', 'music', 'culture', 'artist']):
        return {
            "hook": f"Extreme close-up of brushstrokes or artistic detail, texture and color, intimate perspective",
            "context": f"Artist's workspace or studio environment, tools of the craft, atmospheric lighting",
            "contrast": f"Juxtaposition of the art and the artist's life, visual metaphor for creative struggle",
            "resolution": f"The finished work in its full glory, slow reveal, appreciative perspective",
        }
    
    # Default templates
    return {
        "hook": f"Dramatic extreme close-up related to {main_keyword}, high contrast lighting, shallow depth of field, scroll-stopping composition",
        "context": f"Wide establishing shot showing the setting and context of {main_keyword}, atmospheric lighting, documentary style",
        "contrast": f"Visual representation of conflict or contrast related to {main_keyword}, dramatic lighting, tension in composition",
        "resolution": f"Resolution visual for {main_keyword}, clarity and understanding, warm lighting, satisfying conclusion",
    }


# ============================================================================
# File I/O
# ============================================================================

def write_shotlist_csv(entries: list[ShotlistEntry], output_path: Path) -> None:
    """
    Write shotlist to CSV file.
    
    Args:
        entries: List of ShotlistEntry
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['shot_id', 'segment', 'start_time', 'end_time', 'visual_description', 'notes']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
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
    
    logger.info(f"Wrote shotlist: {output_path} ({len(entries)} shots)")


def read_shotlist_csv(csv_path: Path) -> list[ShotlistEntry]:
    """
    Read shotlist from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of ShotlistEntry
    """
    entries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            entry = ShotlistEntry(
                shot_id=int(row.get('shot_id', 0)),
                segment=row.get('segment', ''),
                start_time=float(row.get('start_time', 0)),
                end_time=float(row.get('end_time', 0)),
                visual_description=row.get('visual_description', ''),
                notes=row.get('notes', ''),
            )
            entries.append(entry)
    
    return entries


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def generate(
    topic: str = typer.Argument(..., help="Video topic"),
    script_file: Optional[Path] = typer.Option(
        None, "--script-file", "-f",
        help="Path to script/voiceover text file",
    ),
    script: Optional[str] = typer.Option(
        None, "--script", "-s",
        help="Script text directly",
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
        help="Number of shots to generate",
    ),
) -> None:
    """
    Generate shotlist from topic and script.
    
    Uses LLM if available, falls back to templates.
    """
    console.print("[bold cyan]=== Shotlist Generator ===[/bold cyan]\n")
    
    # Get script text
    script_text = script
    if script_file and script_file.exists():
        script_text = script_file.read_text(encoding='utf-8')
        console.print(f"Script: {script_file}")
    elif script:
        console.print(f"Script: (provided inline, {len(script)} chars)")
    else:
        console.print(f"Script: (none provided, using topic only)")
        script_text = f"A video about {topic}."
    
    console.print(f"Topic: {topic}")
    console.print(f"Duration: {duration}s")
    console.print(f"Output: {output}")
    
    # Generate shotlist
    console.print("\n[bold]Generating shotlist...[/bold]")
    
    entries = asyncio.run(generate_shotlist_with_llm(
        script=script_text,
        topic=topic,
        duration_hint=duration,
        num_shots=num_shots,
    ))
    
    if not entries:
        console.print("[red]Failed to generate shotlist[/red]")
        raise typer.Exit(1)
    
    # Write CSV
    write_shotlist_csv(entries, output)
    
    # Display results
    console.print(f"\n[green]Generated {len(entries)} shots[/green]\n")
    
    for entry in entries:
        console.print(f"[bold]{entry.shot_id}. {entry.segment}[/bold] ({entry.start_time:.0f}s - {entry.end_time:.0f}s)")
        console.print(f"   {entry.visual_description[:80]}...")
        console.print()


@app.command()
def preview(
    csv_path: Path = typer.Argument(..., help="Path to shotlist.csv"),
) -> None:
    """
    Preview an existing shotlist.
    """
    console.print("[bold cyan]=== Shotlist Preview ===[/bold cyan]\n")
    
    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)
    
    entries = read_shotlist_csv(csv_path)
    
    console.print(f"File: {csv_path}")
    console.print(f"Total shots: {len(entries)}")
    console.print()
    
    total_duration = 0.0
    for entry in entries:
        duration = entry.end_time - entry.start_time
        total_duration += duration
        
        console.print(f"[bold]{entry.shot_id}. {entry.segment}[/bold]")
        console.print(f"   Time: {entry.start_time:.1f}s - {entry.end_time:.1f}s ({duration:.1f}s)")
        console.print(f"   Visual: {entry.visual_description}")
        if entry.notes:
            console.print(f"   Notes: {entry.notes}")
        console.print()
    
    console.print(f"[bold]Total duration: {total_duration:.1f}s[/bold]")


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
