"""
Myth Museum - Cartoon Myth Shorts Generator

Generates standardized 4-image cartoon myth story outputs for YouTube Shorts:
1. The Myth (錯誤印象) - What everyone thinks is true
2. The Doubt (開始懷疑) - Something doesn't add up
3. The Truth (真相) - The actual fact
4. The Reframe (重新理解) - New understanding, myth vs truth

All images use cartoon/flat illustration style suitable for educational content.

Usage:
    python -m pipeline.cartoon_myth_generator "Vikings wore horned helmets" --topic history
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from core.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="cartoon-myth",
    help="Generate 4-image cartoon myth story prompts and YouTube topic",
    add_completion=False,
)


# ============================================================================
# Constants - 4-Image Story Structure
# ============================================================================

MYTH_STORY_STRUCTURE = [
    {
        "scene": "the_myth",
        "label": "The Myth (錯誤印象)",
        "description": "What everyone thinks is true - exaggerated, stereotypical",
        "camera": "Wide shot",
        "mood": "bold, confident, dramatic",
        "visual_hint": "exaggerated representation of the common misconception",
    },
    {
        "scene": "the_doubt",
        "label": "The Doubt (開始懷疑)",
        "description": "Wait, something doesn't add up - hint of questioning",
        "camera": "Medium shot",
        "mood": "curious, uncertain, questioning",
        "visual_hint": "character with confused expression, question marks",
    },
    {
        "scene": "the_truth",
        "label": "The Truth (真相)",
        "description": "The actual fact - clear, simple, easy to understand",
        "camera": "Close-up",
        "mood": "clear, enlightening, revelatory",
        "visual_hint": "clear presentation of the correct information, lightbulb moment",
    },
    {
        "scene": "the_reframe",
        "label": "The Reframe (重新理解)",
        "description": "New understanding - myth vs truth contrast, setup for CTA",
        "camera": "Split comparison shot",
        "mood": "satisfied, thoughtful, conclusive",
        "visual_hint": "side-by-side comparison showing myth crossed out, truth highlighted",
    },
]

# Cartoon Style Suffix - MUST be appended to ALL prompts
CARTOON_STYLE_SUFFIX = (
    "cartoon style, flat illustration, clean educational animation, "
    "simple shapes, bright colors, friendly characters, age-safe, "
    "no text, no subtitles, no logos, no watermarks, no violence, "
    "digital art, vector style, 8k quality"
)


# ============================================================================
# YouTube Shorts Topic Templates
# ============================================================================

# Title templates (deterministic selection based on topic hash)
TITLE_TEMPLATES = [
    "This is not what you think",
    "Everyone gets this wrong",
    "The myth everyone believes",
    "What they never told you",
    "Think again about this",
    "The truth will surprise you",
    "Stop believing this myth",
    "This changes everything",
]

# Required hashtags for all Shorts
REQUIRED_HASHTAGS = [
    "#MythBusted",
    "#DidYouKnow",
    "#Shorts",
]

# Topic-specific hashtags
TOPIC_HASHTAGS = {
    "history": ["#HistoryMyth", "#HistoryFacts", "#LearnOnTikTok", "#History"],
    "science": ["#ScienceMyth", "#ScienceFacts", "#STEM", "#Science"],
    "health": ["#HealthMyth", "#HealthFacts", "#Wellness", "#Health"],
    "psychology": ["#PsychologyMyth", "#MindBlown", "#Psychology", "#Brain"],
    "food": ["#FoodMyth", "#FoodFacts", "#Nutrition", "#Food"],
    "nature": ["#NatureMyth", "#NatureFacts", "#Wildlife", "#Nature"],
    "technology": ["#TechMyth", "#TechFacts", "#Technology", "#Tech"],
    "culture": ["#CultureMyth", "#CultureFacts", "#Culture", "#Traditions"],
}

# Generic hashtags to fill remaining slots
GENERIC_HASHTAGS = [
    "#Education",
    "#FunFacts",
    "#LearnSomethingNew",
    "#FactCheck",
    "#TruthRevealed",
]

# Description templates (no emojis for Windows terminal compatibility)
DESCRIPTION_TEMPLATES = [
    {
        "line1": "You've heard this claim a thousand times.",
        "line2": "But the real story? Completely different.",
        "line3": "Comment what myth you want us to bust next!",
    },
    {
        "line1": "Everyone believes this is true.",
        "line2": "Turns out, it's way more interesting than that.",
        "line3": "Drop a myth you've always wondered about!",
    },
    {
        "line1": "This is what movies and textbooks show you.",
        "line2": "But historians/scientists say otherwise.",
        "line3": "What myth should we tackle next?",
    },
    {
        "line1": "You probably learned this wrong.",
        "line2": "The truth is actually fascinating.",
        "line3": "Tell us a myth you want debunked!",
    },
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CartoonImagePrompt:
    """A single cartoon image prompt for one scene."""
    scene: str
    label: str
    prompt: str
    fallback_keyword: str
    
    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "label": self.label,
            "prompt": self.prompt,
            "fallback_keyword": self.fallback_keyword,
        }


@dataclass
class ShortsTopic:
    """YouTube Shorts topic with title, description, and hashtags."""
    title: str
    description: str
    hashtags: list[str]
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "hashtags": self.hashtags,
        }


@dataclass
class CartoonMythOutput:
    """Complete output for a cartoon myth story."""
    cartoon_image_prompts: list[CartoonImagePrompt]
    shorts_topic: ShortsTopic
    
    def to_dict(self) -> dict:
        return {
            "cartoon_image_prompts": [p.to_dict() for p in self.cartoon_image_prompts],
            "shorts_topic": self.shorts_topic.to_dict(),
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _deterministic_index(text: str, max_index: int) -> int:
    """
    Get a deterministic index based on text hash.
    Same input always produces same output.
    """
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return hash_value % max_index


def _extract_subject(myth: str) -> str:
    """
    Extract the main subject from a myth claim.
    Returns a short subject phrase for use in templates.
    """
    # Remove common prefixes
    myth_clean = myth.strip()
    prefixes_to_remove = [
        "People believe that ",
        "It is said that ",
        "Many think that ",
        "Everyone knows that ",
        "The myth that ",
    ]
    for prefix in prefixes_to_remove:
        if myth_clean.lower().startswith(prefix.lower()):
            myth_clean = myth_clean[len(prefix):]
            break
    
    # Take first few words as subject
    words = myth_clean.split()
    if len(words) <= 4:
        return myth_clean
    
    # Find a natural break point
    subject = " ".join(words[:4])
    if len(subject) > 30:
        subject = " ".join(words[:3])
    
    return subject


def _build_scene_prompt(
    scene_config: dict,
    myth: str,
    truth: str,
    subject: str,
) -> str:
    """
    Build a complete image prompt for a scene.
    Combines scene-specific elements with cartoon style suffix.
    """
    scene = scene_config["scene"]
    camera = scene_config["camera"]
    mood = scene_config["mood"]
    
    # Build scene-specific content
    if scene == "the_myth":
        # Exaggerated representation of the misconception
        content = f"{camera}, {subject} shown in exaggerated stereotypical way, {mood} atmosphere"
    elif scene == "the_doubt":
        # Character questioning the myth
        content = f"{camera}, friendly cartoon character with confused puzzled expression, looking at evidence, question marks floating nearby, {mood} atmosphere"
    elif scene == "the_truth":
        # Clear presentation of truth
        content = f"{camera}, friendly cartoon educator presenting the correct fact clearly, lightbulb moment, {mood} atmosphere"
    elif scene == "the_reframe":
        # Side-by-side comparison
        content = f"{camera}, myth vs truth comparison layout, old belief fading away, new understanding emerging, {mood} atmosphere"
    else:
        content = f"{camera}, educational cartoon scene about {subject}, {mood} atmosphere"
    
    # Combine with cartoon style suffix
    full_prompt = f"{content}, {CARTOON_STYLE_SUFFIX}"
    
    return full_prompt


def _get_fallback_keyword(scene: str, subject: str) -> str:
    """Get a fallback search keyword for stock photos."""
    keywords = {
        "the_myth": f"{subject} cartoon",
        "the_doubt": "confused thinking cartoon",
        "the_truth": "education learning cartoon",
        "the_reframe": "comparison before after cartoon",
    }
    return keywords.get(scene, "educational cartoon")


# ============================================================================
# Main Generator Functions
# ============================================================================

def generate_cartoon_image_prompts(
    myth: str,
    truth: str = "",
    topic: str = "general",
    subject_override: str = "",
) -> list[CartoonImagePrompt]:
    """
    Generate 4 cartoon-style image prompts for a myth story.
    
    Args:
        myth: The myth/misconception claim
        truth: The true fact (optional, used for better prompts)
        topic: Topic category (history, science, health, etc.)
        subject_override: Override auto-extracted subject
    
    Returns:
        List of 4 CartoonImagePrompt objects
    """
    # Extract subject from myth
    subject = subject_override if subject_override else _extract_subject(myth)
    
    prompts = []
    for scene_config in MYTH_STORY_STRUCTURE:
        prompt_text = _build_scene_prompt(scene_config, myth, truth, subject)
        fallback = _get_fallback_keyword(scene_config["scene"], subject)
        
        prompts.append(CartoonImagePrompt(
            scene=scene_config["scene"],
            label=scene_config["label"],
            prompt=prompt_text,
            fallback_keyword=fallback,
        ))
    
    return prompts


def generate_shorts_topic(
    myth: str,
    truth: str = "",
    topic: str = "general",
) -> ShortsTopic:
    """
    Generate YouTube Shorts topic with title, description, and hashtags.
    
    Args:
        myth: The myth/misconception claim
        truth: The true fact (optional)
        topic: Topic category for hashtag selection
    
    Returns:
        ShortsTopic object
    """
    # Deterministic title selection based on myth text
    title_idx = _deterministic_index(myth, len(TITLE_TEMPLATES))
    title = TITLE_TEMPLATES[title_idx]
    
    # Ensure title is <= 45 characters
    if len(title) > 45:
        title = title[:42] + "..."
    
    # Deterministic description selection
    desc_idx = _deterministic_index(myth + "desc", len(DESCRIPTION_TEMPLATES))
    desc_template = DESCRIPTION_TEMPLATES[desc_idx]
    
    # Build description (3 lines)
    description = f"{desc_template['line1']}\n{desc_template['line2']}\n{desc_template['line3']}"
    
    # Build hashtags (8-12 total)
    hashtags = list(REQUIRED_HASHTAGS)  # Start with required (3)
    
    # Add topic-specific hashtags
    topic_lower = topic.lower()
    if topic_lower in TOPIC_HASHTAGS:
        hashtags.extend(TOPIC_HASHTAGS[topic_lower])
    else:
        # Default to history if topic not found
        hashtags.extend(TOPIC_HASHTAGS.get("history", []))
    
    # Fill remaining slots with generic hashtags
    remaining_slots = 10 - len(hashtags)
    if remaining_slots > 0:
        generic_idx = _deterministic_index(myth + "hash", len(GENERIC_HASHTAGS))
        for i in range(remaining_slots):
            idx = (generic_idx + i) % len(GENERIC_HASHTAGS)
            tag = GENERIC_HASHTAGS[idx]
            if tag not in hashtags:
                hashtags.append(tag)
    
    # Limit to 12 hashtags
    hashtags = hashtags[:12]
    
    return ShortsTopic(
        title=title,
        description=description,
        hashtags=hashtags,
    )


def generate_cartoon_myth_output(
    myth: str,
    truth: str = "",
    topic: str = "general",
    subject_override: str = "",
) -> CartoonMythOutput:
    """
    Generate complete cartoon myth output with prompts and topic.
    
    This is the main entry point for generating all cartoon myth content.
    
    Args:
        myth: The myth/misconception claim
        truth: The true fact (optional)
        topic: Topic category
        subject_override: Override auto-extracted subject
    
    Returns:
        CartoonMythOutput with image prompts and shorts topic
    """
    prompts = generate_cartoon_image_prompts(
        myth=myth,
        truth=truth,
        topic=topic,
        subject_override=subject_override,
    )
    
    shorts_topic = generate_shorts_topic(
        myth=myth,
        truth=truth,
        topic=topic,
    )
    
    return CartoonMythOutput(
        cartoon_image_prompts=prompts,
        shorts_topic=shorts_topic,
    )


# ============================================================================
# Integration Helper for export_shorts_pack
# ============================================================================

def add_cartoon_myth_to_metadata(
    metadata: dict,
    packet: dict,
) -> dict:
    """
    Add cartoon myth prompts and shorts topic to existing metadata.
    
    This function is called from export_shorts_pack.py to enrich metadata.
    
    Args:
        metadata: Existing metadata dict
        packet: Original packet data
    
    Returns:
        Enriched metadata dict with cartoon_image_prompts and shorts_topic
    """
    myth = packet.get("claim", "")
    truth = packet.get("truth", packet.get("one_line_verdict", ""))
    topic = packet.get("topic", "general")
    
    if not myth:
        logger.warning("No myth/claim found in packet, skipping cartoon generation")
        return metadata
    
    output = generate_cartoon_myth_output(
        myth=myth,
        truth=truth,
        topic=topic,
    )
    
    metadata["cartoon_image_prompts"] = [p.to_dict() for p in output.cartoon_image_prompts]
    metadata["shorts_topic"] = output.shorts_topic.to_dict()
    
    return metadata


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def generate(
    myth: str = typer.Argument(..., help="The myth/misconception claim"),
    topic: str = typer.Option(
        "history",
        "--topic", "-t",
        help="Topic category (history, science, health, psychology, food, nature, technology, culture)",
    ),
    truth: str = typer.Option(
        "",
        "--truth",
        help="The true fact (optional, improves prompts)",
    ),
    subject: str = typer.Option(
        "",
        "--subject", "-s",
        help="Override auto-extracted subject",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file path",
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--compact",
        help="Pretty print JSON output",
    ),
) -> None:
    """
    Generate 4-image cartoon myth story prompts and YouTube topic.
    
    Examples:
        python -m pipeline.cartoon_myth_generator "Vikings wore horned helmets" -t history
        python -m pipeline.cartoon_myth_generator "We only use 10% of our brain" -t science
        python -m pipeline.cartoon_myth_generator "Goldfish have 3-second memory" -t nature
    """
    console.print("[bold cyan]=== Cartoon Myth Shorts Generator ===[/bold cyan]\n")
    
    console.print(f"[bold]Myth:[/bold] {myth}")
    console.print(f"[bold]Topic:[/bold] {topic}")
    if truth:
        console.print(f"[bold]Truth:[/bold] {truth}")
    console.print()
    
    # Generate output
    output = generate_cartoon_myth_output(
        myth=myth,
        truth=truth,
        topic=topic,
        subject_override=subject,
    )
    
    # Display results (using print for Windows compatibility with Unicode)
    print("\n[Generated 4 Image Prompts]\n")
    for i, prompt in enumerate(output.cartoon_image_prompts, 1):
        print(f"{i}. {prompt.scene}")
        print(f"   Prompt: {prompt.prompt[:100]}...")
        print(f"   Fallback: {prompt.fallback_keyword}")
        print()
    
    print("[YouTube Shorts Topic]\n")
    print(f"Title: {output.shorts_topic.title}")
    print(f"Description:")
    for line in output.shorts_topic.description.split("\n"):
        print(f"   {line}")
    print(f"Hashtags: {' '.join(output.shorts_topic.hashtags)}")
    
    # Save to file if requested
    if output_file:
        output_dict = output.to_dict()
        indent = 2 if pretty else None
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(output_dict, ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        console.print(f"\n[green]Saved to:[/green] {output_file}")


@app.command("example")
def show_example() -> None:
    """
    Show a complete example output with reusable template.
    """
    console.print("[bold cyan]=== Example Cartoon Myth Output ===[/bold cyan]\n")
    
    # Generate example with a generic myth
    example_myth = "Ancient warriors wore elaborate horned helmets in battle"
    example_truth = "Horned helmets were ceremonial, not used in actual combat"
    
    output = generate_cartoon_myth_output(
        myth=example_myth,
        truth=example_truth,
        topic="history",
    )
    
    # Pretty print the full JSON (use ensure_ascii=True for Windows compatibility)
    output_dict = output.to_dict()
    console.print("[bold]Full JSON Output:[/bold]\n")
    print(json.dumps(output_dict, ensure_ascii=True, indent=2))


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
