"""
Myth Museum - Shorts Retention Optimizer

Enhances scripts and visuals for YouTube Shorts algorithm optimization:
1. Mid-video retention hooks (15-25s)
2. Comment-triggering endings
3. Series markers for channel branding
4. Stop-scroll first image prompts

Based on proven Shorts engagement strategies.
"""

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Words per second estimate for timing calculations
WORDS_PER_SECOND = 2.5

# Target video duration (seconds)
TARGET_DURATION_MIN = 45
TARGET_DURATION_MAX = 70

# Target word count for scripts
TARGET_WORDS_MIN = int(TARGET_DURATION_MIN * WORDS_PER_SECOND)  # ~112 words
TARGET_WORDS_MAX = int(TARGET_DURATION_MAX * WORDS_PER_SECOND)  # ~175 words

# Ideal seconds per image for smooth pacing
IDEAL_SECONDS_PER_IMAGE = 8  # Slightly faster pacing for 6 images
MIN_IMAGES = 6  # 6 images for better narrative arc
MAX_IMAGES = 8

# Retention hook insertion window (seconds)
RETENTION_HOOK_START = 15
RETENTION_HOOK_END = 25

# Retention hook templates
RETENTION_HOOKS = [
    "But here's the part most people miss.",
    "And this is where it gets interesting.",
    "But wait — there's something everyone overlooks.",
    "Here's what they don't tell you.",
    "Now, here's the twist.",
    "But the real story is even stranger.",
    "And here's where it all changes.",
    "But that's not the whole story.",
]

# Comment trigger templates (open-ended questions)
COMMENT_TRIGGERS = {
    "challenge": [
        "How were you taught this?",
        "Which version did you grow up with?",
        "What did your school tell you?",
        "Sound familiar?",
    ],
    "choice": [
        "Do you buy the original story — or this one?",
        "Which version makes more sense to you?",
        "What do you think really happened?",
        "Which side are you on?",
    ],
    "curiosity": [
        "What other 'facts' have we been told wrong?",
        "Makes you wonder what else we got backwards.",
        "What myths are you questioning now?",
        "What else should we investigate?",
    ],
}

# Series categories and names (ORDER MATTERS - more specific keywords first)
SERIES_CATEGORIES = [
    # Specific civilizations (highest priority)
    ("aztec", "Lost Civs"),
    ("maya", "Lost Civs"),
    ("mayan", "Lost Civs"),
    ("egypt", "Lost Civs"),
    ("inca", "Lost Civs"),
    ("sumerian", "Lost Civs"),
    ("babylon", "Lost Civs"),
    ("mesopotamia", "Lost Civs"),
    
    # Greek/Roman mythology
    ("hades", "Greek Myths"),
    ("zeus", "Greek Myths"),
    ("poseidon", "Greek Myths"),
    ("athena", "Greek Myths"),
    ("olympus", "Greek Myths"),
    ("greek", "Greek Myths"),
    ("roman myth", "Roman Myths"),
    ("mythology", "Myth Files"),
    ("god", "God Files"),
    
    # Music legends (high priority - before generic history)
    ("beethoven", "Music Legends"),
    ("mozart", "Music Legends"),
    ("bach", "Music Legends"),
    ("chopin", "Music Legends"),
    ("composer", "Music Legends"),
    ("symphony", "Music Legends"),
    ("piano", "Music Legends"),
    ("orchestra", "Music Legends"),
    ("classical music", "Music Legends"),
    
    # Specific historical figures
    ("lincoln", "History Lies"),
    ("napoleon", "History Lies"),
    ("caesar", "Empire Files"),
    ("cleopatra", "Empire Files"),
    
    # History categories
    ("civilization", "Lost Civs"),
    ("ancient", "Lost Civs"),
    ("empire", "Empire Files"),
    ("rome", "Empire Files"),
    ("roman", "Empire Files"),
    ("war", "War Myths"),
    ("battle", "War Myths"),
    ("history", "History Lies"),
    
    # Science
    ("brain", "Mind Tricks"),
    ("psychology", "Mind Tricks"),
    ("memory", "Mind Tricks"),
    ("health", "Health Myths"),
    ("body", "Health Myths"),
    ("food", "Food Myths"),
    ("science", "Science Myths"),
    ("evolution", "Science Myths"),
]

SERIES_DEFAULT = "Myth Museum"


# ============================================================================
# Series Visual Styles
# ============================================================================

SERIES_VISUAL_STYLES = {
    "Greek Myths": {
        "palette": "marble white, gold accents, Mediterranean azure blue, bronze",
        "lighting": "dramatic chiaroscuro, divine golden rays, torch light",
        "texture": "classical sculpture, ancient marble temple columns, olive branches",
        "mood": "epic, mythological, timeless, divine",
        "era": "ancient Greece, classical antiquity",
    },
    "Roman Myths": {
        "palette": "imperial purple, gold, marble white, crimson",
        "lighting": "warm torchlight, Mediterranean sun, dramatic shadows",
        "texture": "Roman columns, laurel wreaths, armor, mosaics",
        "mood": "imperial, powerful, classical",
        "era": "Roman Empire, classical antiquity",
    },
    "History Lies": {
        "palette": "sepia tones, documentary browns, aged paper yellows, black and white",
        "lighting": "natural daylight, journalistic, archival lighting",
        "texture": "old photographs, yellowed newspapers, dusty documents, archives",
        "mood": "investigative, revelatory, journalistic, truth-seeking",
        "era": "varies by topic, historical documentary style",
    },
    "War Myths": {
        "palette": "desaturated grays, smoke blacks, blood red accents, steel blue",
        "lighting": "harsh battlefield light, dramatic shadows, fire glow",
        "texture": "armor, weapons, banners, mud, smoke, fortifications",
        "mood": "intense, historical, heroic, somber",
        "era": "medieval to modern warfare",
    },
    "Lost Civs": {
        "palette": "jungle greens, temple golds, turquoise, terracotta",
        "lighting": "dappled jungle light, mysterious temple shadows, sunset glow",
        "texture": "ancient ruins, carved stone, jungle vines, pyramids",
        "mood": "mysterious, ancient, lost, rediscovered",
        "era": "pre-Columbian, ancient civilizations",
    },
    "Empire Files": {
        "palette": "royal purple, gold, rich reds, marble white",
        "lighting": "grand palace lighting, candlelight, sun rays through windows",
        "texture": "crowns, thrones, palace halls, royal regalia",
        "mood": "powerful, regal, political, dramatic",
        "era": "various empires throughout history",
    },
    "Myth Files": {
        "palette": "mystical purples, ethereal blues, silver, moonlight white",
        "lighting": "magical glow, moonlight, mysterious shadows",
        "texture": "ancient symbols, mystical artifacts, starry skies",
        "mood": "mysterious, magical, otherworldly",
        "era": "timeless, mythological",
    },
    "God Files": {
        "palette": "divine gold, heavenly white, celestial blue, fire orange",
        "lighting": "divine rays, ethereal glow, cosmic light",
        "texture": "clouds, temples, divine symbols, sacred objects",
        "mood": "divine, powerful, transcendent",
        "era": "timeless, religious/mythological",
    },
    "Mind Tricks": {
        "palette": "neural blues, electric purple, brain pink, synaptic white",
        "lighting": "clinical white, mysterious darkness, neon accents",
        "texture": "brain scans, neurons, optical illusions, psychology imagery",
        "mood": "scientific, mind-bending, revelatory",
        "era": "modern neuroscience",
    },
    "Science Myths": {
        "palette": "lab coat white, scientific blue, data green, steel gray",
        "lighting": "laboratory lighting, clean and clinical, focused beams",
        "texture": "lab equipment, experiments, scientific diagrams",
        "mood": "educational, myth-busting, evidence-based",
        "era": "modern science",
    },
    "Music Legends": {
        "palette": "warm sepia, golden candlelight, rich burgundy, ivory white, dark mahogany",
        "lighting": "warm candlelight glow, dramatic spotlight, romantic soft lighting",
        "texture": "sheet music pages, piano keys, concert hall velvet, aged parchment",
        "mood": "passionate, emotional, artistic, timeless, romantic",
        "era": "Classical era Vienna, 18th-19th century Europe",
    },
    "Myth Museum": {
        "palette": "museum bronze, gallery white, artifact gold, display case blue",
        "lighting": "museum spotlights, dramatic artifact lighting",
        "texture": "display cases, artifacts, ancient objects, museum halls",
        "mood": "educational, revelatory, curated",
        "era": "varies by topic",
    },
}

# Default style for unknown series
DEFAULT_VISUAL_STYLE = {
    "palette": "cinematic colors, rich contrast",
    "lighting": "dramatic three-point lighting, natural highlights",
    "texture": "realistic, detailed, high quality",
    "mood": "engaging, documentary, professional",
    "era": "appropriate to topic",
}


def get_series_visual_style(series_name: str) -> dict:
    """Get visual style for a series name."""
    return SERIES_VISUAL_STYLES.get(series_name, DEFAULT_VISUAL_STYLE)


# ============================================================================
# Script Duration Utilities
# ============================================================================

@dataclass
class ScriptValidation:
    """Result of script length validation."""
    word_count: int
    estimated_duration: float  # seconds
    is_valid: bool
    recommendation: str
    target_word_range: tuple[int, int]


def count_words(text: str) -> int:
    """Count words in text, handling mixed English/Chinese."""
    # Remove punctuation for counting
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    words = clean_text.split()
    return len(words)


def validate_script_length(script: str) -> ScriptValidation:
    """
    Validate script length for 45-70 second target duration.
    
    Args:
        script: The script text
        
    Returns:
        ScriptValidation with word count, duration estimate, and recommendation
    """
    word_count = count_words(script)
    estimated_duration = word_count / WORDS_PER_SECOND
    
    if word_count < TARGET_WORDS_MIN:
        is_valid = False
        words_needed = TARGET_WORDS_MIN - word_count
        recommendation = f"Script too short. Add ~{words_needed} more words for 45s minimum."
    elif word_count > TARGET_WORDS_MAX:
        is_valid = False
        words_excess = word_count - TARGET_WORDS_MAX
        recommendation = f"Script too long. Remove ~{words_excess} words for 70s maximum."
    else:
        is_valid = True
        recommendation = f"Script length OK. Estimated duration: {estimated_duration:.1f}s"
    
    return ScriptValidation(
        word_count=word_count,
        estimated_duration=estimated_duration,
        is_valid=is_valid,
        recommendation=recommendation,
        target_word_range=(TARGET_WORDS_MIN, TARGET_WORDS_MAX),
    )


def calculate_optimal_images(duration_seconds: float) -> int:
    """
    Calculate optimal image count for smooth visual pacing.
    
    Target: 8-12 seconds per image for best engagement.
    
    Args:
        duration_seconds: Expected video duration
        
    Returns:
        Optimal number of images (4-6)
    """
    if duration_seconds <= 0:
        return MIN_IMAGES
    
    # Calculate ideal count
    ideal_count = round(duration_seconds / IDEAL_SECONDS_PER_IMAGE)
    
    # Clamp to valid range
    return max(MIN_IMAGES, min(MAX_IMAGES, ideal_count))


def estimate_duration_from_script(script: str) -> float:
    """Estimate video duration from script text."""
    word_count = count_words(script)
    return word_count / WORDS_PER_SECOND


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SeriesInfo:
    """Series information for a video."""
    series_name: str
    episode_number: int
    display_text: str  # e.g., "Greek Myths #3"
    
    def to_dict(self) -> dict:
        return {
            "series_name": self.series_name,
            "episode_number": self.episode_number,
            "display_text": self.display_text,
        }


@dataclass
class OptimizedScript:
    """Script with retention optimizations applied."""
    original_script: str
    optimized_script: str
    retention_hook: str
    retention_hook_position: int  # Word position where hook was inserted
    comment_trigger: str
    series_info: Optional[SeriesInfo] = None
    changes_made: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "original_script": self.original_script,
            "optimized_script": self.optimized_script,
            "retention_hook": self.retention_hook,
            "retention_hook_position": self.retention_hook_position,
            "comment_trigger": self.comment_trigger,
            "series_info": self.series_info.to_dict() if self.series_info else None,
            "changes_made": self.changes_made,
        }


@dataclass 
class StopScrollPrompt:
    """First image prompt optimized for stopping scroll."""
    original_prompt: str
    optimized_prompt: str
    techniques_applied: list[str]
    
    def to_dict(self) -> dict:
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "techniques_applied": self.techniques_applied,
        }


# ============================================================================
# Series Registry
# ============================================================================

class SeriesRegistry:
    """
    Tracks episode numbers for each series.
    Persists to JSON file.
    """
    
    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or Path("config/series_registry.json")
        self.registry: dict[str, int] = {}
        self._load()
    
    def _load(self) -> None:
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load series registry: {e}")
                self.registry = {}
    
    def _save(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)
    
    def get_next_episode(self, series_name: str) -> int:
        """Get and increment episode number for a series."""
        current = self.registry.get(series_name, 0)
        next_num = current + 1
        self.registry[series_name] = next_num
        self._save()
        return next_num
    
    def peek_next_episode(self, series_name: str) -> int:
        """Peek at next episode number without incrementing."""
        return self.registry.get(series_name, 0) + 1
    
    def get_all_series(self) -> dict[str, int]:
        """Get all series and their current episode counts."""
        return dict(self.registry)


# ============================================================================
# Script Optimizer
# ============================================================================

class ShortsOptimizer:
    """
    Optimizes scripts for YouTube Shorts retention.
    """
    
    def __init__(self):
        self.series_registry = SeriesRegistry()
    
    def detect_series(self, topic: str) -> str:
        """
        Detect which series a topic belongs to.
        
        ONLY uses topic keywords for detection (not script content).
        This prevents script words like "warfare" from incorrectly
        categorizing a video about Maya civilization as "War Myths".
        
        Args:
            topic: Video topic
        
        Returns:
            Series name
        """
        topic_lower = topic.lower()
        
        # Check topic keywords only
        for keyword, series in SERIES_CATEGORIES:
            if keyword in topic_lower:
                return series
        
        return SERIES_DEFAULT
    
    def get_series_info(
        self,
        topic: str,
        script: str = "",
        series_override: str = None,
        increment: bool = True,
    ) -> SeriesInfo:
        """
        Get series information for a topic.
        
        Args:
            topic: Video topic
            script: Optional script text (unused, kept for compatibility)
            series_override: Manual series name override
            increment: Whether to increment episode counter
        
        Returns:
            SeriesInfo
        """
        # Use override if provided, otherwise detect from topic
        if series_override:
            series_name = series_override
        else:
            series_name = self.detect_series(topic)
        
        if increment:
            episode_num = self.series_registry.get_next_episode(series_name)
        else:
            episode_num = self.series_registry.peek_next_episode(series_name)
        
        display_text = f"{series_name} #{episode_num}"
        
        return SeriesInfo(
            series_name=series_name,
            episode_number=episode_num,
            display_text=display_text,
        )
    
    def find_retention_hook_position(self, script: str) -> tuple[int, int]:
        """
        Find the best position to insert retention hook.
        
        Target: 15-25 seconds into the script (based on word count).
        
        Args:
            script: Script text
        
        Returns:
            Tuple of (word_index, paragraph_index) for insertion
        """
        words = script.split()
        total_words = len(words)
        
        # Calculate target word range
        target_start_word = int(RETENTION_HOOK_START * WORDS_PER_SECOND)
        target_end_word = int(RETENTION_HOOK_END * WORDS_PER_SECOND)
        
        # Find paragraph breaks
        paragraphs = script.split('\n\n')
        
        # Find best paragraph break within target range
        word_count = 0
        best_position = target_start_word
        
        for i, para in enumerate(paragraphs):
            para_words = len(para.split())
            
            if target_start_word <= word_count <= target_end_word:
                # This paragraph break is in our target range
                return word_count, i
            
            word_count += para_words
        
        # If no good paragraph break, use word count estimate
        return min(target_start_word, total_words - 20), 0
    
    def insert_retention_hook(self, script: str, hook: str = None) -> tuple[str, str, int]:
        """
        Insert retention hook into script.
        
        Args:
            script: Original script
            hook: Optional specific hook to use
        
        Returns:
            Tuple of (modified_script, hook_used, word_position)
        """
        if hook is None:
            hook = random.choice(RETENTION_HOOKS)
        
        word_pos, para_idx = self.find_retention_hook_position(script)
        
        # Try to insert at paragraph break
        paragraphs = script.split('\n\n')
        
        # Find the best paragraph index based on word count
        if para_idx > 0 and para_idx < len(paragraphs):
            # Insert hook as new paragraph
            paragraphs.insert(para_idx, hook)
            modified = '\n\n'.join(paragraphs)
        else:
            # Find paragraph that contains the target word position
            word_count = 0
            insert_para_idx = len(paragraphs) // 2  # Default to middle
            
            for i, para in enumerate(paragraphs):
                para_word_count = len(para.split())
                if word_count + para_word_count >= word_pos and i > 0:
                    insert_para_idx = i
                    break
                word_count += para_word_count
            
            # Insert after this paragraph
            if insert_para_idx < len(paragraphs):
                paragraphs.insert(insert_para_idx, hook)
            else:
                paragraphs.insert(len(paragraphs) - 1, hook)
            
            modified = '\n\n'.join(paragraphs)
        
        # Clean up extra whitespace
        modified = re.sub(r'\n{3,}', '\n\n', modified)
        
        return modified, hook, word_pos
    
    def select_comment_trigger(self, script: str, topic: str) -> str:
        """
        Select appropriate comment trigger based on content.
        
        Args:
            script: Script text
            topic: Topic
        
        Returns:
            Comment trigger question
        """
        # Analyze script to pick category
        script_lower = script.lower()
        
        # If script mentions "you were taught" or "school" -> challenge
        if any(word in script_lower for word in ["taught", "school", "learned", "told"]):
            return random.choice(COMMENT_TRIGGERS["challenge"])
        
        # If script presents two versions -> choice
        if any(word in script_lower for word in ["or", "versus", "version", "actually"]):
            return random.choice(COMMENT_TRIGGERS["choice"])
        
        # Default to curiosity
        return random.choice(COMMENT_TRIGGERS["curiosity"])
    
    def replace_closing_with_trigger(self, script: str, trigger: str) -> str:
        """
        Replace closing statement with comment trigger.
        
        Args:
            script: Script text
            trigger: Comment trigger question
        
        Returns:
            Modified script
        """
        paragraphs = script.strip().split('\n\n')
        
        if len(paragraphs) > 1:
            # Replace last paragraph with trigger
            paragraphs[-1] = trigger
        else:
            # Append trigger
            paragraphs.append(trigger)
        
        return '\n\n'.join(paragraphs)
    
    def optimize_script(
        self,
        script: str,
        topic: str,
        add_retention_hook: bool = True,
        add_comment_trigger: bool = True,
        add_series_info: bool = True,
        series_override: str = None,
    ) -> OptimizedScript:
        """
        Apply all optimizations to a script.
        
        Args:
            script: Original script
            topic: Video topic
            add_retention_hook: Whether to add retention hook
            add_comment_trigger: Whether to add comment trigger
            add_series_info: Whether to add series info
            series_override: Manual series name override
        
        Returns:
            OptimizedScript with all modifications
        """
        changes = []
        modified = script
        retention_hook = ""
        hook_position = 0
        comment_trigger = ""
        series_info = None
        
        # 1. Insert retention hook
        if add_retention_hook:
            modified, retention_hook, hook_position = self.insert_retention_hook(modified)
            changes.append(f"Added retention hook at ~{hook_position} words: '{retention_hook}'")
        
        # 2. Replace closing with comment trigger
        if add_comment_trigger:
            comment_trigger = self.select_comment_trigger(script, topic)
            modified = self.replace_closing_with_trigger(modified, comment_trigger)
            changes.append(f"Added comment trigger: '{comment_trigger}'")
        
        # 3. Get series info (don't modify script, just track)
        if add_series_info:
            series_info = self.get_series_info(
                topic=topic,
                series_override=series_override,
                increment=True,
            )
            changes.append(f"Series: {series_info.display_text}")
        
        return OptimizedScript(
            original_script=script,
            optimized_script=modified,
            retention_hook=retention_hook,
            retention_hook_position=hook_position,
            comment_trigger=comment_trigger,
            series_info=series_info,
            changes_made=changes,
        )


# ============================================================================
# Stop-Scroll Image Optimizer
# ============================================================================

class StopScrollOptimizer:
    """
    Optimizes first image prompts for maximum scroll-stopping power.
    """
    
    # Techniques to apply to first image
    STOP_SCROLL_TECHNIQUES = {
        "close_up": "Extreme close-up, face filling frame,",
        "unusual_angle": "Dutch angle, dramatic low angle looking up,",
        "high_contrast": "High contrast, deep shadows, dramatic lighting,",
        "tension": "Tense atmosphere, moment of suspense,",
        "mystery": "Mysterious, partially obscured, silhouette,",
        "off_center": "Off-center composition, rule of thirds ignored,",
        "eye_contact": "Direct eye contact with viewer, intense gaze,",
    }
    
    # Elements to remove from "nice" prompts
    REMOVE_PATTERNS = [
        r"balanced composition",
        r"symmetrical",
        r"pleasant",
        r"beautiful",
        r"serene",
        r"peaceful",
        r"calm",
        r"gentle",
    ]
    
    def optimize_hook_prompt(self, prompt: str, topic: str) -> StopScrollPrompt:
        """
        Transform a standard prompt into a stop-scroll hook prompt.
        
        Args:
            prompt: Original image prompt
            topic: Video topic
        
        Returns:
            StopScrollPrompt with optimized version
        """
        techniques_applied = []
        optimized = prompt
        
        # Remove "nice" elements
        for pattern in self.REMOVE_PATTERNS:
            if re.search(pattern, optimized, re.IGNORECASE):
                optimized = re.sub(pattern, "", optimized, flags=re.IGNORECASE)
                techniques_applied.append(f"Removed: {pattern}")
        
        # Select 2-3 stop-scroll techniques based on topic
        topic_lower = topic.lower()
        
        selected_techniques = []
        
        # Always add high contrast
        selected_techniques.append("high_contrast")
        
        # Add topic-appropriate techniques
        if any(word in topic_lower for word in ["god", "myth", "legend", "hero"]):
            selected_techniques.extend(["unusual_angle", "mystery"])
        elif any(word in topic_lower for word in ["history", "war", "battle", "empire"]):
            selected_techniques.extend(["tension", "close_up"])
        elif any(word in topic_lower for word in ["science", "brain", "psychology"]):
            selected_techniques.extend(["close_up", "eye_contact"])
        else:
            selected_techniques.extend(["unusual_angle", "tension"])
        
        # Apply selected techniques
        technique_prefix = ""
        for tech in selected_techniques[:3]:
            tech_text = self.STOP_SCROLL_TECHNIQUES.get(tech, "")
            if tech_text:
                technique_prefix += tech_text + " "
                techniques_applied.append(tech)
        
        # Prepend techniques to prompt
        optimized = f"{technique_prefix.strip()} {optimized}"
        
        # Add attention-grabbing suffix
        optimized += ", attention-grabbing, scroll-stopping, visually arresting"
        
        # Clean up
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        optimized = re.sub(r',\s*,', ',', optimized)
        
        return StopScrollPrompt(
            original_prompt=prompt,
            optimized_prompt=optimized,
            techniques_applied=techniques_applied,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def optimize_script_for_shorts(
    script: str,
    topic: str,
    output_path: Optional[Path] = None,
) -> OptimizedScript:
    """
    Optimize a script for YouTube Shorts retention.
    
    Args:
        script: Original script
        topic: Video topic
        output_path: Optional path to save optimization report
    
    Returns:
        OptimizedScript
    """
    optimizer = ShortsOptimizer()
    result = optimizer.optimize_script(script, topic)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved optimization report: {output_path}")
    
    return result


def optimize_hook_image(
    prompt: str,
    topic: str,
) -> StopScrollPrompt:
    """
    Optimize first image prompt for scroll-stopping.
    
    Args:
        prompt: Original image prompt
        topic: Video topic
    
    Returns:
        StopScrollPrompt
    """
    optimizer = StopScrollOptimizer()
    return optimizer.optimize_hook_prompt(prompt, topic)


def get_series_marker(topic: str, script: str = "") -> SeriesInfo:
    """
    Get series marker for a topic.
    
    Args:
        topic: Video topic
        script: Optional script for context
    
    Returns:
        SeriesInfo
    """
    optimizer = ShortsOptimizer()
    return optimizer.get_series_info(topic, script, increment=False)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shorts Retention Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Optimize script command
    opt_parser = subparsers.add_parser("optimize", help="Optimize a script")
    opt_parser.add_argument("--script", "-s", required=True, help="Script file path")
    opt_parser.add_argument("--topic", "-t", required=True, help="Video topic")
    opt_parser.add_argument("--output", "-o", help="Output JSON path")
    
    # Series info command
    series_parser = subparsers.add_parser("series", help="Get series info")
    series_parser.add_argument("--topic", "-t", required=True, help="Video topic")
    series_parser.add_argument("--list", "-l", action="store_true", help="List all series")
    
    # Hook image command
    hook_parser = subparsers.add_parser("hook-image", help="Optimize hook image prompt")
    hook_parser.add_argument("--prompt", "-p", required=True, help="Original prompt")
    hook_parser.add_argument("--topic", "-t", required=True, help="Video topic")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        script = Path(args.script).read_text(encoding="utf-8")
        output_path = Path(args.output) if args.output else None
        
        result = optimize_script_for_shorts(script, args.topic, output_path)
        
        print("\n=== Script Optimization ===")
        print(f"Changes made: {len(result.changes_made)}")
        for change in result.changes_made:
            print(f"  - {change}")
        print(f"\nRetention hook: {result.retention_hook}")
        print(f"Comment trigger: {result.comment_trigger}")
        if result.series_info:
            print(f"Series: {result.series_info.display_text}")
        print("\n=== Optimized Script ===")
        print(result.optimized_script)
    
    elif args.command == "series":
        if args.list:
            registry = SeriesRegistry()
            print("\n=== All Series ===")
            for series, count in registry.get_all_series().items():
                print(f"  {series}: {count} episodes")
        else:
            info = get_series_marker(args.topic)
            print(f"\nSeries: {info.display_text}")
    
    elif args.command == "hook-image":
        result = optimize_hook_image(args.prompt, args.topic)
        print("\n=== Hook Image Optimization ===")
        print(f"Techniques applied: {result.techniques_applied}")
        print(f"\nOriginal: {result.original_prompt}")
        print(f"\nOptimized: {result.optimized_prompt}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
