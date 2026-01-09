"""
Myth Museum - Storyboard Prompt Generator

Generate 4 image prompts per topic with forced visual differentiation.
Each segment must have unique camera/subject/action constraints.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.llm import LLMClient
from core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Constants
# ============================================================================

SEGMENTS = ["hook", "myth_formation", "contrast", "resolution"]

# Visual constraint options
CAMERA_OPTIONS = ["wide", "medium", "close-up"]
SUBJECT_OPTIONS = ["single", "multiple", "environment-only"]
ACTION_OPTIONS = ["static", "in-motion", "interaction"]

# Banned words in prompts (abstract concepts)
BANNED_WORDS = {
    "myth", "belief", "truth", "oversimplifies", "misconception",
    "false", "fact", "evidence", "proof", "claim", "debunk",
    "text", "subtitle", "watermark", "logo", "caption", "title",
    "迷思", "信念", "真相", "事實", "證據", "錯誤",
}

# Required style elements
REQUIRED_STYLE = "cinematic lighting, clean composition, realistic, high quality, 8k"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class VisualConstraints:
    """Visual constraints for an image."""
    camera_distance: str  # wide, medium, close-up
    subject_count: str    # single, multiple, environment-only
    action_state: str     # static, in-motion, interaction
    
    def to_dict(self) -> dict:
        return {
            "camera_distance": self.camera_distance,
            "subject_count": self.subject_count,
            "action_state": self.action_state,
        }
    
    def as_tuple(self) -> tuple:
        return (self.camera_distance, self.subject_count, self.action_state)
    
    def as_prompt_prefix(self) -> str:
        """Convert to prompt prefix."""
        camera_map = {
            "wide": "Wide shot,",
            "medium": "Medium shot,",
            "close-up": "Close-up shot,",
        }
        subject_map = {
            "single": "single subject,",
            "multiple": "group of people,",
            "environment-only": "empty environment,",
        }
        action_map = {
            "static": "still pose,",
            "in-motion": "dynamic movement,",
            "interaction": "hands interacting,",
        }
        
        return f"{camera_map.get(self.camera_distance, '')} {subject_map.get(self.subject_count, '')} {action_map.get(self.action_state, '')}"


@dataclass
class StoryboardPrompt:
    """A single storyboard image prompt."""
    segment: str
    prompt: str
    visual_constraints: VisualConstraints
    
    def to_dict(self) -> dict:
        return {
            "segment": self.segment,
            "prompt": self.prompt,
            "visual_constraints": self.visual_constraints.to_dict(),
        }


@dataclass
class StoryboardPromptSet:
    """Complete set of 4 prompts for a topic."""
    topic: str
    prompts: list[StoryboardPrompt] = field(default_factory=list)
    diversity_score: float = 0.0
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "prompts": [p.to_dict() for p in self.prompts],
            "diversity_score": self.diversity_score,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


# ============================================================================
# Visual Constraint Presets
# ============================================================================

# Default constraint combinations that guarantee visual diversity
DEFAULT_CONSTRAINT_SETS = [
    # Set 1: Maximum diversity
    [
        VisualConstraints("close-up", "single", "interaction"),
        VisualConstraints("wide", "multiple", "static"),
        VisualConstraints("medium", "single", "in-motion"),
        VisualConstraints("wide", "environment-only", "static"),
    ],
    # Set 2: Alternative diversity
    [
        VisualConstraints("medium", "single", "static"),
        VisualConstraints("wide", "multiple", "in-motion"),
        VisualConstraints("close-up", "single", "interaction"),
        VisualConstraints("medium", "environment-only", "static"),
    ],
    # Set 3: Another alternative
    [
        VisualConstraints("wide", "single", "in-motion"),
        VisualConstraints("close-up", "multiple", "interaction"),
        VisualConstraints("medium", "single", "static"),
        VisualConstraints("wide", "environment-only", "static"),
    ],
]


# ============================================================================
# Prompt Generator
# ============================================================================


class StoryboardPromptGenerator:
    """
    Generate visually diverse image prompts for storyboards.
    
    Enforces unique visual constraints across all 4 segments.
    """
    
    def __init__(self):
        """Initialize generator."""
        self.client: Optional[LLMClient] = None
        self.prompt_template = self._load_prompt_template()
    
    def _ensure_client(self) -> LLMClient:
        """Ensure LLM client is initialized."""
        if self.client is None:
            self.client = LLMClient.from_config()
        return self.client
    
    async def close(self) -> None:
        """Close LLM client."""
        if self.client:
            await self.client.close()
            self.client = None
    
    def _load_prompt_template(self) -> str:
        """Load the prompt generation template."""
        return """You are an image prompt generator for educational videos about myths and misconceptions.

Generate a detailed image prompt for the "{segment}" segment of a video about:
Topic: {topic}
Script excerpt: {script_excerpt}

Visual constraints (MUST follow):
- Camera: {camera_distance} shot
- Subject: {subject_count}
- Action: {action_state}

SPECIAL RULES FOR "hook" SEGMENT (STOP-SCROLL MODE):
If this is a "hook" segment, the image MUST be designed to STOP SCROLLING:
- Use EXTREME close-up or unusual Dutch angle
- HIGH CONTRAST with deep shadows
- Create TENSION, MYSTERY, or UNEASE
- Off-center, asymmetric composition
- NOT pretty, balanced, or pleasant - make it ARRESTING
- Think: "What would make someone STOP scrolling?"

Rules for ALL segments:
1. Describe a REALISTIC photograph that could be taken
2. Include specific objects, people, lighting, and environment
3. NO text, watermarks, or abstract concepts in the image
4. Focus on concrete, visible elements

BANNED words (do not use): myth, belief, truth, evidence, fact, false, text, subtitle, watermark

Output ONLY the image prompt (1-2 sentences), nothing else."""
    
    def _validate_prompt(self, prompt: str) -> tuple[bool, list[str]]:
        """
        Validate prompt doesn't contain banned words.
        
        Args:
            prompt: Image prompt to validate
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        prompt_lower = prompt.lower()
        
        for banned in BANNED_WORDS:
            if banned.lower() in prompt_lower:
                violations.append(f"Contains banned word: {banned}")
        
        return len(violations) == 0, violations
    
    def _calculate_diversity_score(self, prompts: list[StoryboardPrompt]) -> float:
        """
        Calculate visual diversity score.
        
        Args:
            prompts: List of storyboard prompts
        
        Returns:
            Diversity score 0.0-1.0
        """
        if len(prompts) < 4:
            return 0.0
        
        # Get all constraint tuples
        constraint_tuples = [p.visual_constraints.as_tuple() for p in prompts]
        
        # Count unique combinations
        unique_combos = len(set(constraint_tuples))
        
        # Count unique values per dimension
        unique_cameras = len(set(c[0] for c in constraint_tuples))
        unique_subjects = len(set(c[1] for c in constraint_tuples))
        unique_actions = len(set(c[2] for c in constraint_tuples))
        
        # Calculate score
        combo_score = unique_combos / 4.0
        dimension_score = (unique_cameras + unique_subjects + unique_actions) / 9.0
        
        return (combo_score * 0.6) + (dimension_score * 0.4)
    
    def _check_duplicate_constraints(self, prompts: list[StoryboardPrompt]) -> int:
        """
        Count how many prompts share the same constraint combination.
        
        Args:
            prompts: List of storyboard prompts
        
        Returns:
            Number of duplicate combinations
        """
        constraint_tuples = [p.visual_constraints.as_tuple() for p in prompts]
        unique_count = len(set(constraint_tuples))
        return len(prompts) - unique_count
    
    def select_constraint_set(self, set_index: int = 0) -> list[VisualConstraints]:
        """
        Select a predefined constraint set.
        
        Args:
            set_index: Index of constraint set to use
        
        Returns:
            List of 4 VisualConstraints
        """
        return DEFAULT_CONSTRAINT_SETS[set_index % len(DEFAULT_CONSTRAINT_SETS)]
    
    async def generate_single_prompt(
        self,
        segment: str,
        topic: str,
        script_excerpt: str,
        constraints: VisualConstraints,
    ) -> StoryboardPrompt:
        """
        Generate a single image prompt.
        
        Args:
            segment: Segment name (hook, myth_formation, etc.)
            topic: Video topic
            script_excerpt: Relevant script text
            constraints: Visual constraints
        
        Returns:
            StoryboardPrompt
        """
        client = self._ensure_client()
        
        if not client.is_configured():
            # Generate template-based prompt without LLM
            return self._generate_template_prompt(segment, topic, constraints)
        
        prompt_request = self.prompt_template.format(
            segment=segment,
            topic=topic,
            script_excerpt=script_excerpt[:300],
            camera_distance=constraints.camera_distance,
            subject_count=constraints.subject_count,
            action_state=constraints.action_state,
        )
        
        messages = [
            {"role": "system", "content": "You are a professional image prompt generator."},
            {"role": "user", "content": prompt_request},
        ]
        
        try:
            response = await client.chat(messages, temperature=0.7)
            prompt_text = response.strip()
            
            # Add style suffix
            prompt_text = f"{prompt_text}, {REQUIRED_STYLE}"
            
            # Validate
            is_valid, violations = self._validate_prompt(prompt_text)
            if not is_valid:
                logger.warning(f"Prompt validation failed: {violations}")
                # Fall back to template
                return self._generate_template_prompt(segment, topic, constraints)
            
            return StoryboardPrompt(
                segment=segment,
                prompt=prompt_text,
                visual_constraints=constraints,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            return self._generate_template_prompt(segment, topic, constraints)
    
    def _generate_template_prompt(
        self,
        segment: str,
        topic: str,
        constraints: VisualConstraints,
    ) -> StoryboardPrompt:
        """
        Generate a template-based prompt without LLM.
        
        Args:
            segment: Segment name
            topic: Video topic
            constraints: Visual constraints
        
        Returns:
            StoryboardPrompt
        """
        # Extract key words from topic (simple extraction)
        topic_words = re.sub(r'[^\w\s]', '', topic).split()[:3]
        topic_desc = " ".join(topic_words)
        
        # STOP-SCROLL MODE for hook segment - designed to make viewers STOP scrolling
        # Uses: high contrast, unusual angles, tension, mystery, unsettling composition
        hook_stop_scroll = (
            f"Extreme close-up, Dutch angle, high contrast dramatic lighting, "
            f"deep shadows, tense atmosphere, mysterious, partially obscured, "
            f"off-center composition, visually arresting, scroll-stopping, "
            f"{topic_desc}, {REQUIRED_STYLE}"
        )
        
        segment_templates = {
            "hook": hook_stop_scroll,  # STOP-SCROLL optimized
            "myth_formation": f"{constraints.as_prompt_prefix()} historical or educational setting showing origin of {topic_desc}, vintage atmosphere, {REQUIRED_STYLE}",
            "contrast": f"{constraints.as_prompt_prefix()} scientific or research environment examining {topic_desc}, laboratory or study setting, {REQUIRED_STYLE}",
            "resolution": f"{constraints.as_prompt_prefix()} enlightening moment of understanding about {topic_desc}, bright and hopeful atmosphere, {REQUIRED_STYLE}",
        }
        
        prompt = segment_templates.get(segment, f"{constraints.as_prompt_prefix()} scene about {topic_desc}, {REQUIRED_STYLE}")
        
        return StoryboardPrompt(
            segment=segment,
            prompt=prompt,
            visual_constraints=constraints,
        )
    
    async def generate_prompt_set(
        self,
        topic: str,
        script: str = "",
        constraint_set_index: int = 0,
    ) -> StoryboardPromptSet:
        """
        Generate complete set of 4 prompts for a topic.
        
        Args:
            topic: Video topic
            script: Full script text (optional)
            constraint_set_index: Which constraint set to use
        
        Returns:
            StoryboardPromptSet with 4 prompts
        """
        result = StoryboardPromptSet(topic=topic)
        
        # Select constraints
        constraints = self.select_constraint_set(constraint_set_index)
        
        # Split script into segments (simple quarter split)
        script_parts = self._split_script(script) if script else [""] * 4
        
        # Generate prompts
        prompts = []
        for i, segment in enumerate(SEGMENTS):
            prompt = await self.generate_single_prompt(
                segment=segment,
                topic=topic,
                script_excerpt=script_parts[i],
                constraints=constraints[i],
            )
            prompts.append(prompt)
        
        result.prompts = prompts
        
        # Calculate diversity
        result.diversity_score = self._calculate_diversity_score(prompts)
        
        # Check for duplicates
        duplicates = self._check_duplicate_constraints(prompts)
        if duplicates >= 3:
            result.validation_errors.append(f"Too many duplicate constraints: {duplicates}")
        
        # Validate all prompts
        for prompt in prompts:
            is_valid, violations = self._validate_prompt(prompt.prompt)
            if not is_valid:
                result.validation_errors.extend(violations)
        
        result.is_valid = len(result.validation_errors) == 0 and result.diversity_score >= 0.6
        
        return result
    
    def _split_script(self, script: str) -> list[str]:
        """Split script into 4 parts for each segment."""
        if not script:
            return [""] * 4
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]
        
        if len(paragraphs) >= 4:
            # Distribute paragraphs to segments
            chunk_size = len(paragraphs) // 4
            return [
                " ".join(paragraphs[:chunk_size]),
                " ".join(paragraphs[chunk_size:chunk_size*2]),
                " ".join(paragraphs[chunk_size*2:chunk_size*3]),
                " ".join(paragraphs[chunk_size*3:]),
            ]
        else:
            # Split by characters
            chars_per_part = len(script) // 4
            return [
                script[:chars_per_part],
                script[chars_per_part:chars_per_part*2],
                script[chars_per_part*2:chars_per_part*3],
                script[chars_per_part*3:],
            ]
    
    def save_prompt_set(
        self,
        prompt_set: StoryboardPromptSet,
        output_path: Path,
    ) -> None:
        """Save prompt set to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(prompt_set.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved prompt set: {output_path}")


# ============================================================================
# Convenience Functions
# ============================================================================


async def generate_storyboard_prompts(
    topic: str,
    script: str = "",
    output_path: Optional[Path] = None,
) -> StoryboardPromptSet:
    """
    Generate storyboard prompts for a topic.
    
    Args:
        topic: Video topic
        script: Optional script text
        output_path: Optional path to save prompts
    
    Returns:
        StoryboardPromptSet
    """
    generator = StoryboardPromptGenerator()
    
    try:
        prompt_set = await generator.generate_prompt_set(topic, script)
        
        if output_path:
            generator.save_prompt_set(prompt_set, output_path)
        
        return prompt_set
        
    finally:
        await generator.close()


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Storyboard Prompt Generator")
    parser.add_argument("--topic", "-t", required=True, help="Video topic")
    parser.add_argument("--script", "-s", help="Path to script file")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--set", type=int, default=0, help="Constraint set index")
    
    args = parser.parse_args()
    
    script = ""
    if args.script:
        script = Path(args.script).read_text(encoding="utf-8")
    
    output_path = Path(args.output) if args.output else None
    
    async def run():
        prompt_set = await generate_storyboard_prompts(
            topic=args.topic,
            script=script,
            output_path=output_path,
        )
        
        print(f"\n=== Storyboard Prompts for: {args.topic} ===")
        print(f"Diversity Score: {prompt_set.diversity_score:.2f}")
        print(f"Valid: {prompt_set.is_valid}")
        
        for p in prompt_set.prompts:
            print(f"\n[{p.segment}]")
            print(f"  Constraints: {p.visual_constraints.to_dict()}")
            print(f"  Prompt: {p.prompt[:100]}...")
        
        if prompt_set.validation_errors:
            print(f"\nValidation Errors: {prompt_set.validation_errors}")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
