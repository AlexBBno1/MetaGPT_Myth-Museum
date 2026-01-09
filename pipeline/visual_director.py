"""
Myth Museum - LLM Visual Director

Analyzes scripts and generates scene-specific image prompts using LLM.
Replaces generic prompts with intelligent, narrative-aware visual directions.

Features:
- Script segmentation into visual scenes
- LLM-powered visual brief generation
- Storyboard template-based generation (6-scene narrative arcs)
- Automatic keyword extraction for fallbacks
- Series-specific style application
- Character consistency across scenes
- Concise prompt generation (max 80 words)
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.llm import LLMClient
from core.logging import get_logger
from pipeline.shorts_optimizer import (
    SERIES_VISUAL_STYLES,
    DEFAULT_VISUAL_STYLE,
    get_series_visual_style,
)
from pipeline.storyboard_templates import (
    get_narrative_arc,
    get_style_template,
    get_arc_with_prompts,
    detect_arc_type,
    STYLE_TEMPLATES,
    NARRATIVE_ARCS,
)
from pipeline.character_library import (
    get_character_for_prompt,
    detect_character_from_topic,
    CharacterProfile,
)

logger = get_logger(__name__)


# ============================================================================
# Constants for Prompt Optimization
# ============================================================================

# Maximum prompt length (characters) - increased for more detailed prompts
MAX_PROMPT_LENGTH = 400
MAX_PROMPT_WORDS = 80

# Visual description mappings - convert abstract concepts to visual descriptions
VISUAL_MAPPINGS = {
    # Actions to visual descriptions
    r'scor\w+\s+(\d+)\s*points?': 'athlete celebrating with raised arms after scoring',
    r'hit.+game.?winner': 'triumphant athlete in victory pose',
    r'collapse[sd]?': 'exhausted person being supported by teammate',
    r'barely\s+stand': 'person hunched over, hands on knees, visibly exhausted',
    r'sweating|shaking': 'close-up of sweating face, intense expression',
    r'pale': 'person with pale complexion, visible exhaustion',
    r'order\w*\s+pizza': 'pizza delivery box on hotel bed, dim lighting',
    r'food\s*poison': 'person clutching stomach in distress',
    r'deliberate\s+sabotage': 'suspicious shadowy figures, noir style',
    
    # Sports contexts
    r'NBA\s*Finals': '1990s basketball arena, championship atmosphere, packed crowd',
    r'Game\s+(\d+)': 'intense basketball game scene, scoreboard visible',
    r'basketball\s+player': 'basketball player in action on court',
    
    # Historical contexts
    r'battle|warfare': 'dramatic battlefield scene with soldiers',
    r'ancient\s+\w+': 'ancient ruins or temple, historical atmosphere',
    r'medieval': 'medieval castle or battlefield scene',
    r'myth|legend': 'mythological scene with dramatic lighting',
    
    # Music and composers
    r'deaf|hearing\s*loss|couldn\'t\s*hear': 'close-up of hand cupping ear, pained expression, dramatic lighting',
    r'symphony|orchestra': 'grand concert hall with orchestra performing, dramatic candlelight',
    r'conduct\w*|baton': 'passionate conductor with raised baton, dynamic dramatic pose',
    r'piano|keyboard': 'elegant grand piano in candlelit room, scattered sheet music',
    r'compos\w*|writing\s*music': 'artist at wooden desk with quill pen, scattered music sheets',
    r'applause|audience|standing\s*ovation': 'concert hall audience in standing ovation, emotional faces, tears',
    r'beethoven': 'wild-haired composer in period clothing, passionate intense expression',
    r'mozart': 'elegant young composer in 18th century attire, powdered wig',
    r'sheet\s*music|score|notes': 'handwritten musical score pages, candlelight, ink quill',
    r'premiere|debut|first\s*performance': 'packed concert hall, anticipation, grand chandeliers',
}

# Words/phrases to filter out (non-visual, abstract)
NON_VISUAL_PATTERNS = [
    r"But here's",
    r"What they don't tell",
    r"varies by topic",
    r"It's one of",
    r"most iconic",
    r"everyone calls",
    r"the real story",
    r"years later",
    r"probably",
    r"actually",
    r"however",
    r"therefore",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneSegment:
    """A segment of script mapped to a visual scene."""
    index: int
    text: str
    scene_type: str  # hook, context, turning_point, resolution
    estimated_duration: float  # seconds
    key_sentence: str  # Most important sentence for visual


@dataclass
class VisualBrief:
    """Detailed visual direction for a scene."""
    scene_type: str
    subjects: str  # Who/what should be visible
    setting: str  # Where, time of day, era
    mood: str  # Lighting, atmosphere, emotion
    camera: str  # Angle, distance
    key_elements: list[str]  # Specific objects, colors, textures
    fallback_keywords: list[str]  # Search terms for stock photos
    
    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "subjects": self.subjects,
            "setting": self.setting,
            "mood": self.mood,
            "camera": self.camera,
            "key_elements": self.key_elements,
            "fallback_keywords": self.fallback_keywords,
        }


@dataclass
class ScenePrompt:
    """Final image prompt with fallback keywords."""
    segment: str  # bg_1, bg_2, etc.
    prompt: str
    fallback_keyword: str
    visual_brief: Optional[VisualBrief] = None
    
    def to_dict(self) -> dict:
        return {
            "segment": self.segment,
            "prompt": self.prompt,
            "fallback_keyword": self.fallback_keyword,
        }


# ============================================================================
# Visual Director
# ============================================================================

class VisualDirector:
    """
    LLM-powered visual director for script-to-image prompt generation.
    
    Analyzes scripts and generates precise, scene-matched image prompts
    with automatic keyword extraction for fallback searches.
    """
    
    # Scene type mapping
    SCENE_TYPES = ["hook", "context", "turning_point", "resolution"]
    
    # Words per second for duration estimation
    WORDS_PER_SECOND = 2.5
    
    def __init__(self):
        self.client: Optional[LLMClient] = None
    
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
    
    def get_series_style(self, series_name: str) -> dict:
        """Get visual style for a series."""
        return get_series_visual_style(series_name)
    
    def segment_script(self, script: str, num_segments: int = 4) -> list[SceneSegment]:
        """
        Split script into visual scenes based on narrative structure.
        
        Args:
            script: Full script text
            num_segments: Number of segments (default 4)
        
        Returns:
            List of SceneSegment objects
        """
        # Clean script
        script = script.strip()
        if not script:
            return []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in re.split(r'\n\n+', script) if p.strip()]
        
        # If we have fewer paragraphs than segments, split differently
        if len(paragraphs) < num_segments:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', script)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Group sentences into segments
            sentences_per_segment = max(1, len(sentences) // num_segments)
            paragraphs = []
            for i in range(num_segments):
                start = i * sentences_per_segment
                end = start + sentences_per_segment if i < num_segments - 1 else len(sentences)
                paragraphs.append(' '.join(sentences[start:end]))
        
        # Distribute paragraphs to segments
        segments = []
        paras_per_segment = max(1, len(paragraphs) // num_segments)
        
        for i in range(num_segments):
            start_idx = i * paras_per_segment
            end_idx = start_idx + paras_per_segment if i < num_segments - 1 else len(paragraphs)
            
            segment_text = ' '.join(paragraphs[start_idx:end_idx])
            
            # Find key sentence (longest or first)
            sentences = re.split(r'(?<=[.!?])\s+', segment_text)
            key_sentence = max(sentences, key=len) if sentences else segment_text[:100]
            
            # Estimate duration
            word_count = len(segment_text.split())
            duration = word_count / self.WORDS_PER_SECOND
            
            segments.append(SceneSegment(
                index=i,
                text=segment_text,
                scene_type=self.SCENE_TYPES[i] if i < len(self.SCENE_TYPES) else "context",
                estimated_duration=duration,
                key_sentence=key_sentence,
            ))
        
        return segments
    
    def extract_keywords(self, text: str, topic: str = "") -> list[str]:
        """
        Extract visual keywords from text using pattern matching.
        
        Args:
            text: Text to extract from
            topic: Topic for additional context
        
        Returns:
            List of keywords
        """
        keywords = []
        
        # Add topic words
        if topic:
            topic_words = re.findall(r'\b[A-Z][a-z]+\b', topic)
            keywords.extend(topic_words)
        
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        keywords.extend(proper_nouns[:5])
        
        # Extract numbers with context (years, dates)
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
        keywords.extend(years[:2])
        
        # Common visual subjects
        visual_words = re.findall(
            r'\b(palace|temple|battle|king|queen|soldier|warrior|city|'
            r'mountain|ocean|desert|forest|castle|church|statue|'
            r'painting|book|document|artifact|throne|crown|sword|'
            r'ship|horse|army|crowd|arena|court|stadium)\b',
            text.lower()
        )
        keywords.extend(visual_words[:3])
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and len(kw) > 2:
                seen.add(kw_lower)
                unique.append(kw)
        
        return unique[:5]  # Return top 5
    
    async def generate_visual_brief(
        self,
        segment: SceneSegment,
        topic: str,
        series_name: str,
    ) -> VisualBrief:
        """
        Generate detailed visual brief for a scene using LLM.
        
        Args:
            segment: Script segment
            topic: Video topic
            series_name: Series name for style
        
        Returns:
            VisualBrief with detailed visual directions
        """
        client = self._ensure_client()
        style = self.get_series_style(series_name)
        
        if not client.is_configured():
            # Fallback to template-based brief
            return self._generate_template_brief(segment, topic, style)
        
        prompt = f"""You are a visual director for documentary short videos.

Analyze this script segment and create a detailed visual brief for an AI image generator.

SCRIPT SEGMENT ({segment.scene_type}):
"{segment.text}"

TOPIC: {topic}

SERIES STYLE:
- Color Palette: {style['palette']}
- Lighting: {style['lighting']}
- Textures: {style['texture']}
- Mood: {style['mood']}
- Era: {style['era']}

Create a visual brief with these fields:
1. scene_type: One of (establishing, action, detail, emotional, revelation)
2. subjects: Specific people/objects that should be visible (be precise: "basketball player in red #23 jersey" not "athlete")
3. setting: Exact location, time period, time of day
4. mood: Lighting style, atmosphere, emotional tone
5. camera: Shot type (wide/medium/close-up), angle (eye level/low/high/Dutch)
6. key_elements: Array of 4-6 specific visual elements to include
7. fallback_keywords: Array of 3-5 simple search terms for stock photos

IMPORTANT:
- Be SPECIFIC about subjects (include colors, numbers, clothing details)
- Match the visual to the EXACT content of the script segment
- Ensure the image tells THIS part of the story
- No text, watermarks, or logos in the image

Output valid JSON only."""

        try:
            messages = [
                {"role": "system", "content": "You are a professional visual director. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            
            response = await client.chat_json(messages, temperature=0.4)
            
            if response:
                return VisualBrief(
                    scene_type=response.get("scene_type", segment.scene_type),
                    subjects=response.get("subjects", "relevant scene"),
                    setting=response.get("setting", "appropriate setting"),
                    mood=response.get("mood", style["mood"]),
                    camera=response.get("camera", "medium shot"),
                    key_elements=response.get("key_elements", [])[:6],
                    fallback_keywords=response.get("fallback_keywords", self.extract_keywords(segment.text, topic))[:5],
                )
            
        except Exception as e:
            logger.warning(f"LLM visual brief failed: {e}")
        
        # Fallback to template
        return self._generate_template_brief(segment, topic, style)
    
    def _generate_template_brief(
        self,
        segment: SceneSegment,
        topic: str,
        style: dict,
    ) -> VisualBrief:
        """Generate template-based brief without LLM - with smart visual extraction."""
        keywords = self.extract_keywords(segment.text, topic)
        
        # Extract specific visual subjects from the segment text
        subjects = self._extract_visual_subjects(segment.text, topic)
        
        # Get appropriate setting based on topic
        setting = self._infer_setting_from_topic(topic, style.get("era", ""))
        
        scene_configs = {
            "hook": {
                "scene_type": "establishing",
                "camera": "extreme close-up",
                "mood": "dramatic",
            },
            "context": {
                "scene_type": "establishing", 
                "camera": "wide shot",
                "mood": "atmospheric",
            },
            "turning_point": {
                "scene_type": "revelation",
                "camera": "medium shot",
                "mood": "tense",
            },
            "resolution": {
                "scene_type": "emotional",
                "camera": "wide shot",
                "mood": "reflective",
            },
        }
        
        config = scene_configs.get(segment.scene_type, scene_configs["context"])
        
        return VisualBrief(
            scene_type=config["scene_type"],
            subjects=subjects,
            setting=setting,
            mood=config["mood"],
            camera=config["camera"],
            key_elements=[],  # Keep empty for concise prompts
            fallback_keywords=keywords or [topic.split()[0]],
        )
    
    def _extract_visual_subjects(self, text: str, topic: str) -> str:
        """
        Extract and transform text into concrete visual descriptions.
        
        This is the key improvement - converting abstract mentions 
        to specific visual scenes.
        """
        # First, try to match known visual patterns
        for pattern, visual_desc in VISUAL_MAPPINGS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return visual_desc
        
        # Extract specific visual elements
        visuals = []
        
        # People with descriptions
        if re.search(r'basketball|NBA|player', text, re.IGNORECASE):
            if re.search(r'exhausted|sick|pale|shaking', text, re.IGNORECASE):
                visuals.append("exhausted basketball player, sweating, hands on knees")
            elif re.search(r'scor|points|winner', text, re.IGNORECASE):
                visuals.append("basketball player celebrating victory, arms raised")
            else:
                visuals.append("basketball player on court, intense focus")
        
        if re.search(r'Jordan|Pippen', text, re.IGNORECASE):
            if re.search(r'arms|support|collapse', text, re.IGNORECASE):
                visuals.append("two basketball players, one supporting the other")
        
        # Locations
        if re.search(r'arena|stadium|court', text, re.IGNORECASE):
            visuals.append("professional basketball arena, crowd in background")
        
        if re.search(r'hotel|room|pizza', text, re.IGNORECASE):
            visuals.append("dimly lit hotel room, mysterious atmosphere")
        
        # Historical elements
        if re.search(r'knight|armor|medieval', text, re.IGNORECASE):
            visuals.append("armored medieval knight, dramatic pose")
        
        if re.search(r'battle|war|fight', text, re.IGNORECASE):
            visuals.append("dramatic battlefield scene")
        
        if re.search(r'greek|olymp|god|myth', text, re.IGNORECASE):
            visuals.append("ancient Greek temple, divine atmosphere")
        
        # Music and composers
        if re.search(r'beethoven', text, re.IGNORECASE):
            if re.search(r'deaf|hear|couldn\'t', text, re.IGNORECASE):
                visuals.append("stylized cartoon Beethoven with wild hair, hand to ear, emotional expression")
            elif re.search(r'conduct|premiere|symphony', text, re.IGNORECASE):
                visuals.append("stylized cartoon Beethoven conducting orchestra, passionate pose, concert hall")
            else:
                visuals.append("stylized cartoon Beethoven at piano, candlelight, sheet music scattered")
        
        if re.search(r'piano|keyboard', text, re.IGNORECASE):
            visuals.append("elegant grand piano with sheet music, warm candlelight glow")
        
        if re.search(r'symphony|orchestra|concert', text, re.IGNORECASE):
            visuals.append("grand concert hall with orchestra, dramatic lighting, velvet seats")
        
        if re.search(r'sheet\s*music|score|notes|composing', text, re.IGNORECASE):
            visuals.append("handwritten musical score pages scattered on desk, quill pen, candlelight")
        
        if re.search(r'applause|ovation|audience|crowd', text, re.IGNORECASE):
            visuals.append("concert hall audience standing ovation, emotional tears, chandelier lighting")
        
        # If we found specific visuals, use them
        if visuals:
            return visuals[0]  # Use the most relevant one
        
        # Fallback: use topic-based description
        return self._topic_to_visual(topic)
    
    def _topic_to_visual(self, topic: str) -> str:
        """Convert topic to a generic but appropriate visual description."""
        topic_lower = topic.lower()
        
        topic_visuals = {
            "jordan": "basketball player in red jersey, 1990s NBA atmosphere",
            "flu game": "exhausted athlete, dramatic sports moment",
            "nba": "professional basketball arena, championship atmosphere",
            "teutonic": "medieval knight in white surcoat with black cross",
            "battle": "dramatic medieval battlefield, armored warriors",
            "myth": "ancient temple ruins, mystical atmosphere",
            "greek": "classical Greek architecture, marble columns",
            "napoleon": "military commander in 19th century uniform",
            "aztec": "ancient pyramid temple, jungle setting",
            "maya": "Mayan ruins, tropical forest backdrop",
            "egypt": "ancient Egyptian monuments, golden desert",
            # Music legends
            "beethoven": "stylized cartoon Beethoven with wild hair, passionate expression, candlelight",
            "mozart": "elegant young composer in 18th century attire, harpsichord",
            "bach": "baroque composer at organ, church interior, golden light",
            "chopin": "romantic pianist at grand piano, Parisian salon",
            "symphony": "grand orchestra in concert hall, dramatic lighting",
            "piano": "elegant grand piano with sheet music, warm atmosphere",
            "composer": "classical composer at desk writing music, candlelight",
            "classical music": "ornate concert hall interior, golden chandeliers",
        }
        
        for keyword, visual in topic_visuals.items():
            if keyword in topic_lower:
                return visual
        
        return "dramatic historical scene"
    
    def _infer_setting_from_topic(self, topic: str, default_era: str) -> str:
        """Infer a specific setting from the topic."""
        topic_lower = topic.lower()
        
        topic_settings = {
            "nba": "1990s basketball arena",
            "jordan": "NBA Finals court",
            "flu game": "Utah basketball arena",
            "teutonic": "frozen medieval battlefield",
            "battle": "historic battlefield",
            "greek": "ancient Greece",
            "myth": "mythological realm",
            "napoleon": "19th century Europe",
            "aztec": "ancient Mesoamerica",
            "maya": "Mayan city",
            "egypt": "ancient Egypt",
            "rome": "Roman Empire",
            # Music settings
            "beethoven": "Vienna 1824, grand concert hall",
            "mozart": "18th century Vienna, royal court",
            "bach": "German baroque church interior",
            "chopin": "19th century Parisian salon",
            "symphony": "classical era concert hall",
            "piano": "elegant European salon",
            "composer": "18th-19th century European study",
            "classical music": "ornate European concert hall",
        }
        
        for keyword, setting in topic_settings.items():
            if keyword in topic_lower:
                return setting
        
        # Clean up default era
        if default_era and "varies" not in default_era.lower():
            return default_era.split(',')[0].strip()
        
        return "historical setting"
    
    
    def brief_to_prompt(self, brief: VisualBrief, style: dict) -> str:
        """
        Convert visual brief to a CONCISE Imagen prompt (max 50 words).
        
        Structure: [SHOT] [SUBJECT] [SETTING] [LIGHTING] [STYLE] [SUFFIX]
        
        Args:
            brief: Visual brief
            style: Series style dict
        
        Returns:
            Concise image generation prompt string
        """
        return self._build_concise_prompt(brief, style)
    
    def _build_concise_prompt(self, brief: VisualBrief, style: dict) -> str:
        """
        Build a concise, effective prompt optimized for AI image generation.
        
        Target: 30-50 words max for best results.
        """
        # 1. SHOT TYPE (keep short)
        shot = self._simplify_shot_type(brief.camera)
        
        # 2. SUBJECT (transform to visual description)
        subject = self._transform_to_visual(brief.subjects)
        
        # 3. SETTING (one phrase only)
        setting = self._extract_key_setting(brief.setting, style.get("era", ""))
        
        # 4. LIGHTING (pick one style)
        lighting = self._pick_primary_lighting(style.get("lighting", ""))
        
        # 5. MOOD (one word)
        mood = self._extract_primary_mood(brief.mood)
        
        # Build prompt in optimal order
        parts = [
            shot,
            subject,
            setting,
            lighting,
            mood,
            "8k photorealistic, no text",
        ]
        
        # Join and clean
        prompt = ", ".join(p for p in parts if p and p.strip())
        
        # Final cleanup
        prompt = self._clean_prompt(prompt)
        
        # Enforce length limit
        if len(prompt) > MAX_PROMPT_LENGTH:
            prompt = self._truncate_prompt(prompt)
        
        return prompt
    
    def _simplify_shot_type(self, camera: str) -> str:
        """Extract just the shot type."""
        shot_keywords = {
            "extreme close-up": "Extreme close-up",
            "close-up": "Close-up",
            "medium shot": "Medium shot",
            "wide shot": "Wide shot",
            "establishing": "Wide establishing shot",
            "dutch angle": "Dutch angle",
            "low angle": "Low angle shot",
            "high angle": "High angle shot",
        }
        
        camera_lower = camera.lower()
        for key, value in shot_keywords.items():
            if key in camera_lower:
                return value
        
        return "Medium shot"
    
    def _transform_to_visual(self, subjects: str) -> str:
        """Transform abstract subjects to concrete visual descriptions."""
        result = subjects
        
        # Filter out non-visual phrases
        for pattern in NON_VISUAL_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        
        # Apply visual mappings
        for pattern, replacement in VISUAL_MAPPINGS.items():
            if re.search(pattern, result, re.IGNORECASE):
                # Keep the mapping result but also preserve specific details
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    result = replacement
                    break
        
        # Clean up
        result = re.sub(r'\s+', ' ', result).strip()
        result = result.strip(',').strip()
        
        # Truncate if too long
        if len(result) > 80:
            result = result[:80].rsplit(' ', 1)[0]
        
        return result if result else "dramatic scene"
    
    def _extract_key_setting(self, setting: str, era: str) -> str:
        """Extract one key setting phrase."""
        # Remove "varies by topic" and similar
        setting = re.sub(r'varies by topic,?\s*', '', setting, flags=re.IGNORECASE)
        era = re.sub(r'varies by topic,?\s*', '', era, flags=re.IGNORECASE)
        
        # Prefer specific era if available
        if era and "varies" not in era.lower():
            # Take first meaningful phrase from era
            era_parts = era.split(',')
            if era_parts:
                return era_parts[0].strip()
        
        if setting and "varies" not in setting.lower():
            setting_parts = setting.split(',')
            if setting_parts:
                return setting_parts[0].strip()
        
        return ""
    
    def _pick_primary_lighting(self, lighting: str) -> str:
        """Pick the most impactful lighting style."""
        # Priority order of lighting styles
        priority_lighting = [
            "chiaroscuro",
            "dramatic shadows",
            "golden rays",
            "rim light",
            "torch light",
            "natural daylight",
            "cinematic lighting",
        ]
        
        lighting_lower = lighting.lower()
        for style in priority_lighting:
            if style in lighting_lower:
                return style + " lighting"
        
        # Default
        return "dramatic lighting"
    
    def _extract_primary_mood(self, mood: str) -> str:
        """Extract one primary mood word."""
        # Priority moods
        priority_moods = [
            "dramatic", "intense", "mysterious", "epic",
            "heroic", "somber", "triumphant", "tense",
            "revelatory", "atmospheric", "haunting",
        ]
        
        mood_lower = mood.lower()
        for m in priority_moods:
            if m in mood_lower:
                return m + " atmosphere"
        
        return "cinematic atmosphere"
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean up the prompt."""
        # Remove duplicate words
        words = prompt.split()
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower().strip(',')
            if word_lower not in seen or word_lower in ['the', 'a', 'an', 'and', 'with']:
                seen.add(word_lower)
                unique_words.append(word)
        
        prompt = ' '.join(unique_words)
        
        # Clean punctuation
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = re.sub(r',\s*$', '', prompt)
        
        return prompt.strip()
    
    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max length while keeping it coherent."""
        if len(prompt) <= MAX_PROMPT_LENGTH:
            return prompt
        
        # Try to cut at a comma
        truncated = prompt[:MAX_PROMPT_LENGTH]
        last_comma = truncated.rfind(',')
        if last_comma > MAX_PROMPT_LENGTH // 2:
            truncated = truncated[:last_comma]
        
        # Ensure we still have the suffix
        if "no text" not in truncated:
            truncated += ", no text"
        
        return truncated
    
    async def generate_prompts(
        self,
        script: str,
        topic: str,
        series_name: str,
        num_images: int = 4,
    ) -> list[ScenePrompt]:
        """
        Generate image prompts for a script.
        
        Main entry point - analyzes script and creates scene-matched prompts.
        
        Args:
            script: Full script text
            topic: Video topic
            series_name: Series name for style
            num_images: Number of images to generate
        
        Returns:
            List of ScenePrompt objects
        """
        logger.info(f"Generating visual prompts for: {topic} ({series_name})")
        
        # Get series style
        style = self.get_series_style(series_name)
        
        # Segment script
        segments = self.segment_script(script, num_images)
        
        if not segments:
            logger.warning("No segments found, using fallback prompts")
            return self._generate_fallback_prompts(topic, style, num_images)
        
        # Generate visual briefs for each segment
        prompts = []
        for i, segment in enumerate(segments):
            try:
                brief = await self.generate_visual_brief(segment, topic, series_name)
                prompt_text = self.brief_to_prompt(brief, style)
                
                prompts.append(ScenePrompt(
                    segment=f"bg_{i+1}",
                    prompt=prompt_text,
                    fallback_keyword=" ".join(brief.fallback_keywords[:3]),
                    visual_brief=brief,
                ))
                
                logger.debug(f"Scene {i+1} ({segment.scene_type}): {prompt_text[:100]}...")
                
            except Exception as e:
                logger.error(f"Failed to generate prompt for segment {i}: {e}")
                # Add fallback prompt
                keywords = self.extract_keywords(segment.text, topic)
                prompts.append(ScenePrompt(
                    segment=f"bg_{i+1}",
                    prompt=f"{style['lighting']}, scene related to {topic}, {style['mood']}, cinematic 8k, no text",
                    fallback_keyword=" ".join(keywords[:2]) or topic,
                ))
        
        return prompts
    
    def _generate_fallback_prompts(
        self,
        topic: str,
        style: dict,
        num_images: int,
    ) -> list[ScenePrompt]:
        """Generate fallback prompts when script parsing fails."""
        prompts = []
        scene_types = ["establishing", "context", "action", "resolution"]
        
        for i in range(num_images):
            scene_type = scene_types[i % len(scene_types)]
            
            prompt = (
                f"{scene_type} shot related to {topic}, "
                f"{style['lighting']}, {style['palette']}, "
                f"{style['mood']} atmosphere, "
                f"cinematic composition, photorealistic, 8k quality, no text"
            )
            
            prompts.append(ScenePrompt(
                segment=f"bg_{i+1}",
                prompt=prompt,
                fallback_keyword=topic.split()[0] if topic else "history",
            ))
        
        return prompts


# ============================================================================
# Convenience Functions
# ============================================================================

async def generate_scene_prompts(
    script: str,
    topic: str,
    series_name: str = "History Lies",
    num_images: int = 4,
) -> list[dict]:
    """
    Generate image prompts from script.
    
    Convenience function that returns list of dicts for direct use.
    
    Args:
        script: Full script text
        topic: Video topic
        series_name: Series name
        num_images: Number of images
    
    Returns:
        List of prompt dicts with 'segment', 'prompt', 'fallback_keyword'
    """
    director = VisualDirector()
    
    try:
        prompts = await director.generate_prompts(script, topic, series_name, num_images)
        return [p.to_dict() for p in prompts]
    finally:
        await director.close()


def get_style_for_series(series_name: str) -> dict:
    """Get visual style for a series name."""
    return get_series_visual_style(series_name)


# ============================================================================
# Storyboard Director - Template-based generation
# ============================================================================

class StoryboardDirector:
    """
    Template-based visual director using 6-scene narrative arcs.
    
    Provides more consistent results than LLM-only generation by using
    predefined scene structures and style templates.
    
    Usage:
        director = StoryboardDirector()
        prompts = director.generate_prompts(
            topic="Da Vinci Mona Lisa",
            arc_type="myth_buster",
            style_id="oil_painting_cartoon",
            character="aged wise Leonardo da Vinci with long flowing beard",
        )
    """
    
    def __init__(self):
        self.llm_client: Optional[LLMClient] = None
    
    def _ensure_llm_client(self) -> LLMClient:
        """Ensure LLM client is initialized."""
        if self.llm_client is None:
            self.llm_client = LLMClient.from_config()
        return self.llm_client
    
    async def close(self) -> None:
        """Close LLM client."""
        if self.llm_client:
            await self.llm_client.close()
            self.llm_client = None
    
    def generate_prompts(
        self,
        topic: str,
        arc_type: str = "",
        style_id: str = "oil_painting_cartoon",
        character: str = "",
        artifact: str = "",
        evidence: str = "",
        setting: str = "",
        series_name: str = "",
    ) -> list[dict]:
        """
        Generate 6 prompts using storyboard templates.
        
        Automatically detects characters from the topic using the character library
        for consistent appearance across all scenes.
        
        Args:
            topic: Video topic (e.g., "Da Vinci Mona Lisa")
            arc_type: Narrative arc type ("myth_buster", "historical_figure", "lost_civilization")
                      If empty, will be auto-detected from topic
            style_id: Visual style template ID
            character: Main character description for consistency (auto-detected if empty)
            artifact: Artifact/painting description (for famous_artifact scene)
            evidence: Evidence description (for evidence scene)
            setting: Setting description
            series_name: Optional series name for arc detection
        
        Returns:
            List of prompt dicts for image generation
        """
        # Auto-detect arc type if not specified
        if not arc_type:
            arc_type = detect_arc_type(topic, series_name)
            logger.info(f"Auto-detected arc type: {arc_type}")
        
        # Auto-detect character from topic if not specified
        if not character:
            char_profile = detect_character_from_topic(topic)
            if char_profile:
                character = char_profile.full_description
                logger.info(f"Auto-detected character: {char_profile.name}")
                
                # Also use character's typical setting if not specified
                if not setting and char_profile.typical_setting:
                    setting = char_profile.typical_setting
        
        # Generate prompts using templates
        prompts = get_arc_with_prompts(
            arc_type=arc_type,
            style_id=style_id,
            topic=topic,
            character=character,
            artifact=artifact,
            evidence=evidence,
            setting=setting,
        )
        
        logger.info(f"Generated {len(prompts)} storyboard prompts for: {topic}")
        return prompts
    
    async def generate_prompts_with_llm(
        self,
        topic: str,
        script: str = "",
        arc_type: str = "",
        style_id: str = "oil_painting_cartoon",
        series_name: str = "",
    ) -> list[dict]:
        """
        Generate prompts with LLM enhancement for character and artifact descriptions.
        
        Uses LLM to analyze topic/script and extract character, artifact, and evidence
        descriptions, then feeds them into the template system.
        
        Args:
            topic: Video topic
            script: Optional script for context
            arc_type: Narrative arc type (auto-detected if empty)
            style_id: Visual style template ID
            series_name: Optional series name
        
        Returns:
            List of enhanced prompt dicts
        """
        client = self._ensure_llm_client()
        
        # Auto-detect arc type
        if not arc_type:
            arc_type = detect_arc_type(topic, series_name)
        
        # Use LLM to extract character and artifact descriptions
        character = ""
        artifact = ""
        evidence = ""
        setting = ""
        
        if client.is_configured():
            try:
                extraction_prompt = f"""Analyze this topic and extract visual descriptions for video generation.

TOPIC: {topic}
{f'SCRIPT: {script[:500]}' if script else ''}

Provide a JSON object with these fields:
1. character: Detailed physical description of the main historical figure (age, appearance, clothing, expression)
2. artifact: Description of the famous artifact/painting/creation associated with this topic
3. evidence: What scientific evidence or proof should be shown
4. setting: The historical setting/era/location

Example for "Da Vinci Mona Lisa":
{{
    "character": "aged wise Leonardo da Vinci with long flowing white beard, wearing burgundy velvet robes, intelligent kind eyes, Renaissance master appearance",
    "artifact": "the Mona Lisa painting with her enigmatic smile, sfumato technique visible, mysterious expression",
    "evidence": "facial muscle anatomy diagrams, sfumato technique cross-sections, optical illusion demonstrations",
    "setting": "Renaissance Florence workshop, candlelit studio, parchment and paint"
}}

Output valid JSON only."""

                messages = [
                    {"role": "system", "content": "You extract visual descriptions from topics. Output valid JSON only."},
                    {"role": "user", "content": extraction_prompt},
                ]
                
                response = await client.chat_json(messages, temperature=0.3)
                
                if response:
                    character = response.get("character", "")
                    artifact = response.get("artifact", "")
                    evidence = response.get("evidence", "")
                    setting = response.get("setting", "")
                    logger.info(f"LLM extracted: character='{character[:50]}...'" if character else "LLM extraction empty")
                    
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        # Generate prompts with extracted descriptions
        return self.generate_prompts(
            topic=topic,
            arc_type=arc_type,
            style_id=style_id,
            character=character,
            artifact=artifact,
            evidence=evidence,
            setting=setting,
            series_name=series_name,
        )
    
    def list_available_arcs(self) -> list[str]:
        """List available narrative arc types."""
        return list(NARRATIVE_ARCS.keys())
    
    def list_available_styles(self) -> list[str]:
        """List available style template IDs."""
        return list(STYLE_TEMPLATES.keys())


async def generate_storyboard_prompts(
    topic: str,
    script: str = "",
    arc_type: str = "",
    style_id: str = "oil_painting_cartoon",
    use_llm: bool = True,
) -> list[dict]:
    """
    Convenience function for storyboard-based prompt generation.
    
    Args:
        topic: Video topic
        script: Optional script for context
        arc_type: Narrative arc type (auto-detected if empty)
        style_id: Visual style template ID
        use_llm: Whether to use LLM for character extraction
    
    Returns:
        List of prompt dicts for image generation
    """
    director = StoryboardDirector()
    
    try:
        if use_llm:
            return await director.generate_prompts_with_llm(
                topic=topic,
                script=script,
                arc_type=arc_type,
                style_id=style_id,
            )
        else:
            return director.generate_prompts(
                topic=topic,
                arc_type=arc_type,
                style_id=style_id,
            )
    finally:
        await director.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Director")
    parser.add_argument("--script", "-s", required=True, help="Path to script file")
    parser.add_argument("--topic", "-t", required=True, help="Video topic")
    parser.add_argument("--series", default="History Lies", help="Series name")
    parser.add_argument("--output", "-o", help="Output JSON path")
    
    args = parser.parse_args()
    
    script = Path(args.script).read_text(encoding="utf-8")
    
    async def run():
        prompts = await generate_scene_prompts(
            script=script,
            topic=args.topic,
            series_name=args.series,
        )
        
        print(f"\n=== Visual Prompts for: {args.topic} ===\n")
        for p in prompts:
            print(f"[{p['segment']}]")
            print(f"  Prompt: {p['prompt'][:150]}...")
            print(f"  Fallback: {p['fallback_keyword']}")
            print()
        
        if args.output:
            Path(args.output).write_text(
                json.dumps(prompts, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"Saved to: {args.output}")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
