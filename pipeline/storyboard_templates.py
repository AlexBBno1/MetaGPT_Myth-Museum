"""
Myth Museum - Storyboard Templates

Defines narrative arcs and visual style templates for consistent,
high-quality video generation.

Usage:
    from pipeline.storyboard_templates import (
        get_narrative_arc,
        get_style_template,
        build_scene_prompt,
    )
    
    arc = get_narrative_arc("myth_buster")
    style = get_style_template("oil_painting_cartoon")
    prompt = build_scene_prompt(arc[0], style, topic="Da Vinci")
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Narrative Arcs - 6-Image Story Structures
# ============================================================================

@dataclass
class SceneTemplate:
    """Template for a single scene in the narrative arc."""
    scene_id: str
    name: str
    description: str
    camera: str  # wide, medium, close-up, extreme close-up
    mood: str
    visual_focus: str  # What should be the main focus
    prompt_template: str  # Template with {topic}, {character}, {style} placeholders


# Myth Buster Arc - For debunking common myths
MYTH_BUSTER_ARC = [
    SceneTemplate(
        scene_id="mystery_intro",
        name="Mystery Intro",
        description="Set up the mystery, hint at secrets",
        camera="wide silhouette",
        mood="mysterious, suspenseful",
        visual_focus="atmosphere and intrigue",
        prompt_template=(
            "{style}, mysterious {setting} shrouded in shadows, "
            "silhouette of {character} against candlelight, "
            "floating cryptic symbols and ancient texts, "
            "dramatic chiaroscuro lighting, sepia and gold tones, "
            "conspiracy theory atmosphere, 9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="famous_artifact",
        name="Famous Artifact",
        description="Close-up of the famous artifact/painting/subject",
        camera="extreme close-up",
        mood="reverent, detailed",
        visual_focus="the iconic subject matter",
        prompt_template=(
            "Museum quality close-up of {artifact}, "
            "dramatic spotlight lighting, visible brushstrokes and texture, "
            "photorealistic detail, soft vignette edges, "
            "art gallery atmosphere, cinematic composition, "
            "9:16 vertical, ultra high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="the_research",
        name="The Research",
        description="The character studying/researching",
        camera="medium shot",
        mood="scholarly, focused",
        visual_focus="the research process",
        prompt_template=(
            "{style}, {character} in Renaissance workshop, "
            "surrounded by research materials and diagrams, "
            "warm candlelight illumination, scholarly atmosphere, "
            "parchment papers and quill pens scattered, "
            "focused intellectual expression, sepia tones, "
            "9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="evidence",
        name="The Evidence",
        description="Presenting the scientific evidence",
        camera="close-up detail",
        mood="revealing, educational",
        visual_focus="the proof/evidence",
        prompt_template=(
            "Detailed close-up of {evidence}, "
            "scientific diagram style, clear annotations visible, "
            "educational illustration quality, "
            "soft directional lighting, clean composition, "
            "documentary photography aesthetic, "
            "9:16 vertical, ultra sharp detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="revelation",
        name="The Revelation",
        description="The 'aha' moment of revelation",
        camera="medium shot",
        mood="triumphant, enlightening",
        visual_focus="the moment of understanding",
        prompt_template=(
            "{style}, {character} with eureka expression, "
            "bright revealing light breaking through shadows, "
            "magnifying glass revealing truth, "
            "brain and optical diagrams floating nearby, "
            "warm golden tones replacing dark shadows, "
            "triumphant intellectual moment, "
            "9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="victory",
        name="Victory Conclusion",
        description="Triumphant conclusion, myth busted",
        camera="wide heroic",
        mood="victorious, conclusive",
        visual_focus="the triumphant character with discoveries",
        prompt_template=(
            "{style}, triumphant {character} standing proudly, "
            "surrounded by inventions and discoveries, "
            "myth symbols crossed out on one side, "
            "scientific achievements glowing on other side, "
            "golden victorious lighting, heroic composition, "
            "Renaissance master pose, "
            "9:16 vertical, high detail, no text"
        ),
    ),
]

# Historical Figure Arc - For biographical content
HISTORICAL_FIGURE_ARC = [
    SceneTemplate(
        scene_id="iconic_portrait",
        name="Iconic Portrait",
        description="Iconic portrait of the figure",
        camera="portrait close-up",
        mood="dignified, timeless",
        visual_focus="the person's iconic appearance",
        prompt_template=(
            "{style}, iconic portrait of {character}, "
            "dignified pose, period-accurate clothing, "
            "soft Rembrandt lighting, warm tones, "
            "museum quality painting style, "
            "9:16 vertical, masterpiece quality, no text"
        ),
    ),
    SceneTemplate(
        scene_id="early_life",
        name="Early Life",
        description="Scene from early life/origin",
        camera="wide establishing",
        mood="nostalgic, formative",
        visual_focus="the environment that shaped them",
        prompt_template=(
            "{style}, young {character} in formative setting, "
            "period-accurate environment, soft nostalgic lighting, "
            "hints of future greatness, "
            "warm sepia tones, documentary style, "
            "9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="defining_moment",
        name="Defining Moment",
        description="The defining moment/achievement",
        camera="dramatic medium",
        mood="intense, pivotal",
        visual_focus="the key achievement",
        prompt_template=(
            "{style}, {character} at pivotal moment, "
            "dramatic lighting emphasizing importance, "
            "historical accuracy, intense atmosphere, "
            "documentary photography quality, "
            "9:16 vertical, cinematic, no text"
        ),
    ),
    SceneTemplate(
        scene_id="at_work",
        name="At Work",
        description="The figure at work in their element",
        camera="medium environmental",
        mood="focused, authentic",
        visual_focus="their craft/profession",
        prompt_template=(
            "{style}, {character} deeply focused on work, "
            "authentic period workspace, tools of trade visible, "
            "warm working atmosphere, natural lighting, "
            "documentary intimacy, professional dedication, "
            "9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="legacy_impact",
        name="Legacy Impact",
        description="Their lasting impact on history",
        camera="wide symbolic",
        mood="grand, impactful",
        visual_focus="the scope of their influence",
        prompt_template=(
            "{style}, symbolic representation of {character}'s legacy, "
            "their creations and influence spreading outward, "
            "epic scale, golden hour lighting, "
            "timeless monumentality, "
            "9:16 vertical, grand composition, no text"
        ),
    ),
    SceneTemplate(
        scene_id="remembered",
        name="Remembered Forever",
        description="How they are remembered today",
        camera="wide reverent",
        mood="reverent, eternal",
        visual_focus="their enduring memory",
        prompt_template=(
            "{style}, {character} immortalized in history, "
            "surrounded by symbols of their achievements, "
            "ethereal golden glow, timeless composition, "
            "legendary status visualized, "
            "9:16 vertical, masterpiece quality, no text"
        ),
    ),
]

# Lost Civilization Arc - For ancient history content
LOST_CIVILIZATION_ARC = [
    SceneTemplate(
        scene_id="ruins_discovery",
        name="Ruins Discovery",
        description="Discovery of ancient ruins",
        camera="wide establishing",
        mood="mysterious, awe-inspiring",
        visual_focus="the scale of the ruins",
        prompt_template=(
            "Majestic ancient {civilization} ruins emerging from jungle, "
            "golden hour lighting through vegetation, "
            "dramatic scale, explorer silhouette for scale, "
            "mysterious atmosphere, archaeological discovery moment, "
            "9:16 vertical, cinematic, no text"
        ),
    ),
    SceneTemplate(
        scene_id="civilization_glory",
        name="Civilization at Peak",
        description="The civilization at its peak",
        camera="wide panoramic",
        mood="grand, prosperous",
        visual_focus="the thriving ancient city",
        prompt_template=(
            "Ancient {civilization} city at peak glory, "
            "bustling markets and grand temples, "
            "golden sunlight, vibrant colors, "
            "historically accurate architecture, "
            "epic scale, documentary style, "
            "9:16 vertical, high detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="daily_life",
        name="Daily Life",
        description="Daily life of ancient people",
        camera="medium intimate",
        mood="authentic, human",
        visual_focus="relatable human moments",
        prompt_template=(
            "Daily life scene in ancient {civilization}, "
            "authentic period clothing and activities, "
            "warm intimate lighting, human connection, "
            "documentary photography style, "
            "9:16 vertical, realistic detail, no text"
        ),
    ),
    SceneTemplate(
        scene_id="sacred_ritual",
        name="Sacred Ritual",
        description="Sacred rituals and beliefs",
        camera="medium dramatic",
        mood="mystical, sacred",
        visual_focus="spiritual practices",
        prompt_template=(
            "Sacred ritual in ancient {civilization} temple, "
            "dramatic torchlight and shadows, "
            "mystical atmosphere, priest figures, "
            "authentic ceremonial objects, "
            "9:16 vertical, atmospheric, no text"
        ),
    ),
    SceneTemplate(
        scene_id="decline",
        name="The Decline",
        description="The decline/fall",
        camera="wide somber",
        mood="melancholic, dramatic",
        visual_focus="the end of an era",
        prompt_template=(
            "Ancient {civilization} in decline, "
            "abandoned buildings, nature reclaiming, "
            "dramatic storm clouds, somber lighting, "
            "bittersweet atmosphere, "
            "9:16 vertical, emotional, no text"
        ),
    ),
    SceneTemplate(
        scene_id="modern_mystery",
        name="Modern Mystery",
        description="The mystery that remains today",
        camera="wide mysterious",
        mood="intriguing, timeless",
        visual_focus="unanswered questions",
        prompt_template=(
            "Modern archaeologist studying {civilization} artifacts, "
            "ancient symbols and modern technology contrast, "
            "mysterious lighting, questions still unanswered, "
            "documentary style, sense of wonder, "
            "9:16 vertical, cinematic, no text"
        ),
    ),
]

# Dictionary of all narrative arcs
NARRATIVE_ARCS = {
    "myth_buster": MYTH_BUSTER_ARC,
    "historical_figure": HISTORICAL_FIGURE_ARC,
    "lost_civilization": LOST_CIVILIZATION_ARC,
}


# ============================================================================
# Visual Style Templates
# ============================================================================

@dataclass
class StyleTemplate:
    """Visual style template for consistent aesthetics."""
    style_id: str
    name: str
    description: str
    style_prefix: str  # Added to beginning of prompts
    color_palette: str
    lighting: str
    texture: str


STYLE_TEMPLATES = {
    "oil_painting_cartoon": StyleTemplate(
        style_id="oil_painting_cartoon",
        name="Oil Painting Cartoon",
        description="Renaissance oil painting style cartoon, warm and scholarly",
        style_prefix="Oil painting style cartoon, Renaissance aesthetic",
        color_palette="warm sepia tones, burgundy, gold accents, aged parchment yellows",
        lighting="warm candlelight, soft chiaroscuro, golden hour glow",
        texture="visible brushstrokes, canvas texture, aged patina",
    ),
    "realistic": StyleTemplate(
        style_id="realistic",
        name="Photorealistic",
        description="Photorealistic, museum quality documentary style",
        style_prefix="Photorealistic, museum quality photography",
        color_palette="natural colors, rich earth tones, subtle contrast",
        lighting="cinematic lighting, soft shadows, natural light",
        texture="high detail, sharp focus, documentary quality",
    ),
    "flat_illustration": StyleTemplate(
        style_id="flat_illustration",
        name="Flat Illustration",
        description="Clean flat illustration, educational cartoon",
        style_prefix="Flat vector illustration, clean educational cartoon",
        color_palette="bright primary colors, clean whites, bold accents",
        lighting="flat even lighting, no harsh shadows",
        texture="smooth gradients, clean edges, minimal texture",
    ),
    "pixar_3d": StyleTemplate(
        style_id="pixar_3d",
        name="Pixar 3D Style",
        description="Pixar-style 3D cartoon, expressive and polished",
        style_prefix="Pixar-style 3D cartoon, polished render",
        color_palette="vibrant saturated colors, warm highlights, cool shadows",
        lighting="studio lighting, rim lights, soft ambient occlusion",
        texture="smooth subsurface scattering, clean plastic-like surfaces",
    ),
    "watercolor": StyleTemplate(
        style_id="watercolor",
        name="Watercolor",
        description="Soft watercolor illustration, dreamy and artistic",
        style_prefix="Soft watercolor illustration, dreamy artistic style",
        color_palette="muted pastels, soft washes, gentle gradients",
        lighting="soft diffused light, no harsh shadows",
        texture="visible watercolor bleeding, paper texture, organic edges",
    ),
    "cinematic": StyleTemplate(
        style_id="cinematic",
        name="Cinematic",
        description="Hollywood cinematic style, dramatic and polished",
        style_prefix="Cinematic movie still, Hollywood production quality",
        color_palette="teal and orange, rich blacks, dramatic contrast",
        lighting="dramatic three-point lighting, lens flares, volumetric fog",
        texture="film grain, shallow depth of field, anamorphic bokeh",
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_narrative_arc(arc_type: str) -> list[SceneTemplate]:
    """
    Get a narrative arc by type.
    
    Args:
        arc_type: One of "myth_buster", "historical_figure", "lost_civilization"
    
    Returns:
        List of SceneTemplate objects
    """
    return NARRATIVE_ARCS.get(arc_type, MYTH_BUSTER_ARC)


def get_style_template(style_id: str) -> StyleTemplate:
    """
    Get a style template by ID.
    
    Args:
        style_id: One of the style IDs (e.g., "oil_painting_cartoon")
    
    Returns:
        StyleTemplate object
    """
    return STYLE_TEMPLATES.get(style_id, STYLE_TEMPLATES["oil_painting_cartoon"])


def detect_arc_type(topic: str, series_name: str = "") -> str:
    """
    Detect the best narrative arc type based on topic and series.
    
    Args:
        topic: The video topic
        series_name: Optional series name for context
    
    Returns:
        Arc type string
    """
    topic_lower = topic.lower()
    series_lower = series_name.lower() if series_name else ""
    
    # Lost civilization keywords
    civ_keywords = ["aztec", "maya", "mayan", "egypt", "egyptian", "inca", "sumerian", 
                    "babylon", "mesopotamia", "civilization", "ancient", "ruins"]
    if any(kw in topic_lower or kw in series_lower for kw in civ_keywords):
        return "lost_civilization"
    
    # Historical figure keywords (names of famous people)
    figure_keywords = ["da vinci", "leonardo", "beethoven", "mozart", "napoleon", 
                       "cleopatra", "caesar", "einstein", "newton", "darwin"]
    if any(kw in topic_lower for kw in figure_keywords):
        return "historical_figure"
    
    # Default to myth buster
    return "myth_buster"


def build_scene_prompt(
    scene: SceneTemplate,
    style: StyleTemplate,
    topic: str,
    character: str = "",
    artifact: str = "",
    evidence: str = "",
    setting: str = "",
    civilization: str = "",
) -> str:
    """
    Build a complete prompt from scene and style templates.
    
    Args:
        scene: SceneTemplate to use
        style: StyleTemplate to apply
        topic: Video topic
        character: Main character description
        artifact: Artifact/painting description (for famous_artifact scene)
        evidence: Evidence description (for evidence scene)
        setting: Setting description
        civilization: Civilization name (for lost_civilization arc)
    
    Returns:
        Complete prompt string
    """
    # Default values
    if not character:
        character = f"scholarly figure related to {topic}"
    if not artifact:
        artifact = f"famous artifact or artwork related to {topic}"
    if not evidence:
        evidence = f"scientific evidence and diagrams about {topic}"
    if not setting:
        setting = f"historical setting for {topic}"
    if not civilization:
        civilization = topic
    
    # Build prompt from template
    prompt = scene.prompt_template.format(
        style=style.style_prefix,
        topic=topic,
        character=character,
        artifact=artifact,
        evidence=evidence,
        setting=setting,
        civilization=civilization,
    )
    
    return prompt


def get_arc_with_prompts(
    arc_type: str,
    style_id: str,
    topic: str,
    character: str = "",
    artifact: str = "",
    evidence: str = "",
    setting: str = "",
    civilization: str = "",
) -> list[dict]:
    """
    Get a complete narrative arc with all prompts filled in.
    
    Returns a list of dicts ready for image generation:
    [{"segment": "bg_1", "prompt": "...", "fallback_keyword": "..."}]
    
    Args:
        arc_type: Narrative arc type
        style_id: Style template ID
        topic: Video topic
        character: Main character description
        artifact: Artifact description
        evidence: Evidence description
        setting: Setting description
        civilization: Civilization name
    
    Returns:
        List of prompt dicts for image generation
    """
    arc = get_narrative_arc(arc_type)
    style = get_style_template(style_id)
    
    prompts = []
    for i, scene in enumerate(arc):
        prompt = build_scene_prompt(
            scene=scene,
            style=style,
            topic=topic,
            character=character,
            artifact=artifact,
            evidence=evidence,
            setting=setting,
            civilization=civilization,
        )
        
        prompts.append({
            "segment": f"bg_{i+1}",
            "prompt": prompt,
            "fallback_keyword": f"{topic} {scene.visual_focus}",
            "scene_name": scene.name,
            "scene_id": scene.scene_id,
        })
    
    return prompts


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    """Test the storyboard templates."""
    import json
    
    print("=== Storyboard Templates Test ===\n")
    
    # Test myth buster arc
    prompts = get_arc_with_prompts(
        arc_type="myth_buster",
        style_id="oil_painting_cartoon",
        topic="Da Vinci Mona Lisa",
        character="aged wise Leonardo da Vinci with long flowing white beard, wearing burgundy velvet robes",
        artifact="the Mona Lisa painting with her enigmatic smile",
        evidence="facial muscle anatomy diagrams and sfumato technique analysis",
        setting="Renaissance Florence workshop",
    )
    
    print("Myth Buster Arc - Da Vinci Example:\n")
    for p in prompts:
        print(f"[{p['segment']}] {p['scene_name']}")
        print(f"  Prompt: {p['prompt'][:100]}...")
        print()
    
    print("\n=== Available Arcs ===")
    for arc_type in NARRATIVE_ARCS:
        print(f"  - {arc_type}: {len(NARRATIVE_ARCS[arc_type])} scenes")
    
    print("\n=== Available Styles ===")
    for style_id, style in STYLE_TEMPLATES.items():
        print(f"  - {style_id}: {style.name}")


if __name__ == "__main__":
    main()
