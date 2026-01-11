"""
Generate Ninja Myth Video - "Ninjas Never Wore Black"

This script generates a YouTube Short about the myth that ninjas wore all black,
revealing the truth that real ninjas dressed as ordinary people.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from pipeline.generate_short import ShortVideoGenerator, GenerationConfig


# ============================================================================
# Video Configuration
# ============================================================================

TOPIC = "Ninja Black Clothing Myth"

SCRIPT = """Think ninjas wore black? Hollywood lied to you.

We've all seen it - the iconic ninja in black, sneaking through shadows, invisible in the night.

But here's the thing - pure black actually stands OUT in moonlight. Try it yourself. Dark blue or gray blends better with night shadows.

So where did the black ninja come from? Kabuki theater. Stagehands wore black to be 'invisible' to audiences. When a ninja character appeared from nowhere, they used the same trick. Audiences loved it - and the myth was born.

Real ninjas? They were spies. They dressed as farmers, merchants, traveling monks - anyone who wouldn't attract attention. The best disguise is no disguise at all.

So the next time you see a ninja in black... remember, if you can see them, they're not a very good ninja."""

# Custom image prompts with Ukiyo-e style
IMAGE_PROMPTS = [
    {
        "segment": "bg_1",
        "prompt": (
            "Traditional Japanese ukiyo-e woodblock print style, dramatic black ninja silhouette "
            "against giant full moon, Japanese castle in background, bold black outlines, "
            "indigo blue sky, gold and cream colors, Hokusai wave style clouds, "
            "large red X mark overlay crossing out the ninja, 9:16 vertical, no text"
        ),
        "fallback_keyword": "ninja silhouette moon japan"
    },
    {
        "segment": "bg_2",
        "prompt": (
            "Ukiyo-e woodblock print style, classic ninja in black outfit crouching on "
            "traditional Japanese rooftop tiles, moonlit Edo castle background, "
            "cherry blossom petals falling, bold black outlines, dramatic pose, "
            "flat indigo and vermillion colors, traditional Japanese aesthetic, "
            "9:16 vertical, no text"
        ),
        "fallback_keyword": "ninja rooftop japan"
    },
    {
        "segment": "bg_3",
        "prompt": (
            "Ukiyo-e woodblock print style, split image comparison diagram, "
            "left side shows pure black figure standing out against moonlit night sky, "
            "right side shows dark blue figure blending into shadows, "
            "educational illustration style, Japanese wave border design, "
            "indigo and cream colors, 9:16 vertical, no text"
        ),
        "fallback_keyword": "ninja comparison night"
    },
    {
        "segment": "bg_4",
        "prompt": (
            "Traditional ukiyo-e theater scene, Kabuki stage with colorful actors in "
            "elaborate kimono costumes, black-clad kuroko stagehands visible, "
            "one stagehand dramatically revealing as ninja character, "
            "audience watching in amazement, paper lantern lighting, "
            "vibrant vermillion red and gold colors, 9:16 vertical, no text"
        ),
        "fallback_keyword": "kabuki theater japan"
    },
    {
        "segment": "bg_5",
        "prompt": (
            "Bustling Edo period marketplace ukiyo-e woodblock print style, "
            "merchants farmers and monks walking among crowds, "
            "subtle red circles highlighting 5 disguised ninjas dressed as ordinary people, "
            "hidden weapons barely visible in bags and sleeves, "
            "busy composition, Hiroshige style, indigo and ochre colors, "
            "9:16 vertical, no text"
        ),
        "fallback_keyword": "edo market japan crowd"
    },
    {
        "segment": "bg_6",
        "prompt": (
            "Modern Tokyo street ukiyo-e fusion style, contemporary Japanese person "
            "in casual modern clothes giving subtle knowing look to viewer, "
            "traditional ukiyo-e wave patterns blended into modern cityscape background, "
            "ninja shuriken barely visible peeking from messenger bag, "
            "blend of traditional indigo blue with modern neon accents, "
            "9:16 vertical, no text"
        ),
        "fallback_keyword": "tokyo modern japan"
    },
]


async def main():
    """Generate the ninja myth video."""
    print("=" * 60)
    print("Generating: Ninja Black Clothing Myth")
    print("Style: Ukiyo-e (Japanese Woodblock Print)")
    print("Series: Japan Myths #1")
    print("=" * 60)
    print()
    
    # Create generator
    generator = ShortVideoGenerator()
    
    # Create config
    config = GenerationConfig(
        topic=TOPIC,
        script=SCRIPT,
        image_quality="high",
        subtitle_style="punch",
        auto_prompts=False,  # Use custom prompts
    )
    
    # Override with custom prompts
    config.custom_prompts = IMAGE_PROMPTS
    
    # Generate
    print("Starting video generation...")
    print()
    
    result = await generator.generate(config)
    
    if result.success:
        print()
        print("=" * 60)
        print("SUCCESS!")
        print(f"Video: {result.final_video}")
        print(f"Folder: {result.output_folder}")
        print("=" * 60)
        
        # Rename folder to match naming convention
        old_folder = result.output_folder
        new_name = "japan-myths_ninja-black"
        new_folder = old_folder.parent / new_name
        
        if old_folder != new_folder and not new_folder.exists():
            print(f"\nRenaming folder to: {new_name}")
            old_folder.rename(new_folder)
            print(f"Final location: {new_folder}")
    else:
        print()
        print("=" * 60)
        print(f"FAILED: {result.error}")
        print("=" * 60)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
