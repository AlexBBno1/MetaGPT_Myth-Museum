"""
Generate Odyssey "10 Year" Myth Video

Watercolor fantasy style illustrations for the myth about
Odysseus being "lost" for 10 years.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
from pipeline.tts import GoogleTTSProvider, get_audio_duration
from pipeline.generate_short import ShortVideoGenerator, GenerationConfig


# Watercolor fantasy style prefix
STYLE_PREFIX = "Dreamy watercolor illustration, soft fantasy art style, mythological storybook aesthetic"
STYLE_SUFFIX = "ocean blues, sunset oranges, soft greens, ethereal whites, visible brush strokes, wet-on-wet blending, paper grain texture"

# Scene prompts for 6 images
SCENE_PROMPTS = [
    # Scene 1: Hook - Odysseus's ship in stormy seas
    {
        "scene": "hook",
        "description": "Odysseus ship in dramatic stormy seas",
        "prompt": f"{STYLE_PREFIX}, ancient Greek wooden ship tossed by massive waves in violent storm, lightning strikes illuminating dark clouds, hero figure gripping the mast, turbulent sea foam and spray, dramatic diagonal composition, {STYLE_SUFFIX}",
        "fallback_keyword": "greek ship storm watercolor",
    },
    # Scene 2: Setup - Classic "lost hero" imagery
    {
        "scene": "setup",
        "description": "Odysseus on ship deck looking lost",
        "prompt": f"{STYLE_PREFIX}, Greek warrior Odysseus standing on ship deck studying ancient map with confused expression, vast empty ocean stretching to horizon, lonely figure against endless sea, muted blue-gray color palette, melancholic atmosphere, {STYLE_SUFFIX}",
        "fallback_keyword": "odysseus ship map watercolor",
    },
    # Scene 3: Twist - Calypso's paradise island
    {
        "scene": "twist",
        "description": "Calypso's magical paradise island Ogygia",
        "prompt": f"{STYLE_PREFIX}, lush tropical paradise island Ogygia, beautiful immortal nymph Calypso goddess silhouette standing by crystal clear turquoise waters, exotic flowers and waterfalls, ethereal golden light filtering through palm trees, dreamlike magical atmosphere, {STYLE_SUFFIX}",
        "fallback_keyword": "paradise island goddess watercolor",
    },
    # Scene 4: Evidence - Odysseus weeping on the shore
    {
        "scene": "evidence",
        "description": "Odysseus weeping on the beach looking toward home",
        "prompt": f"{STYLE_PREFIX}, lonely figure of Odysseus sitting on rocky shore at sunset, tears streaming down weathered face, gazing toward distant horizon, warm orange and purple sunset colors, melancholic emotional scene, longing for home, {STYLE_SUFFIX}",
        "fallback_keyword": "man crying beach sunset watercolor",
    },
    # Scene 5: Zeus intervention
    {
        "scene": "intervention",
        "description": "Zeus on Mount Olympus commanding Calypso",
        "prompt": f"{STYLE_PREFIX}, mighty Zeus king of gods on Mount Olympus throne among clouds, golden thunderbolt in hand, commanding divine authority, celestial rays of light, majestic white robes and silver beard, dramatic heavenly atmosphere, {STYLE_SUFFIX}",
        "fallback_keyword": "zeus god olympus clouds watercolor",
    },
    # Scene 6: Conclusion - Odysseus sailing home
    {
        "scene": "conclusion",
        "description": "Odysseus finally sailing toward Ithaca",
        "prompt": f"{STYLE_PREFIX}, small Greek sailing boat on calm peaceful waters at dawn, hero Odysseus at helm looking toward distant island of Ithaca on horizon, hopeful golden sunrise, peaceful resolution, journey's end approaching, {STYLE_SUFFIX}",
        "fallback_keyword": "boat sailing sunrise island watercolor",
    },
]


async def generate_odyssey_images():
    """Generate all 6 images for the Odyssey video."""
    output_dir = Path("outputs/shorts/greek-myths_odyssey-10years")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visual briefs
    briefs_path = output_dir / "visual_briefs.json"
    with open(briefs_path, "w", encoding="utf-8") as f:
        json.dump(SCENE_PROMPTS, f, indent=2, ensure_ascii=False)
    print(f"Saved visual briefs to {briefs_path}")
    
    # Initialize image provider
    provider = ImageProviderWithFallback()
    
    images = []
    for i, scene in enumerate(SCENE_PROMPTS, 1):
        image_path = output_dir / f"image_{i:02d}.jpg"
        
        # Skip if already exists
        if image_path.exists():
            print(f"\n[{i}/6] SKIP (exists): {scene['description']}")
            images.append(image_path)
            continue
        
        print(f"\n[{i}/6] Generating: {scene['description']}")
        
        try:
            result = await provider.generate_image(
                prompt=scene["prompt"],
                output_path=image_path,
                aspect_ratio="9:16",
                quality="high",
            )
            
            if result and image_path.exists():
                print(f"  [OK] Saved: {image_path}")
                images.append(image_path)
            else:
                print(f"  [FAIL] Failed to generate image {i}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print(f"\nGenerated {len(images)}/6 images")
    return images


async def generate_tts():
    """Generate TTS audio from script."""
    output_dir = Path("outputs/shorts/greek-myths_odyssey-10years")
    script_path = output_dir / "voiceover.txt"
    audio_path = output_dir / "voiceover.mp3"
    srt_path = output_dir / "captions.srt"
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return None
    
    script = script_path.read_text(encoding="utf-8")
    
    print("\nGenerating TTS audio...")
    tts = GoogleTTSProvider()
    
    # Google TTS automatically generates captions.srt alongside the audio
    await tts.synthesize(
        text=script,
        output_path=audio_path,
        voice="en-US-Casual-K",
        rate="+0%",
    )
    
    duration = get_audio_duration(audio_path)
    print(f"  [OK] Audio: {audio_path} ({duration:.1f}s)")
    
    if srt_path.exists():
        print(f"  [OK] Captions: {srt_path}")
    
    return audio_path, duration


async def render_video():
    """Render the final video."""
    output_dir = Path("outputs/shorts/greek-myths_odyssey-10years")
    
    # Get images
    images = sorted(output_dir.glob("image_*.jpg"))
    if len(images) < 6:
        print(f"Not enough images: {len(images)}/6")
        return None
    
    # Get audio
    audio_path = output_dir / "voiceover.mp3"
    if not audio_path.exists():
        print("Audio not found")
        return None
    
    duration = get_audio_duration(audio_path)
    
    # Create generator
    generator = ShortVideoGenerator()
    
    # Create ASS subtitles
    srt_path = output_dir / "captions.srt"
    ass_path = output_dir / "captions.ass"
    
    print("\nCreating subtitles...")
    generator._create_ass_with_series_marker(
        srt_path=srt_path,
        ass_path=ass_path,
        series_text="Greek Myths",
        duration=duration,
        subtitle_style="punch",
    )
    print(f"  [OK] ASS subtitles: {ass_path}")
    
    # Create slideshow
    slideshow_path = output_dir / "background.mp4"
    print("\nCreating slideshow...")
    generator._create_slideshow(
        images=list(images),
        output_path=slideshow_path,
        duration=duration,
        use_ken_burns=True,
        use_crossfade=True,
    )
    print(f"  [OK] Slideshow: {slideshow_path}")
    
    # Render final video
    final_path = output_dir / "final.mp4"
    print("\nRendering final video...")
    generator._render_final(
        slideshow_path=slideshow_path,
        audio_path=audio_path,
        ass_path=ass_path,
        output_path=final_path,
    )
    
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Video complete: {final_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return final_path
    else:
        print("[FAIL] Video render failed")
        return None


async def main():
    print("=" * 60)
    print("ODYSSEY '10 YEAR' MYTH VIDEO GENERATOR")
    print("=" * 60)
    
    # Step 1: Generate images
    print("\n[PHASE 1] Generating watercolor fantasy images...")
    images = await generate_odyssey_images()
    
    if len(images) < 6:
        print("Not enough images generated. Aborting.")
        return
    
    # Step 2: Generate TTS
    print("\n[PHASE 2] Generating TTS audio...")
    result = await generate_tts()
    if not result:
        print("TTS generation failed. Aborting.")
        return
    
    # Step 3: Render video
    print("\n[PHASE 3] Rendering final video...")
    video = await render_video()
    
    if video:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Output: {video}")
        print("=" * 60)
        
        # Open video
        import subprocess
        subprocess.run(["start", str(video)], shell=True)


if __name__ == "__main__":
    asyncio.run(main())
