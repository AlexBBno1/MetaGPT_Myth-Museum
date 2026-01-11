"""
Generate Ninja Myth Video V2 - Extended 70 Second Version

This script:
1. Uses kept images from v1 (bg_1, bg_3)
2. Generates 4 new ukiyo-e style images
3. Creates extended 70-second video
"""

import asyncio
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from pipeline.generate_short import ShortVideoGenerator, GenerationConfig
from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback


# ============================================================================
# Video Configuration
# ============================================================================

TOPIC = "Ninja Black Clothing Myth"
OUTPUT_FOLDER = Path("outputs/shorts/japan-myths_ninja-black-v2")

# Extended script (~175 words for 70 seconds)
SCRIPT = """Think ninjas wore black? That's one of the biggest lies in history. And Hollywood made sure you believed it.

For decades, movies showed us the same image - a shadowy figure in black, leaping across rooftops, invisible in the darkness. The perfect assassin.

But here's what they got wrong. Pure black doesn't hide you in moonlight - it makes you MORE visible. Dark blue or gray actually blends with shadows. Any real spy would know this.

So where did the black ninja come from? Kabuki theater. In Japanese theater, stagehands wore all black - audiences were trained to 'not see' them. When playwrights needed a ninja to appear from nowhere, they used the same trick. Audiences gasped. The myth was born.

Real ninjas were intelligence agents. They dressed as farmers, merchants, monks, even samurai. The best ninja looked exactly like everyone else. Their greatest weapon wasn't a sword - it was anonymity.

So the next time you see a ninja in black... remember, that's just theater. A real ninja could be standing right next to you. And if you could see them? They wouldn't be very good at their job."""

# New image prompts (only for scenes 2, 4, 5, 6)
NEW_IMAGE_PROMPTS = {
    "bg_2": (
        "Traditional Japanese ukiyo-e woodblock print style, dramatic ninja silhouette "
        "crouching on Japanese castle rooftop against giant full moon, "
        "Edo period castle architecture, cherry blossom petals in wind, "
        "bold black outlines, indigo night sky, gold and cream accents, "
        "Hokusai and Hiroshige inspired, flat colors, 9:16 vertical, no text"
    ),
    "bg_4": (
        "Traditional ukiyo-e woodblock print style, Kabuki theater stage scene, "
        "colorful actors in elaborate kimono costumes performing, "
        "black-clad kuroko stagehands visible moving props, "
        "one stagehand dramatically transforming into ninja character, "
        "paper lantern lighting, vermillion red curtains, gold details, "
        "audience silhouettes watching, Japanese theatrical atmosphere, "
        "bold outlines, flat colors, 9:16 vertical, no text"
    ),
    "bg_5": (
        "Ukiyo-e woodblock print style, bustling Edo period Japanese marketplace, "
        "crowds of merchants farmers monks samurai walking, "
        "several ordinary-looking people subtly revealed as disguised ninjas, "
        "one farmer with hidden blade in basket, one monk with concealed weapons, "
        "busy composition, Hiroshige style, indigo and ochre warm colors, "
        "traditional Japanese architecture, market stalls, "
        "9:16 vertical, no text"
    ),
    "bg_6": (
        "Modern Tokyo street scene with ukiyo-e artistic influence, "
        "contemporary Japanese businessman in suit walking in crowd, "
        "subtle knowing confident expression looking at viewer, "
        "barely visible ninja shuriken shape in briefcase shadow, "
        "blend of traditional ukiyo-e wave patterns with modern city elements, "
        "neon signs with traditional indigo blue color palette, "
        "cinematic depth of field, mysterious atmosphere, "
        "9:16 vertical, no text"
    ),
}


async def generate_new_images():
    """Generate the 4 new images."""
    print("=" * 60)
    print("Generating 4 New Images for Ninja V2")
    print("=" * 60)
    print()
    
    # Create backgrounds folder
    bg_folder = OUTPUT_FOLDER / "backgrounds"
    bg_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize image provider
    provider = ImageProviderWithFallback()
    
    for segment, prompt in NEW_IMAGE_PROMPTS.items():
        print(f"\n[{segment}] Generating...")
        output_path = bg_folder / f"{segment}.jpg"
        
        result = await provider.generate_image(
            prompt=prompt,
            output_path=output_path,
            fallback_keyword=f"ninja japan {segment}",
            quality="high",
        )
        
        if result and output_path.exists():
            print(f"  OK: {output_path}")
        else:
            print(f"  FAILED: {segment}")
    
    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("=" * 60)


def organize_images():
    """Organize all 6 images into correct order."""
    print("\nOrganizing images...")
    
    bg_folder = OUTPUT_FOLDER / "backgrounds"
    
    # Copy kept images to correct positions
    # Scene 1 (keep_scene1) -> bg_1
    src1 = bg_folder / "keep_scene1.jpg"
    dst1 = bg_folder / "bg_1.jpg"
    if src1.exists():
        shutil.copy(src1, dst1)
        print(f"  bg_1: Kept from v1 (ninja face)")
    
    # Scene 3 (keep_scene3) -> bg_3
    src3 = bg_folder / "keep_scene3.jpg"
    dst3 = bg_folder / "bg_3.jpg"
    if src3.exists():
        shutil.copy(src3, dst3)
        print(f"  bg_3: Kept from v1 (city ninja)")
    
    # New images should already be in place (bg_2, bg_4, bg_5, bg_6)
    for i in [2, 4, 5, 6]:
        path = bg_folder / f"bg_{i}.jpg"
        if path.exists():
            print(f"  bg_{i}: New ukiyo-e image")
        else:
            print(f"  bg_{i}: MISSING!")
    
    # Cleanup temp files
    for temp in ["keep_scene1.jpg", "keep_scene3.jpg"]:
        temp_path = bg_folder / temp
        if temp_path.exists():
            temp_path.unlink()
    
    print("\nImage organization complete!")


async def generate_video():
    """Generate the final video."""
    print("\n" + "=" * 60)
    print("Generating Final Video")
    print("=" * 60)
    
    # Create generator
    generator = ShortVideoGenerator()
    
    # Create config with custom output folder
    config = GenerationConfig(
        topic=TOPIC,
        script=SCRIPT,
        image_quality="high",
        subtitle_style="punch",
        auto_prompts=False,
    )
    
    # We need to use the existing images, so we'll manually run the pipeline steps
    from pipeline.tts import GoogleTTSProvider, get_audio_duration
    from pipeline.shorts_optimizer import ShortsOptimizer
    
    # Get series info
    optimizer = ShortsOptimizer()
    series_info = optimizer.get_series_info(TOPIC, increment=True)
    print(f"\nSeries: {series_info.display_text}")
    
    # Save series info
    import json
    series_path = OUTPUT_FOLDER / "series_info.json"
    with open(series_path, "w", encoding="utf-8") as f:
        json.dump({
            "series_name": series_info.series_name,
            "episode_number": series_info.episode_number,
            "display_text": series_info.display_text,
        }, f, indent=2)
    
    # Generate TTS
    print("\nGenerating TTS...")
    tts = GoogleTTSProvider()
    audio_path = OUTPUT_FOLDER / "voiceover.mp3"
    srt_path = OUTPUT_FOLDER / "captions.srt"
    
    await tts.synthesize(
        text=SCRIPT,
        output_path=audio_path,
        voice="en-US-AriaNeural",  # Will be auto-converted to Google voice
    )
    # SRT is automatically generated at captions.srt by GoogleTTSProvider
    
    duration = get_audio_duration(audio_path)
    print(f"  Audio: {audio_path}")
    print(f"  Duration: {duration:.1f}s")
    
    # Save voiceover text
    voiceover_path = OUTPUT_FOLDER / "voiceover.txt"
    with open(voiceover_path, "w", encoding="utf-8") as f:
        f.write(SCRIPT)
    
    # Create ASS subtitles
    print("\nCreating subtitles...")
    ass_path = OUTPUT_FOLDER / "captions.ass"
    generator._create_ass_with_series_marker(
        srt_path=srt_path,
        ass_path=ass_path,
        series_text=series_info.display_text,
        duration=duration,
        subtitle_style="punch",
    )
    print(f"  ASS: {ass_path}")
    
    # Get images
    bg_folder = OUTPUT_FOLDER / "backgrounds"
    images = sorted(bg_folder.glob("bg_*.jpg"))
    print(f"\nImages: {len(images)}")
    
    # Create slideshow
    print("\nCreating slideshow...")
    slideshow_path = OUTPUT_FOLDER / "background.mp4"
    generator._create_slideshow(
        images=images,
        output_path=slideshow_path,
        duration=duration,
        use_ken_burns=True,
        use_crossfade=True,
    )
    
    # Render final video
    print("Rendering final video...")
    final_path = OUTPUT_FOLDER / "final.mp4"
    generator._render_final(
        slideshow_path=slideshow_path,
        audio_path=audio_path,
        ass_path=ass_path,
        output_path=final_path,
    )
    
    # Quality check
    print("\nQuality check...")
    from pipeline.generate_short import QualityGuard
    guard = QualityGuard()
    if guard.validate(final_path, duration):
        print("  All checks passed!")
    else:
        for error in guard.errors:
            print(f"  ERROR: {error}")
    
    # Save metadata
    metadata_path = OUTPUT_FOLDER / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "topic": TOPIC,
            "series": {
                "name": series_info.series_name,
                "episode": series_info.episode_number,
                "display": series_info.display_text,
            },
            "duration": duration,
            "style": "ukiyo_e",
            "images_generated": len(images),
            "version": "v2",
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"Video: {final_path}")
    print(f"Duration: {duration:.1f}s")
    print("=" * 60)
    
    return final_path


async def main():
    """Main function."""
    print("=" * 60)
    print("Ninja Myth Video V2 - Extended 70 Second Version")
    print("=" * 60)
    print()
    print("Plan:")
    print("  1. Generate 4 new ukiyo-e images")
    print("  2. Keep bg_1 (ninja face) and bg_3 (city ninja)")
    print("  3. Create extended 70s video")
    print()
    
    # Step 1: Generate new images
    await generate_new_images()
    
    # Step 2: Organize images
    organize_images()
    
    # Step 3: Generate video
    await generate_video()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
