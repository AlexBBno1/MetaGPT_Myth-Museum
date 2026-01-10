"""
Generate Galaxy Speed Myth Video

Story: The Milky Way Isn't Standing Still
Style: Sci-Fi Cinematic (Interstellar aesthetic)
Series: Space Myths #1
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
from pipeline.tts import GoogleTTSProvider, get_audio_duration


async def generate_galaxy_video():
    """Generate the complete galaxy speed myth video."""
    
    output_dir = Path("outputs/shorts/space-myths_galaxy-speed")
    bg_dir = output_dir / "backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GALAXY SPEED MYTH VIDEO GENERATION")
    print("=" * 60)
    
    # Load visual briefs
    briefs_path = output_dir / "visual_briefs.json"
    with open(briefs_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    print(f"\nLoaded {len(prompts)} visual briefs")
    
    # =========================================================================
    # Phase 1: Generate Images
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: IMAGE GENERATION (Sci-Fi Cinematic Style)")
    print("=" * 60)
    
    provider = ImageProviderWithFallback()
    
    for i, p in enumerate(prompts):
        output_path = bg_dir / f"{p['segment']}.jpg"
        print(f"\n[{i+1}/{len(prompts)}] {p['scene_name']}")
        print(f"    Prompt: {p['prompt'][:80]}...")
        
        result = await provider.generate_image(
            prompt=p["prompt"],
            output_path=output_path,
            fallback_keyword=p.get("fallback_keyword", "galaxy space"),
            aspect_ratio="9:16",
            quality="high",  # Use Imagen 3 for best quality
        )
        
        if result.success:
            size_kb = output_path.stat().st_size / 1024
            print(f"    [OK] {result.source} - {size_kb:.1f}KB")
        else:
            print(f"    [FAILED] {result.error}")
    
    # =========================================================================
    # Phase 2: Generate TTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: TTS GENERATION")
    print("=" * 60)
    
    script_path = output_dir / "voiceover.txt"
    script = script_path.read_text(encoding="utf-8")
    
    tts = GoogleTTSProvider()
    audio_path = output_dir / "voiceover.mp3"
    
    await tts.synthesize(
        text=script,
        output_path=audio_path,
        voice="en-US-Casual-K",
    )
    
    duration = get_audio_duration(audio_path)
    print(f"Audio generated: {duration:.1f}s")
    
    # NOTE: Google TTS automatically generates captions.srt with accurate
    # word-level timing using SSML marks. We use that directly instead of
    # manually calculating subtitle timing.
    srt_path = output_dir / "captions.srt"
    if srt_path.exists():
        print(f"Using TTS-generated SRT with accurate timing: {srt_path}")
    else:
        print("[WARNING] TTS did not generate SRT file!")
    
    # =========================================================================
    # Phase 3: Render Video using the main pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: VIDEO RENDERING")
    print("=" * 60)
    
    from pipeline.generate_short import ShortVideoGenerator
    
    generator = ShortVideoGenerator()
    
    # Create ASS with series marker and punch style
    ass_path = output_dir / "captions.ass"
    generator._create_ass_with_series_marker(
        srt_path=srt_path,
        ass_path=ass_path,
        series_text="Space Myths #1",
        duration=duration,
        subtitle_style="punch",
    )
    print(f"ASS created with punch style: {ass_path}")
    
    # Create slideshow with Ken Burns effect
    images = sorted(bg_dir.glob("bg_*.jpg"))
    slideshow_path = output_dir / "background.mp4"
    
    print(f"Creating slideshow from {len(images)} images...")
    generator._create_slideshow(
        images=images,
        output_path=slideshow_path,
        duration=duration,
        use_ken_burns=True,
        use_crossfade=True,
    )
    print(f"Slideshow: {slideshow_path}")
    
    # Final render
    final_path = output_dir / "final.mp4"
    print("Rendering final video with subtitles...")
    generator._render_final(
        slideshow_path=slideshow_path,
        audio_path=audio_path,
        ass_path=ass_path,
        output_path=final_path,
    )
    
    # Quality check
    from pipeline.generate_short import QualityGuard
    
    quality_guard = QualityGuard()
    quality_passed = quality_guard.validate(final_path, duration)
    
    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)
    
    if quality_passed:
        print("[PASS] All quality checks passed!")
    else:
        for error in quality_guard.errors:
            print(f"[ERROR] {error}")
    
    for warning in quality_guard.warnings:
        print(f"[WARNING] {warning}")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"Final video: {final_path}")
        print(f"File size: {size_mb:.1f} MB")
        print(f"Duration: {duration:.1f}s")
        print(f"Series: Space Myths #1")
        print(f"\nTo view: start {final_path}")
    else:
        print("[ERROR] Final video was not created!")
    
    # Save metadata
    metadata = {
        "topic": "Galaxy Speed - We're Not Standing Still",
        "series": {
            "name": "Space Myths",
            "episode": 1,
            "display": "Space Myths #1"
        },
        "duration": duration,
        "style": "sci_fi_cinematic",
        "images_generated": len(list(bg_dir.glob("bg_*.jpg"))),
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return final_path


if __name__ == "__main__":
    asyncio.run(generate_galaxy_video())
