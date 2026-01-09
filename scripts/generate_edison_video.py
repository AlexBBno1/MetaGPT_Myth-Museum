"""Generate Edison 1000 Failures Myth Video."""

import asyncio
import json
from pathlib import Path


async def generate_images():
    """Generate all 6 images for the Edison video."""
    from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
    
    folder = Path("outputs/shorts/history-myths_edison-1000-failures")
    
    # Load visual briefs
    with open(folder / "visual_briefs.json", "r", encoding="utf-8") as f:
        briefs = json.load(f)
    
    provider = ImageProviderWithFallback()
    
    print("Generating 6 images for Edison myth video...")
    print("Using Imagen 3 (1 image/min rate limit)\n")
    
    for i, brief in enumerate(briefs):
        segment = brief["segment"]
        output_path = folder / "backgrounds" / f"{segment}.jpg"
        print(f"[{i+1}/6] {brief['scene_name']}...")
        
        result = await provider.generate_image(
            prompt=brief["prompt"],
            output_path=output_path,
            fallback_keyword=brief["fallback_keyword"],
            quality="high",
        )
        
        if result.success:
            print(f"    OK ({result.source}, {result.latency_ms}ms)")
        else:
            print(f"    FAILED: {result.error}")
    
    print("\nImage generation complete!")
    return folder


async def generate_audio(folder: Path):
    """Generate TTS audio from voiceover script."""
    from pipeline.tts import GoogleTTSProvider, get_audio_duration
    
    script = (folder / "voiceover.txt").read_text(encoding="utf-8")
    audio_path = folder / "voiceover.mp3"
    
    print("Generating voiceover audio...")
    tts = GoogleTTSProvider()
    await tts.synthesize(
        text=script,
        output_path=audio_path,
        voice="en-US-Casual-K",
    )
    
    duration = get_audio_duration(audio_path)
    print(f"    Audio duration: {duration:.1f}s")
    return duration


def generate_subtitles(folder: Path, duration: float):
    """Generate SRT and ASS subtitles."""
    from pipeline.tts import script_to_srt
    
    script = (folder / "voiceover.txt").read_text(encoding="utf-8")
    srt_path = folder / "captions.srt"
    
    print("Creating subtitles...")
    script_to_srt(script, srt_path, duration)
    print("    SRT created")
    
    return srt_path


def render_video(folder: Path, duration: float):
    """Render the final video."""
    import subprocess
    
    images = sorted((folder / "backgrounds").glob("bg_*.jpg"))
    if not images:
        print("ERROR: No images found!")
        return
    
    slideshow_path = folder / "background.mp4"
    final_path = folder / "final.mp4"
    audio_path = folder / "voiceover.mp3"
    srt_path = folder / "captions.srt"
    
    # Calculate duration per image
    num_images = len(images)
    duration_per_image = duration / num_images
    
    print(f"Rendering slideshow ({num_images} images, {duration_per_image:.1f}s each)...")
    
    # Create concat file for FFmpeg
    concat_file = folder / "concat.txt"
    with open(concat_file, "w") as f:
        for img in images:
            f.write(f"file '{img.absolute()}'\n")
            f.write(f"duration {duration_per_image}\n")
        # Add last image again for proper ending
        f.write(f"file '{images[-1].absolute()}'\n")
    
    # Build slideshow with Ken Burns effect
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.001,1.2)':d=125:s=1080x1920",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(slideshow_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Slideshow error: {result.stderr[:200]}")
        # Fallback: simple slideshow without Ken Burns
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(slideshow_path),
        ]
        subprocess.run(cmd, capture_output=True)
    
    print("    Slideshow created")
    
    # Combine with audio and subtitles
    print("Rendering final video with audio and subtitles...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(slideshow_path),
        "-i", str(audio_path),
        "-vf", f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(final_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Final video error: {result.stderr[:200]}")
        # Fallback: without subtitles
        cmd = [
            "ffmpeg", "-y",
            "-i", str(slideshow_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(final_path),
        ]
        subprocess.run(cmd, capture_output=True)
    
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"SUCCESS!")
        print(f"{'='*50}")
        print(f"Video: {final_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Duration: {duration:.1f}s")
    else:
        print("\nFailed to create final video!")


async def main():
    """Main entry point."""
    folder = Path("outputs/shorts/history-myths_edison-1000-failures")
    
    # Step 1: Generate images
    await generate_images()
    
    # Step 2: Generate audio
    duration = await generate_audio(folder)
    
    # Step 3: Generate subtitles
    generate_subtitles(folder, duration)
    
    # Step 4: Render video
    render_video(folder, duration)


if __name__ == "__main__":
    asyncio.run(main())
