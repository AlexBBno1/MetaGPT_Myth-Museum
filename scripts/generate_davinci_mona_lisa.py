"""
Generate cartoon-style images for Da Vinci Mona Lisa Smile myth story
Following the same style as the T-Rex feathers video
"""
import asyncio
import json
import subprocess
from pathlib import Path
from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
from pipeline.tts import GoogleTTSProvider, get_audio_duration


async def generate_images():
    """Generate 6 cartoon images for the Da Vinci myth video."""
    provider = ImageProviderWithFallback()
    
    prompts_path = Path("outputs/shorts/art-myths_davinci-mona-lisa/image_prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    output_dir = Path("outputs/shorts/art-myths_davinci-mona-lisa/backgrounds")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Da Vinci Mona Lisa Myth Images")
    print("=" * 60)
    
    for i, prompt_data in enumerate(data["prompts"], 1):
        output_path = output_dir / f"{prompt_data['segment']}.jpg"
        scene = prompt_data["scene"]
        label = prompt_data["label"]
        print(f"\n[{i}/6] {label}")
        print(f"  Scene: {scene}")
        
        result = await provider.generate_image(
            prompt=prompt_data["prompt"],
            output_path=output_path,
            fallback_keyword=prompt_data["fallback_keyword"],
            aspect_ratio="9:16",
        )
        
        if result.success:
            print(f"  Status: OK ({result.source})")
        else:
            error = result.error[:80] if result.error else "Unknown"
            print(f"  Status: FAILED - {error}")
    
    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("=" * 60)


async def generate_tts():
    """Generate TTS audio for the voiceover."""
    output_dir = Path("outputs/shorts/art-myths_davinci-mona-lisa")
    
    voiceover_path = output_dir / "voiceover.txt"
    audio_path = output_dir / "voiceover.mp3"
    
    script = voiceover_path.read_text(encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("Generating TTS Audio")
    print("=" * 60)
    
    tts = GoogleTTSProvider()
    await tts.synthesize(
        text=script,
        output_path=audio_path,
        voice="en-US-Journey-D",
        rate="+0%",
    )
    
    duration = get_audio_duration(audio_path)
    print(f"  Audio: {audio_path}")
    print(f"  Duration: {duration:.1f}s")
    
    return duration


def render_video(duration: float):
    """Render final video with Ken Burns effect and crossfade."""
    output_dir = Path("outputs/shorts/art-myths_davinci-mona-lisa")
    bg_dir = output_dir / "backgrounds"
    
    images = sorted(bg_dir.glob("bg_*.jpg"))
    print(f"\n" + "=" * 60)
    print(f"Rendering Video ({len(images)} images, {duration:.1f}s)")
    print("=" * 60)
    
    if len(images) == 0:
        print("ERROR: No images found!")
        return False
    
    # Calculate time per image with crossfade
    crossfade_duration = 0.8
    num_images = len(images)
    total_crossfade_time = crossfade_duration * (num_images - 1)
    time_per_image = (duration + total_crossfade_time) / num_images
    
    frames_per_image = int(time_per_image * 30)  # 30fps
    
    # Ken Burns configs
    kb_configs = [
        "z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='if(lte(zoom,1.0),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='min(zoom+0.0006,1.12)':x='0':y='ih/2-(ih/zoom/2)'",
        "z='min(zoom+0.0006,1.12)':x='iw-(iw/zoom)':y='ih/2-(ih/zoom/2)'",
    ]
    
    # Build filter complex
    filter_parts = []
    
    for i, img in enumerate(images):
        kb = kb_configs[i % len(kb_configs)]
        filter_parts.append(
            f"[{i}:v]scale=1200:2133:force_original_aspect_ratio=increase,"
            f"zoompan={kb}:d={frames_per_image}:s=1080x1920:fps=30,"
            f"setsar=1,format=yuv420p[v{i}];"
        )
    
    # Progressive crossfade chain
    current_label = "v0"
    for i in range(1, num_images):
        offset = time_per_image * i - crossfade_duration * i
        next_label = f"xf{i-1}" if i < num_images - 1 else "outv"
        filter_parts.append(
            f"[{current_label}][v{i}]xfade=transition=fade:duration={crossfade_duration}:offset={offset:.2f}[{next_label}];"
        )
        current_label = next_label
    
    filter_parts[-1] = filter_parts[-1].rstrip(';')
    filter_complex = ''.join(filter_parts)
    
    # Build input args
    input_args = []
    for img in images:
        input_args.extend(['-loop', '1', '-t', str(time_per_image + 1), '-i', str(img)])
    
    # Create slideshow
    slideshow_path = output_dir / "background_slideshow.mp4"
    cmd = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        str(slideshow_path)
    ]
    
    print("  Creating slideshow with Ken Burns effect...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return False
    print(f"  Slideshow: {slideshow_path}")
    
    # Final render with audio
    audio_path = output_dir / "voiceover.mp3"
    final_path = output_dir / "final.mp4"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(slideshow_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0',
        '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
        '-shortest',
        str(final_path)
    ]
    
    print("  Rendering final video with audio...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return False
    
    print(f"  Final: {final_path}")
    return True


async def main():
    print("\n" + "=" * 60)
    print("DA VINCI MONA LISA SMILE MYTH VIDEO GENERATOR")
    print("=" * 60)
    
    # Step 1: Generate images
    await generate_images()
    
    # Step 2: Generate TTS
    duration = await generate_tts()
    
    # Step 3: Render video
    success = render_video(duration)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! Video generated:")
        print("outputs/shorts/art-myths_davinci-mona-lisa/final.mp4")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED - Check errors above")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
