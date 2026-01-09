"""
Render Da Vinci myth video using cartoon images + original paintings
Combines cartoon storytelling with famous Da Vinci artworks
"""
import subprocess
from pathlib import Path
from pipeline.tts import GoogleTTSProvider, get_audio_duration
import asyncio


async def main():
    output_dir = Path("outputs/shorts/renaissance-masters_davinci-secrets")
    
    # Voiceover script
    voiceover_text = """You think Da Vinci hid secret codes in his paintings? The truth is even more fascinating.

The Da Vinci Code made millions believe the Mona Lisa and Last Supper contain hidden prophecies.

But historians say these "codes" are modern speculation, not Da Vinci's intention.

Here's what Da Vinci actually hid: pure scientific genius.

The Mona Lisa's mysterious smile? He studied over 40 facial muscles to create it.

The Last Supper's revolutionary perspective? It transformed Renaissance art forever.

The Vitruvian Man? Perfect mathematical proportions of the human body.

Da Vinci didn't need secret codes. His real code was scientific thinking 500 years ahead of his time.

Follow for more myth busters!"""
    
    # Save voiceover text
    voiceover_path = output_dir / "voiceover.txt"
    voiceover_path.write_text(voiceover_text, encoding="utf-8")
    print(f"Saved voiceover: {voiceover_path}")
    
    # Generate TTS
    audio_path = output_dir / "voiceover.mp3"
    tts = GoogleTTSProvider()
    await tts.synthesize(
        text=voiceover_text,
        output_path=audio_path,
        voice="en-US-Journey-D",
        rate="+0%",
    )
    print(f"Generated audio: {audio_path}")
    
    duration = get_audio_duration(audio_path)
    print(f"Duration: {duration:.1f}s")
    
    # Create SRT captions (basic)
    srt_path = output_dir / "captions.srt"
    
    # Use cartoon backgrounds
    cartoon_dir = output_dir / "cartoon_backgrounds"
    images = sorted(cartoon_dir.glob("bg_*.jpg"))
    print(f"Found {len(images)} cartoon images")
    
    # Create slideshow with Ken Burns effect
    time_per_image = duration / len(images)
    frames_per_image = int(time_per_image * 30)
    
    # Ken Burns configs
    kb_configs = [
        f"z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        f"z='if(lte(zoom,1.0),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        f"z='min(zoom+0.0006,1.12)':x='0':y='ih/2-(ih/zoom/2)'",
        f"z='min(zoom+0.0006,1.12)':x='iw-(iw/zoom)':y='ih/2-(ih/zoom/2)'",
    ]
    
    # Build filter complex
    filter_parts = []
    crossfade_duration = 0.8
    
    for i, img in enumerate(images):
        kb = kb_configs[i % len(kb_configs)]
        filter_parts.append(
            f"[{i}:v]scale=1200:2133:force_original_aspect_ratio=increase,"
            f"zoompan={kb}:d={frames_per_image}:s=1080x1920:fps=30,"
            f"setsar=1,format=yuv420p[v{i}];"
        )
    
    # Progressive crossfade chain
    current_label = "v0"
    for i in range(1, len(images)):
        offset = time_per_image * i - crossfade_duration * i
        next_label = f"xf{i-1}" if i < len(images) - 1 else "outv"
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
    slideshow_path = output_dir / "cartoon_slideshow.mp4"
    cmd = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-t', str(duration),
        str(slideshow_path)
    ]
    
    print("Creating slideshow with Ken Burns effect...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return
    print(f"Created slideshow: {slideshow_path}")
    
    # Final render with audio
    final_path = output_dir / "davinci_cartoon_final.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-i', str(slideshow_path),
        '-i', str(audio_path),
        '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0',
        '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(final_path)
    ]
    
    print("Rendering final video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return
    
    print(f"\n✓ Complete!")
    print(f"Final video: {final_path}")
    print(f"Duration: {duration:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
