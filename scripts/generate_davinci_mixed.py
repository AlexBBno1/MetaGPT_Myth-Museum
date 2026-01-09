"""
Generate Da Vinci Mona Lisa Myth video with:
- Real paintings (Mona Lisa, Last Supper) 
- Cartoon Da Vinci character images
- Subtitles (ASS format)
- Ken Burns effect + crossfade

Following the same structure as science-myths_trex-feathers
"""
import asyncio
import json
import subprocess
import re
from pathlib import Path
from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
from pipeline.tts import GoogleTTSProvider, get_audio_duration


OUTPUT_DIR = Path("outputs/shorts/art-myths_davinci-mona-lisa")
BG_DIR = OUTPUT_DIR / "backgrounds"

# Real painting URLs from Wikimedia Commons (via web.archive.org for reliability)
REAL_PAINTINGS = [
    {
        "name": "Mona Lisa",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "filename": "mona_lisa.jpg"
    },
    {
        "name": "Last Supper",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/The_Last_Supper_-_Leonardo_Da_Vinci_-_High_Resolution_32x16.jpg/1280px-The_Last_Supper_-_Leonardo_Da_Vinci_-_High_Resolution_32x16.jpg",
        "filename": "last_supper.jpg"
    }
]

# Cartoon prompts for 4 additional images
CARTOON_PROMPTS = [
    {
        "segment": "cartoon_1",
        "label": "The Myth - Mysterious Da Vinci",
        "prompt": "cartoon style flat illustration, friendly cartoon Leonardo da Vinci character with long hair and beard in Renaissance clothing, holding a glowing mysterious portrait with alien symbols and question marks floating around, dramatic purple and green spooky lighting, dark mysterious background, simple shapes bright colors, no text no watermarks, digital art vector style, 9:16 vertical, 8k quality",
        "fallback": "da vinci mystery cartoon"
    },
    {
        "segment": "cartoon_2",
        "label": "The Doubt - Confused Da Vinci",
        "prompt": "cartoon style flat illustration, friendly cartoon Leonardo da Vinci character with confused puzzled expression scratching his head, colorful question marks and exclamation marks floating around him, curious uncertain atmosphere, orange and yellow warm background, simple shapes bright colors, no text no watermarks, digital art vector style, 9:16 vertical, 8k quality",
        "fallback": "confused thinking cartoon"
    },
    {
        "segment": "cartoon_3",
        "label": "The Truth - Scientific Genius",
        "prompt": "cartoon style flat illustration, happy smiling cartoon Leonardo da Vinci character in artist studio painting at an easel, surrounded by scientific tools magnifying glass anatomy books and paint brushes, golden lightbulb glowing above his head showing eureka moment, warm bright atmosphere, simple shapes bright colors, no text no watermarks, digital art vector style, 9:16 vertical, 8k quality",
        "fallback": "cartoon artist scientist eureka"
    },
    {
        "segment": "cartoon_4",
        "label": "Final Verdict - Myth vs Truth",
        "prompt": "cartoon style flat illustration split screen comparison final verdict, left side shows alien conspiracy symbols with big red X mark crossed out fading away, right side shows scientific genius brain and lightbulb with green checkmark glowing, satisfied conclusive atmosphere, dark background, simple shapes bright colors, no text no watermarks, digital art vector style, 9:16 vertical, 8k quality",
        "fallback": "comparison myth truth checkmark"
    }
]

VOICEOVER_SCRIPT = """Everyone thinks Mona Lisa's smile hides a secret alien message.

The Da Vinci Code made millions believe she holds supernatural powers.

But wait. What did Da Vinci actually do?

He spent 16 years studying facial muscles. Over 40 muscles. For one smile.

He invented sfumato. Layers of transparent paint so thin you can't see brushstrokes.

And the Last Supper? Revolutionary perspective that changed art forever.

The mystery? It's an optical illusion. Your brain can't decide if she's smiling or not.

Da Vinci wasn't hiding alien codes. He was 500 years ahead in neuroscience.

That's the real secret. Pure scientific genius.

Follow for more art myth busters!"""


async def download_real_paintings():
    """Download Mona Lisa and Last Supper from Wikimedia."""
    import urllib.request
    
    print("\n" + "=" * 60)
    print("Step 1: Downloading Real Paintings")
    print("=" * 60)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for painting in REAL_PAINTINGS:
        output_path = BG_DIR / painting["filename"]
        print(f"  Downloading {painting['name']}...")
        
        try:
            req = urllib.request.Request(painting["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            print(f"    OK: {output_path}")
        except Exception as e:
            print(f"    FAILED: {e}")
            # Try web.archive.org as fallback
            try:
                archive_url = f"https://web.archive.org/web/2024/{painting['url']}"
                req = urllib.request.Request(archive_url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(output_path, 'wb') as f:
                        f.write(response.read())
                print(f"    OK (archive): {output_path}")
            except Exception as e2:
                print(f"    FAILED (archive): {e2}")


async def generate_cartoon_images():
    """Generate cartoon Da Vinci images using Vertex Imagen."""
    print("\n" + "=" * 60)
    print("Step 2: Generating Cartoon Images")
    print("=" * 60)
    
    provider = ImageProviderWithFallback()
    
    for i, prompt_data in enumerate(CARTOON_PROMPTS, 1):
        output_path = BG_DIR / f"{prompt_data['segment']}.jpg"
        print(f"  [{i}/4] {prompt_data['label']}...")
        
        result = await provider.generate_image(
            prompt=prompt_data["prompt"],
            output_path=output_path,
            fallback_keyword=prompt_data["fallback"],
            aspect_ratio="9:16",
        )
        
        if result.success:
            print(f"    OK ({result.source})")
        else:
            error = result.error[:60] if result.error else "Unknown"
            print(f"    FAILED: {error}")


async def generate_tts():
    """Generate TTS audio and get synced SRT."""
    print("\n" + "=" * 60)
    print("Step 3: Generating TTS Audio")
    print("=" * 60)
    
    # Save voiceover script
    voiceover_path = OUTPUT_DIR / "voiceover.txt"
    voiceover_path.write_text(VOICEOVER_SCRIPT, encoding="utf-8")
    
    audio_path = OUTPUT_DIR / "voiceover.mp3"
    
    tts = GoogleTTSProvider()
    await tts.synthesize(
        text=VOICEOVER_SCRIPT,
        output_path=audio_path,
        voice="en-US-Journey-D",
        rate="+0%",
    )
    
    duration = get_audio_duration(audio_path)
    print(f"  Audio: {audio_path}")
    print(f"  Duration: {duration:.1f}s")
    
    return duration


def create_ass_subtitles(duration: float):
    """Create ASS subtitles with series marker."""
    print("\n" + "=" * 60)
    print("Step 4: Creating Subtitles (ASS)")
    print("=" * 60)
    
    srt_path = OUTPUT_DIR / "captions.srt"
    ass_path = OUTPUT_DIR / "captions.ass"
    
    # Read the auto-generated SRT from TTS
    if not srt_path.exists():
        print("  WARNING: No SRT file found")
        return
    
    srt_content = srt_path.read_text(encoding="utf-8")
    
    # Create ASS file
    ass_content = '''[Script Info]
Title: Da Vinci Mona Lisa Myth
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,4,0,5,80,80,300,1
Style: SeriesMarker,Arial,32,&H99FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,9,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    
    # Parse SRT and convert to ASS dialogue lines
    srt_blocks = re.split(r'\n\n+', srt_content.strip())
    
    for block in srt_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})',
                lines[1]
            )
            if time_match:
                sh, sm, ss, sms = time_match.groups()[:4]
                eh, em, es, ems = time_match.groups()[4:]
                start = f"{int(sh)}:{sm}:{ss}.{sms[:2]}"
                end = f"{int(eh)}:{em}:{es}.{ems[:2]}"
                text = ' '.join(lines[2:]).replace('\n', ' ')
                ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    
    # Add series marker
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    centiseconds = int((duration % 1) * 100)
    end_time = f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    ass_content += f"Dialogue: 1,0:00:00.00,{end_time},SeriesMarker,,0,0,0,,Art Myths #1\n"
    
    ass_path.write_text(ass_content, encoding="utf-8")
    print(f"  ASS: {ass_path}")


def organize_backgrounds():
    """Organize images into bg_1.jpg through bg_6.jpg for rendering."""
    print("\n" + "=" * 60)
    print("Step 5: Organizing Background Images")
    print("=" * 60)
    
    import shutil
    
    # Image sequence: cartoon → real → cartoon → real → cartoon → cartoon
    # 1. cartoon_1 (myth intro)
    # 2. mona_lisa (real painting - mysterious smile)
    # 3. cartoon_2 (doubt)
    # 4. last_supper (real painting - perspective)
    # 5. cartoon_3 (truth - scientific genius)
    # 6. cartoon_4 (final verdict)
    
    sequence = [
        ("cartoon_1.jpg", "bg_1.jpg"),
        ("mona_lisa.jpg", "bg_2.jpg"),
        ("cartoon_2.jpg", "bg_3.jpg"),
        ("last_supper.jpg", "bg_4.jpg"),
        ("cartoon_3.jpg", "bg_5.jpg"),
        ("cartoon_4.jpg", "bg_6.jpg"),
    ]
    
    for src_name, dst_name in sequence:
        src = BG_DIR / src_name
        dst = BG_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src_name} -> {dst_name}")
        else:
            print(f"  WARNING: {src_name} not found!")


def render_video(duration: float):
    """Render final video with Ken Burns, crossfade, and subtitles."""
    print("\n" + "=" * 60)
    print(f"Step 6: Rendering Video ({duration:.1f}s)")
    print("=" * 60)
    
    images = sorted(BG_DIR.glob("bg_*.jpg"))
    num_images = len(images)
    
    if num_images == 0:
        print("  ERROR: No bg_*.jpg images found!")
        return False
    
    print(f"  Found {num_images} images")
    
    # Calculate timing
    crossfade_duration = 0.8
    total_crossfade_time = crossfade_duration * (num_images - 1)
    time_per_image = (duration + total_crossfade_time) / num_images
    frames_per_image = int(time_per_image * 30)
    
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
    
    # Progressive crossfade
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
    slideshow_path = OUTPUT_DIR / "background.mp4"
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
    
    # Final render with audio AND subtitles
    audio_path = OUTPUT_DIR / "voiceover.mp3"
    ass_path = OUTPUT_DIR / "captions.ass"
    final_path = OUTPUT_DIR / "final.mp4"
    
    # Escape path for ffmpeg
    ass_path_escaped = str(ass_path).replace('\\', '/').replace(':', '\\:')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(slideshow_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-vf', f"ass='{ass_path_escaped}'",
        '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0',
        '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
        '-shortest',
        str(final_path)
    ]
    
    print("  Rendering final video with subtitles...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False
    
    print(f"  Final: {final_path}")
    return True


async def main():
    print("\n" + "=" * 60)
    print("DA VINCI MONA LISA MYTH VIDEO GENERATOR")
    print("Mixed: Real Paintings + Cartoon Characters + Subtitles")
    print("=" * 60)
    
    BG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download real paintings
    await download_real_paintings()
    
    # Step 2: Generate cartoon images
    await generate_cartoon_images()
    
    # Step 3: Generate TTS
    duration = await generate_tts()
    
    # Step 4: Create subtitles
    create_ass_subtitles(duration)
    
    # Step 5: Organize images
    organize_backgrounds()
    
    # Step 6: Render video
    success = render_video(duration)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Video: {OUTPUT_DIR / 'final.mp4'}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED - Check errors above")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
