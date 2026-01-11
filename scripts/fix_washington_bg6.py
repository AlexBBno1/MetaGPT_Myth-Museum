"""
Fix Washington Cherry Tree Video - Regenerate only bg_6
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback


async def main():
    output_folder = Path("outputs/shorts/america-myths_washington-cherry-tree-myth")
    bg_folder = output_folder / "backgrounds"
    
    # New simpler prompt for bg_6
    prompt = (
        "Oil painting still life, open antique leather book on wooden desk "
        "showing handwritten story page, large red wax seal stamped across the text, "
        "quill pen beside book, warm candlelight, 18th century study aesthetic, "
        "symbolic revelation moment, museum quality, 9:16 vertical, no text"
    )
    
    print("=" * 60)
    print("Regenerating bg_6 for Washington Cherry Tree")
    print("=" * 60)
    print()
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    provider = ImageProviderWithFallback()
    output_path = bg_folder / "bg_6.jpg"
    
    # Backup old image
    if output_path.exists():
        backup_path = bg_folder / "bg_6_old.jpg"
        output_path.rename(backup_path)
        print(f"Backed up old image to: {backup_path}")
    
    result = await provider.generate_image(
        prompt=prompt,
        output_path=output_path,
        fallback_keyword="antique book desk candlelight painting",
        quality="high",
    )
    
    if result.success:
        print(f"\nSUCCESS: {output_path}")
        print(f"Source: {result.source}")
        return 0
    else:
        print(f"\nFAILED: {result.error}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
