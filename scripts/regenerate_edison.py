"""Regenerate Edison images with oil_painting_cartoon style."""

import asyncio
import json
from pathlib import Path


async def regenerate_images():
    """Regenerate all 6 images with oil painting cartoon style."""
    from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback
    
    folder = Path("outputs/shorts/history-myths_edison-1000-failures")
    
    with open(folder / "visual_briefs.json", "r", encoding="utf-8") as f:
        briefs = json.load(f)
    
    provider = ImageProviderWithFallback()
    
    print("Regenerating 6 images with oil_painting_cartoon style...")
    print("Using Imagen 3 (1 image/min rate limit)\n")
    
    for i, brief in enumerate(briefs):
        segment = brief["segment"]
        output_path = folder / "backgrounds" / f"{segment}.jpg"
        scene_name = brief["scene_name"]
        print(f"[{i+1}/6] {scene_name}...")
        
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
    
    print("\nImage regeneration complete!")


if __name__ == "__main__":
    asyncio.run(regenerate_images())
