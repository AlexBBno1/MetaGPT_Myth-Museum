"""
Generate cartoon-style images for Da Vinci myth story
"""
import asyncio
import json
from pathlib import Path
from pipeline.image_providers.vertex_imagen import ImageProviderWithFallback


async def generate_cartoon_images():
    provider = ImageProviderWithFallback()
    
    prompts_path = Path("outputs/shorts/renaissance-masters_davinci-secrets/davinci_cartoon_prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    output_dir = Path("outputs/shorts/renaissance-masters_davinci-secrets/cartoon_backgrounds")
    output_dir.mkdir(exist_ok=True)
    
    for i, prompt_data in enumerate(data["cartoon_image_prompts"], 1):
        output_path = output_dir / f"cartoon_{i}.jpg"
        scene = prompt_data["scene"]
        print(f"Generating {i}/4: {scene}...")
        
        result = await provider.generate_image(
            prompt=prompt_data["prompt"],
            output_path=output_path,
            fallback_keyword=prompt_data["fallback_keyword"],
            aspect_ratio="9:16",
        )
        
        if result.success:
            print(f"  OK ({result.source})")
        else:
            error = result.error[:50] if result.error else "Unknown"
            print(f"  FAILED: {error}")
    
    print("\nDone! Cartoon images generated.")


if __name__ == "__main__":
    asyncio.run(generate_cartoon_images())
