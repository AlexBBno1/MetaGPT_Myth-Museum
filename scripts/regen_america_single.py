"""
Regenerate a single America Myths video.

Usage:
    python scripts/regen_america_single.py washington
    python scripts/regen_america_single.py wildwest
    python scripts/regen_america_single.py thanksgiving
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
from pipeline.shorts_optimizer import ShortsOptimizer


# ============================================================================
# Video Configurations (with improved prompts)
# ============================================================================

VIDEOS = {
    "washington": {
        "topic": "Washington Cherry Tree Myth",
        "style": "colonial_portrait",
        "folder": "america-myths_washington-cherry-tree-myth",
        "script": """George Washington never told a lie? That's actually the biggest lie in American history.

You know the story. Young George chops down his father's cherry tree. When confronted, he says "I cannot tell a lie" and confesses everything. The perfect tale of American honesty.

But here's the truth - this story was completely invented. A man named Parson Weems made it up in 1806, seven years after Washington died. He needed to sell books, so he created the perfect founding father myth.

The real Washington? He was a brilliant spymaster who ran one of history's greatest intelligence networks. He deceived the British constantly. Lying was literally part of his job.

So the most famous story about America's most honest president... is a lie told by a book salesman.

Sometimes the myth tells you more about what people wanted to believe than what actually happened.""",
        "image_prompts": [
            {
                "segment": "bg_1",
                "prompt": "Oil painting portrait of adult George Washington with finger raised to lips in shh gesture, mysterious knowing smile, dramatic candlelit shadows, dark background, 18th century colonial aesthetic, museum quality, 9:16 vertical portrait, no text",
                "fallback_keyword": "george washington portrait painting",
            },
            {
                "segment": "bg_2",
                "prompt": "Oil painting still life of small cherry tree stump with rustic hatchet embedded in wood, dramatic morning light, Virginia colonial garden setting, symbolic composition, visible brushstrokes, 9:16 vertical, no text",
                "fallback_keyword": "cherry tree stump axe painting",
            },
            {
                "segment": "bg_3",
                "prompt": "Oil painting of colonial era writer at wooden desk with quill pen, stack of leather books, oval portrait of Washington on wall behind, warm candlelight, 18th century study interior, scholarly atmosphere, 9:16 vertical, no text",
                "fallback_keyword": "colonial writer desk books painting",
            },
            {
                "segment": "bg_4",
                "prompt": "Oil painting of George Washington in military tent studying secret map by candlelight, invisible ink bottle on table, coded letter in hand, dramatic shadows, spy master atmosphere, Revolutionary War setting, 9:16 vertical, no text",
                "fallback_keyword": "george washington military tent map",
            },
            {
                "segment": "bg_5",
                "prompt": "Oil painting of secret midnight meeting, cloaked figures exchanging sealed letters by lantern light, colonial tavern back room, mysterious shadows, Revolutionary War espionage atmosphere, dramatic chiaroscuro, 9:16 vertical, no text",
                "fallback_keyword": "colonial secret meeting lantern night",
            },
            {
                "segment": "bg_6",
                "prompt": "Oil painting split composition, left side innocent young boy, right side cunning military general Washington, same face at different ages, truth vs myth visual metaphor, dramatic lighting, museum quality, 9:16 vertical, no text",
                "fallback_keyword": "george washington portrait two faces",
            },
        ],
    },
    "wildwest": {
        "topic": "Wild West Wasn't Wild Myth",
        "style": "vintage_sepia",
        "folder": "america-myths_wild-west-wasn-t-wild-myth",
        "script": """The Wild West? Hollywood invented it.

We've all seen it - dusty streets, quick-draw duels, outlaws robbing banks every week. The most dangerous place in American history, right?

Wrong. Historical records show something shocking. The famous town of Tombstone in 1881 had only five murders. Five. The entire year. Modern cities have worse rates.

Most frontier towns actually banned carrying guns in public. Yes, really. The OK Corral gunfight? It started because the Earps were enforcing a gun control law.

Real cowboys spent their days herding cattle, fixing fences, and being incredibly bored. The average cowboy never fired his gun at another person. Ever.

So where did the myth come from? Dime novels and Wild West shows needed exciting stories to sell tickets.

The real Wild West's biggest danger wasn't outlaws. It was boredom. And maybe dysentery.""",
        "image_prompts": [
            {
                "segment": "bg_1",
                "prompt": "Vintage sepia photograph style, dramatic movie poster aesthetic, cowboy silhouette with gun drawn against sunset, crumbling film reel border, 1880s daguerreotype grain, 9:16 vertical, no text",
                "fallback_keyword": "wild west movie poster vintage",
            },
            {
                "segment": "bg_2",
                "prompt": "Vintage sepia photograph, action western movie scene, stagecoach chase with gunfire, dramatic explosion and dust clouds, Hollywood movie still aesthetic, high contrast, aged film grain, 9:16 vertical, no text",
                "fallback_keyword": "western movie stagecoach action scene",
            },
            {
                "segment": "bg_3",
                "prompt": "Vintage sepia photograph, peaceful empty western main street, single tumbleweed rolling, no people, boring quiet afternoon, simple wooden buildings, calm atmosphere, documentary realism, 9:16 vertical, no text",
                "fallback_keyword": "empty western town peaceful street",
            },
            {
                "segment": "bg_4",
                "prompt": "Vintage sepia photograph, sheriff standing next to wooden barrel full of collected pistols, 1880s frontier town entrance, law enforcement scene, historical documentary style, 9:16 vertical, no text",
                "fallback_keyword": "old west sheriff guns barrel",
            },
            {
                "segment": "bg_5",
                "prompt": "Vintage sepia photograph, tired cowboys repairing wooden fence posts, one man yawning, cattle in background, mundane boring ranch work, dusty clothes, no action, authentic 1880s documentary style, 9:16 vertical, no text",
                "fallback_keyword": "cowboys fence repair ranch work boring",
            },
            {
                "segment": "bg_6",
                "prompt": "Vintage sepia style circus poster, Buffalo Bill Wild West Show advertisement, dramatic illustrations of cowboys and performers, entertainment spectacle aesthetic, aged paper texture, 9:16 vertical, no text",
                "fallback_keyword": "buffalo bill wild west show poster",
            },
        ],
    },
    "thanksgiving": {
        "topic": "Thanksgiving Turkey Myth",
        "style": "americana",
        "folder": "america-myths_thanksgiving-turkey-myth",
        "script": """Thanksgiving turkey? The Pilgrims would be very confused.

Every November, millions of Americans gather around a roasted turkey, believing they're honoring a 400-year-old tradition. There's just one problem - it's probably not what they ate.

The only record of the 1621 feast mentions venison - that's deer. Historian accounts suggest they ate lobster, eel, clams, and corn. Turkey? Maybe. But definitely not the star of the show.

So how did turkey become THE Thanksgiving food? Thank Sarah Josepha Hale. In the 1800s, she campaigned for a national Thanksgiving holiday and promoted turkey as the centerpiece. It was basically a marketing campaign.

The cranberry sauce, the pumpkin pie, the specific menu we follow - all invented centuries after that first feast.

So this Thanksgiving, remember: you're not following a Pilgrim tradition. You're following a 19th century magazine editor's recipe recommendations. Traditions sometimes start with good marketing.""",
        "image_prompts": [
            {
                "segment": "bg_1",
                "prompt": "Americana illustration style, confused Pilgrim man in black hat holding modern roasted turkey leg, question marks floating around his head, warm autumn background, humorous expression, Saturday Evening Post aesthetic, 9:16 vertical, no text",
                "fallback_keyword": "pilgrim confused turkey thanksgiving",
            },
            {
                "segment": "bg_2",
                "prompt": "Classic Americana illustration, happy 1950s family gathered around golden roasted turkey on dining table, warm indoor lighting, traditional American Thanksgiving scene, nostalgic warm colors, painted illustration style, 9:16 vertical, no text",
                "fallback_keyword": "family thanksgiving dinner turkey 1950s",
            },
            {
                "segment": "bg_3",
                "prompt": "Americana illustration style, 1621 outdoor harvest feast, Pilgrims and Native Americans eating venison deer meat and red lobsters, corn on table, autumn trees background, NO turkey on table, warm earthy tones, 9:16 vertical, no text",
                "fallback_keyword": "first thanksgiving pilgrims native americans feast",
            },
            {
                "segment": "bg_4",
                "prompt": "Victorian illustration style, determined woman editor at wooden desk covered in papers and magazines, quill pen in hand, 1860s fashion dress, warm lamplight, vintage magazine office, professional atmosphere, 9:16 vertical, no text",
                "fallback_keyword": "victorian woman editor desk magazines",
            },
            {
                "segment": "bg_5",
                "prompt": "Vintage 1800s advertisement poster style, colorful turkey dinner promotion, Victorian era marketing aesthetic, decorative borders, aged paper texture, commercial advertising illustration, warm autumn colors, 9:16 vertical, no text",
                "fallback_keyword": "vintage turkey advertisement poster 1800s",
            },
            {
                "segment": "bg_6",
                "prompt": "Americana illustration style, beautiful golden roasted turkey dinner on elegant table, warm satisfied atmosphere, traditional Thanksgiving celebration, warm autumn colors, painted illustration, 9:16 vertical, no text",
                "fallback_keyword": "thanksgiving turkey dinner traditional",
            },
        ],
    },
}


async def regenerate_video(video_key: str):
    """Regenerate a single video with new images."""
    if video_key not in VIDEOS:
        print(f"Unknown video: {video_key}")
        print(f"Available: {', '.join(VIDEOS.keys())}")
        return 1
    
    config_data = VIDEOS[video_key]
    output_base = Path("outputs/shorts")
    old_folder = output_base / config_data["folder"]
    
    # Delete old folder if exists
    if old_folder.exists():
        print(f"Removing old folder: {old_folder}")
        shutil.rmtree(old_folder)
    
    # Get series info (don't increment - reuse same episode number)
    optimizer = ShortsOptimizer()
    
    print("=" * 60)
    print(f"Regenerating: {config_data['topic']}")
    print(f"Style: {config_data['style']}")
    print("=" * 60)
    
    generator = ShortVideoGenerator()
    
    config = GenerationConfig(
        topic=config_data["topic"],
        script=config_data["script"],
        image_quality="high",
        subtitle_style="punch",
        auto_prompts=False,
        image_prompts=config_data["image_prompts"],
        series_override="America Myths",
    )
    
    result = await generator.generate(config)
    
    if result.success:
        print(f"\nSUCCESS: {result.final_video}")
        return 0
    else:
        print(f"\nFAILED: {result.error}")
        return 1


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/regen_america_single.py <video>")
        print("Videos: washington, wildwest, thanksgiving")
        return 1
    
    video_key = sys.argv[1].lower()
    return await regenerate_video(video_key)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
