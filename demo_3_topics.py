"""
Demo Script: Generate packets for 3 topics (health, history, science)

This script demonstrates the Myth Museum pipeline by:
1. Initializing the database
2. Adding sample raw content for 3 topics
3. Extracting claims
4. Gathering evidence
5. Generating verdicts
6. Creating video script packets

Run: python demo_3_topics.py
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows SSL certificate issues for development
# Set this BEFORE any httpx imports
os.environ["MYTH_MUSEUM_SKIP_SSL"] = "1"

# Lenient evidence mode: allow proceeding with fewer source types
os.environ["MYTH_MUSEUM_MIN_EVIDENCE_SOURCES"] = "1"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix Unicode for Windows
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from rich.console import Console
from rich.table import Table

from core.config import get_db_path, ensure_directories
from core.db import (
    init_db,
    get_connection,
    insert_source,
    insert_raw_item,
    get_table_counts,
    get_all_packets,
)
from core.models import SourceConfig, RawItem
from core.textnorm import compute_hash
from core.constants import SourceTypeEnum

console = Console()


# Sample content for 3 topics
SAMPLE_DATA = {
    "health": [
        {
            "title": "Common Health Myths Debunked",
            "url": "https://example.com/health-myths",
            "content": """
            People say drinking 8 glasses of water daily is essential for good health.
            This is a widely believed myth. Many believe that cracking your knuckles 
            causes arthritis. Some claim that eating carrots improves night vision.
            It is said that we only use 10% of our brain. Studies show that sugar 
            makes children hyperactive, but this is actually a misconception.
            """
        },
        {
            "title": "Vitamin and Supplement Myths",
            "url": "https://example.com/vitamin-myths",
            "content": """
            It is believed that vitamin C prevents colds. People think that taking 
            multivitamins every day is necessary for health. Some say that 
            detox diets clean your body of toxins. The claim that organic food 
            is more nutritious than conventional food is widely spread.
            """
        },
    ],
    "history": [
        {
            "title": "Historical Myths That Everyone Believes",
            "url": "https://example.com/history-myths",
            "content": """
            People believe that Napoleon was a short man. The myth says that 
            Vikings wore horned helmets in battle. Many claim that the Great Wall 
            of China is visible from space. It is said that medieval people 
            thought the Earth was flat. Some believe Christopher Columbus 
            discovered America.
            """
        },
        {
            "title": "Ancient History Misconceptions",
            "url": "https://example.com/ancient-myths",
            "content": """
            People think that Egyptian pyramids were built by slaves. The belief 
            that gladiators always fought to the death is common. Many say that 
            ancient Romans used vomitoriums to vomit during feasts. It is claimed 
            that Einstein failed math in school.
            """
        },
    ],
    "science": [
        {
            "title": "Science Myths You Probably Believe",
            "url": "https://example.com/science-myths",
            "content": """
            It is said that lightning never strikes the same place twice. People 
            believe that goldfish have a 3-second memory. The myth claims that 
            bats are blind. Many think humans have exactly 5 senses. Some say 
            that evolution claims we evolved from monkeys.
            """
        },
        {
            "title": "Physics and Space Misconceptions",
            "url": "https://example.com/physics-myths",
            "content": """
            People believe that there is a dark side of the moon that never sees 
            sunlight. It is claimed that black holes are cosmic vacuum cleaners. 
            Many think that seasons are caused by Earth's distance from the sun. 
            The myth says that dropping a penny from the Empire State Building 
            could kill someone.
            """
        },
    ],
}


async def setup_demo_data():
    """Initialize database and insert sample data."""
    console.print("\n[bold cyan]=== Myth Museum Demo: 3 Topics ===[/bold cyan]\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize database
    db_path = get_db_path()
    init_db(db_path)
    console.print(f"[green]OK[/green] Database initialized: {db_path}")
    
    with get_connection(db_path) as conn:
        # Create sources for each topic
        topic_sources = {}
        for topic in SAMPLE_DATA.keys():
            source = SourceConfig(
                name=f"Demo {topic.title()} Source",
                type=SourceTypeEnum.RSS,
                config={"topic": topic},
                enabled=True,
            )
            source_id = insert_source(conn, source)
            topic_sources[topic] = source_id
            console.print(f"[green]OK[/green] Created source for {topic}: ID {source_id}")
        
        # Insert raw items for each topic
        total_items = 0
        for topic, items in SAMPLE_DATA.items():
            source_id = topic_sources[topic]
            for item in items:
                raw_item = RawItem(
                    source_id=source_id,
                    url=item["url"],
                    title=item["title"],
                    content=item["content"],
                    published_at=datetime.now(),
                    fetched_at=datetime.now(),
                    hash=compute_hash(item["content"]),
                )
                insert_raw_item(conn, raw_item)
                total_items += 1
        
        conn.commit()
        console.print(f"[green]OK[/green] Inserted {total_items} raw items for 3 topics\n")
        
        # Show status
        counts = get_table_counts(conn)
        
    return db_path


async def run_pipeline():
    """Run the full pipeline."""
    from pipeline.extract_claims import process_raw_items
    from pipeline.build_evidence import process_claims_for_evidence
    from pipeline.judge_claim import process_claims_for_verdict
    from pipeline.generate_scripts import process_verdicts_for_scripts
    
    db_path = get_db_path()
    
    with get_connection(db_path) as conn:
        # Step 1: Extract claims
        console.print("[bold]Step 1: Extracting claims from raw content...[/bold]")
        claims_count = await process_raw_items(conn, use_llm=False, limit=50)
        conn.commit()
        console.print(f"  → Extracted {claims_count} claims\n")
        
        # Step 2: Build evidence
        console.print("[bold]Step 2: Gathering evidence for claims...[/bold]")
        evidence_count = await process_claims_for_evidence(conn, min_score=0, limit=20)
        conn.commit()
        console.print(f"  → Gathered evidence for {evidence_count} claims\n")
        
        # Step 3: Generate verdicts
        console.print("[bold]Step 3: Generating verdicts...[/bold]")
        verdicts_count = await process_claims_for_verdict(conn, use_llm=False, limit=20)
        conn.commit()
        console.print(f"  → Generated {verdicts_count} verdicts\n")
        
        # Step 4: Create packets
        console.print("[bold]Step 4: Creating content packets...[/bold]")
        packets_count = await process_verdicts_for_scripts(conn, use_llm=False, limit=20)
        conn.commit()
        console.print(f"  → Created {packets_count} packets\n")
        
        return packets_count


async def show_results():
    """Display generated packets by topic."""
    db_path = get_db_path()
    
    with get_connection(db_path) as conn:
        counts = get_table_counts(conn)
        packets = get_all_packets(conn, limit=50)
    
    # Show database status
    console.print("[bold cyan]=== Database Status ===[/bold cyan]")
    table = Table()
    table.add_column("Table", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for name, count in counts.items():
        table.add_row(name, str(count))
    console.print(table)
    
    # Show packets by topic
    console.print("\n[bold cyan]=== Generated Packets by Topic ===[/bold cyan]")
    
    topics_found = {"health": [], "history": [], "science": [], "other": []}
    
    for packet in packets:
        topic = packet.packet_json.get("topic", "other")
        claim = packet.packet_json.get("claim", "")[:60]
        verdict = packet.packet_json.get("verdict", "Unknown")
        
        if topic in topics_found:
            topics_found[topic].append((packet.claim_id, claim, verdict))
        else:
            topics_found["other"].append((packet.claim_id, claim, verdict))
    
    for topic, items in topics_found.items():
        if items:
            console.print(f"\n[bold magenta][{topic.upper()}][/bold magenta]")
            for claim_id, claim, verdict in items:
                verdict_color = {
                    "True": "green",
                    "False": "red",
                    "Misleading": "yellow",
                    "Depends": "blue",
                    "Unverified": "dim",
                }.get(verdict, "white")
                
                console.print(f"  [{verdict_color}]{verdict:12}[/{verdict_color}] #{claim_id}: {claim}...")
    
    # Show file locations
    console.print("\n[bold cyan]=== Output Files ===[/bold cyan]")
    output_path = Path("outputs/packets")
    if output_path.exists():
        json_files = list(output_path.glob("*.json"))
        md_files = list(output_path.glob("*.md"))
        console.print(f"  JSON files: {len(json_files)}")
        console.print(f"  Markdown files: {len(md_files)}")
        console.print(f"  Location: {output_path.absolute()}")
        
        # Show sample file
        if json_files:
            console.print(f"\n[dim]Sample output: {json_files[0]}[/dim]")


async def main():
    """Main demo function."""
    try:
        # Setup data
        await setup_demo_data()
        
        # Run pipeline
        packets_count = await run_pipeline()
        
        # Show results
        await show_results()
        
        console.print("\n[bold green]Demo complete![/bold green]")
        console.print("  You can now view packets in outputs/packets/")
        console.print("  Or start the API: uvicorn app.main:app --reload")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
