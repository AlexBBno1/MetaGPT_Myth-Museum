"""
Myth Museum - Daily Pipeline CLI

Command-line interface for running the fact-checking pipeline.
"""

import asyncio
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from core.config import ensure_directories, get_db_path, load_config
from core.db import get_connection, get_table_counts, init_db as db_init_db
from core.logging import get_logger

# Initialize Typer app
app = typer.Typer(
    name="myth-museum",
    help="Myth Museum: Automated fact-checking content pipeline",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


class OrchestratorType(str, Enum):
    """Orchestrator mode for pipeline execution."""
    LOCAL = "local"
    METAGPT = "metagpt"


class VeoProviderType(str, Enum):
    """Veo provider for video generation."""
    GEMINI = "gemini"
    VERTEX = "vertex"


# ============================================================================
# Commands
# ============================================================================


@app.command()
def init_db(
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        "-d",
        help="Path to SQLite database file",
    ),
) -> None:
    """
    Initialize the database with required schema.
    
    Creates the SQLite database file and all required tables.
    """
    ensure_directories()
    
    if db_path:
        path = Path(db_path)
    else:
        path = get_db_path()
    
    db_init_db(path)
    console.print(f"[green]OK[/green] Database initialized at: {path}")


@app.command()
def status() -> None:
    """
    Show database status and table counts.
    """
    db_path = get_db_path()
    
    if not db_path.exists():
        console.print("[yellow]Database not found. Run 'init-db' first.[/yellow]")
        return
    
    with get_connection(db_path) as conn:
        counts = get_table_counts(conn)
    
    table = Table(title="Myth Museum Database Status")
    table.add_column("Table", style="cyan")
    table.add_column("Count", style="green", justify="right")
    
    for table_name, count in counts.items():
        table.add_row(table_name, str(count))
    
    console.print(table)
    console.print(f"\nDatabase: {db_path}")


@app.command()
def ingest(
    source_type: str = typer.Option(
        "all",
        "--source-type",
        "-s",
        help="Type of sources to ingest: rss, wiki, factcheck, or all",
    ),
) -> None:
    """
    Ingest data from configured sources.
    
    Fetches content from RSS feeds, Wikipedia, and fact-check sources.
    """
    console.print("[yellow]Ingesting from sources...[/yellow]")
    
    # Import here to avoid circular imports
    from ingest.rss import ingest_all_rss
    from ingest.wiki import ingest_wiki_topics
    from ingest.factcheck import ingest_factcheck_sources
    
    config = load_config()
    db_path = get_db_path()
    
    with get_connection(db_path) as conn:
        total_ingested = 0
        
        if source_type in ("all", "rss"):
            count = asyncio.run(ingest_all_rss(conn, config))
            console.print(f"  RSS: {count} items")
            total_ingested += count
        
        if source_type in ("all", "wiki"):
            count = asyncio.run(ingest_wiki_topics(conn, config))
            console.print(f"  Wikipedia: {count} items")
            total_ingested += count
        
        if source_type in ("all", "factcheck"):
            count = asyncio.run(ingest_factcheck_sources(conn, config))
            console.print(f"  Factcheck: {count} items")
            total_ingested += count
        
        conn.commit()
    
    console.print(f"[green]OK[/green] Ingested {total_ingested} new items")


@app.command()
def run_daily(
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of claims to process",
    ),
    min_score: int = typer.Option(
        50,
        "--min-score",
        "-m",
        help="Minimum claim score to process (0-100)",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Filter by topic (health, history, science, etc.)",
    ),
    use_llm_extract: bool = typer.Option(
        False,
        "--use-llm-extract",
        help="Use LLM for claim extraction",
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--use-llm-judge",
        help="Use LLM for verdict generation",
    ),
    use_llm_script: bool = typer.Option(
        False,
        "--use-llm-script",
        help="Use LLM for script generation",
    ),
    orchestrator: OrchestratorType = typer.Option(
        OrchestratorType.LOCAL,
        "--orchestrator",
        "-o",
        help="Orchestration mode: local or metagpt",
    ),
    skip_ingest: bool = typer.Option(
        False,
        "--skip-ingest",
        help="Skip the ingestion step",
    ),
    export_shorts: bool = typer.Option(
        False,
        "--export-shorts",
        help="Export shorts production folders after generating scripts",
    ),
    make_queue: bool = typer.Option(
        False,
        "--make-queue",
        help="Create shorts queue after exporting (requires --export-shorts)",
    ),
    prepare_queue: bool = typer.Option(
        False,
        "--prepare-queue",
        help="Prepare queue items (export + TTS) after creating queue",
    ),
    render_shorts: bool = typer.Option(
        False,
        "--render-shorts",
        help="Render ready shorts to final.mp4 (requires ffmpeg)",
    ),
    generate_broll: bool = typer.Option(
        False,
        "--generate-broll",
        help="Generate Veo B-roll video clips (requires Veo API)",
    ),
    compose_video: bool = typer.Option(
        False,
        "--compose-video",
        help="Compose video shorts from B-roll clips",
    ),
    veo_provider: VeoProviderType = typer.Option(
        VeoProviderType.GEMINI,
        "--veo-provider",
        help="Veo provider: gemini or vertex",
    ),
) -> None:
    """
    Run the full daily pipeline.
    
    Executes: ingest -> extract_claims -> build_evidence -> judge -> generate_scripts
    Optionally: export shorts, create queue, prepare items, render videos.
    Optionally: generate B-roll with Veo, compose video shorts.
    """
    console.print("[bold cyan]=== Myth Museum Daily Pipeline ===[/bold cyan]\n")
    
    config = load_config()
    db_path = get_db_path()
    
    # Ensure database exists
    if not db_path.exists():
        console.print("[yellow]Database not found, initializing...[/yellow]")
        db_init_db(db_path)
    
    # Auto-enable dependencies for the pipeline chain
    # render_shorts -> prepare_queue -> make_queue -> export_shorts
    if render_shorts and not prepare_queue:
        console.print("[yellow]Note: --render-shorts requires --prepare-queue. Enabling it.[/yellow]")
        prepare_queue = True
    
    if prepare_queue and not make_queue:
        console.print("[yellow]Note: --prepare-queue requires --make-queue. Enabling it.[/yellow]")
        make_queue = True
    
    if make_queue and not export_shorts:
        console.print("[yellow]Note: --make-queue requires --export-shorts. Enabling it.[/yellow]")
        export_shorts = True
    
    # Veo pipeline dependencies
    # compose_video -> generate_broll
    if compose_video and not generate_broll:
        console.print("[yellow]Note: --compose-video requires --generate-broll. Enabling it.[/yellow]")
        generate_broll = True
    
    if export_shorts or make_queue or prepare_queue or render_shorts or generate_broll or compose_video:
        console.print("")  # Add spacing after warnings
    
    # Run pipeline based on orchestrator mode
    if orchestrator == OrchestratorType.LOCAL:
        _run_local_pipeline(
            config=config,
            db_path=db_path,
            limit=limit,
            min_score=min_score,
            topic=topic,
            use_llm_extract=use_llm_extract,
            use_llm_judge=use_llm_judge,
            use_llm_script=use_llm_script,
            skip_ingest=skip_ingest,
            export_shorts=export_shorts,
            make_queue=make_queue,
            prepare_queue=prepare_queue,
            render_shorts=render_shorts,
            generate_broll=generate_broll,
            compose_video=compose_video,
            veo_provider=veo_provider,
        )
    else:
        _run_metagpt_pipeline(
            config=config,
            db_path=db_path,
            limit=limit,
            min_score=min_score,
            topic=topic,
            export_shorts=export_shorts,
            make_queue=make_queue,
            prepare_queue=prepare_queue,
            render_shorts=render_shorts,
            generate_broll=generate_broll,
            compose_video=compose_video,
            veo_provider=veo_provider,
        )


def _run_local_pipeline(
    config: dict,
    db_path: Path,
    limit: int,
    min_score: int,
    topic: Optional[str],
    use_llm_extract: bool,
    use_llm_judge: bool,
    use_llm_script: bool,
    skip_ingest: bool,
    export_shorts: bool = False,
    make_queue: bool = False,
    prepare_queue: bool = False,
    render_shorts: bool = False,
    generate_broll: bool = False,
    compose_video: bool = False,
    veo_provider: VeoProviderType = VeoProviderType.GEMINI,
) -> None:
    """Run the pipeline using local Python functions."""
    from pipeline.extract_claims import process_raw_items
    from pipeline.build_evidence import process_claims_for_evidence
    from pipeline.judge_claim import process_claims_for_verdict
    from pipeline.generate_scripts import process_verdicts_for_scripts
    
    # Determine total steps based on options
    total_steps = 5
    if export_shorts:
        total_steps += 1
    if make_queue:
        total_steps += 1
    if prepare_queue:
        total_steps += 1
    if render_shorts:
        total_steps += 1
    if generate_broll:
        total_steps += 1
    if compose_video:
        total_steps += 1
    
    with get_connection(db_path) as conn:
        stats = {"raw_items": 0, "claims": 0, "evidence": 0, "verdicts": 0, "packets": 0, "shorts": 0}
        
        # Step 1: Ingest
        if not skip_ingest:
            console.print(f"[bold]Step 1/{total_steps}: Ingesting sources...[/bold]")
            from ingest.rss import ingest_all_rss
            from ingest.wiki import ingest_wiki_topics
            from ingest.factcheck import ingest_factcheck_sources
            
            stats["raw_items"] += asyncio.run(ingest_all_rss(conn, config))
            stats["raw_items"] += asyncio.run(ingest_wiki_topics(conn, config))
            stats["raw_items"] += asyncio.run(ingest_factcheck_sources(conn, config))
            conn.commit()
            console.print(f"  -> Ingested {stats['raw_items']} new items\n")
        else:
            console.print(f"[bold]Step 1/{total_steps}: Skipping ingestion[/bold]\n")
        
        # Step 2: Extract claims
        console.print(f"[bold]Step 2/{total_steps}: Extracting claims...[/bold]")
        stats["claims"] = asyncio.run(process_raw_items(
            conn, 
            use_llm=use_llm_extract,
            limit=limit,
        ))
        conn.commit()
        console.print(f"  -> Extracted {stats['claims']} claims\n")
        
        # Step 3: Build evidence
        console.print(f"[bold]Step 3/{total_steps}: Gathering evidence...[/bold]")
        stats["evidence"] = asyncio.run(process_claims_for_evidence(
            conn,
            min_score=min_score,
            topic=topic,
            limit=limit,
        ))
        conn.commit()
        console.print(f"  -> Gathered evidence for {stats['evidence']} claims\n")
        
        # Step 4: Judge claims
        console.print(f"[bold]Step 4/{total_steps}: Generating verdicts...[/bold]")
        stats["verdicts"] = asyncio.run(process_claims_for_verdict(
            conn,
            use_llm=use_llm_judge,
            limit=limit,
        ))
        conn.commit()
        console.print(f"  -> Generated {stats['verdicts']} verdicts\n")
        
        # Step 5: Generate scripts
        console.print(f"[bold]Step 5/{total_steps}: Creating packets...[/bold]")
        stats["packets"] = asyncio.run(process_verdicts_for_scripts(
            conn,
            use_llm=use_llm_script,
            limit=limit,
        ))
        conn.commit()
        console.print(f"  -> Created {stats['packets']} packets\n")
        
        # Step 6: Export shorts (optional)
        current_step = 5
        if export_shorts:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Exporting shorts folders...[/bold]")
            from pipeline.export_shorts_pack import process_packets_for_shorts
            
            shorts_output_dir = Path("outputs/shorts")
            stats["shorts"] = asyncio.run(process_packets_for_shorts(
                conn=conn,
                limit=limit,
                topic=topic,
                from_files=False,
                output_dir=shorts_output_dir,
            ))
            console.print(f"  -> Exported {stats['shorts']} shorts folders to {shorts_output_dir}\n")
        
        # Step 7: Create shorts queue (optional, must be after export_shorts)
        queue_result = None
        queue_date = date.today().isoformat()
        if make_queue:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Creating shorts queue...[/bold]")
            from pipeline.select_for_shorts import create_daily_queue
            
            queue_result = create_daily_queue(
                queue_date=queue_date,
                limit=limit,
                min_confidence=0.5,
                topic_mix=True,
                from_files=False,
            )
            stats["queue"] = queue_result.get("count", 0)
            stats["queue_path"] = queue_result.get("csv_path")
            stats["queue_topic_stats"] = queue_result.get("topic_stats", {})
            console.print(f"  -> Created queue with {stats['queue']} items\n")
        
        # Step 8: Prepare queue items (optional, must be after make_queue)
        if prepare_queue:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Preparing queue items...[/bold]")
            from pipeline.prepare_shorts import prepare_queue as do_prepare_queue
            
            prepare_result = asyncio.run(do_prepare_queue(
                queue_date=queue_date,
                limit=limit,
            ))
            stats["prepared"] = prepare_result.get("processed", 0)
            stats["prepare_errors"] = prepare_result.get("errors", 0)
            console.print(f"  -> Prepared {stats['prepared']} items ({stats['prepare_errors']} errors)\n")
        
        # Step 9: Render shorts (optional, must be after prepare_queue)
        if render_shorts:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Rendering shorts videos...[/bold]")
            from pipeline.render_basic_short import render_from_queue, check_ffmpeg, FFmpegNotFoundError
            
            try:
                check_ffmpeg()
                render_result = render_from_queue(
                    queue_date=queue_date,
                    limit=limit,
                )
                stats["rendered"] = render_result.get("rendered", 0)
                stats["render_failed"] = render_result.get("failed", 0)
                console.print(f"  -> Rendered {stats['rendered']} videos ({stats['render_failed']} failed)\n")
            except FFmpegNotFoundError as e:
                console.print(f"[red]ffmpeg not available: {e}[/red]")
                stats["rendered"] = 0
                stats["render_failed"] = 0
        
        # Step 10: Generate B-roll with Veo (optional)
        if generate_broll:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Generating Veo B-roll...[/bold]")
            
            from pipeline.veo import UnifiedVeoProvider, VeoProvider as VeoProviderEnum
            from pipeline.shotlist_generator import generate_shotlist_with_llm, write_shotlist_csv
            
            provider_enum = VeoProviderEnum.GEMINI if veo_provider == VeoProviderType.GEMINI else VeoProviderEnum.VERTEX
            veo = UnifiedVeoProvider(preferred_provider=provider_enum)
            
            available = veo.get_available_providers()
            if not available:
                console.print(f"  [red]No Veo providers available![/red]")
                stats["broll_generated"] = 0
            else:
                console.print(f"  Available providers: {', '.join(available)}")
                
                # Find folders that need B-roll
                shorts_dir = Path("outputs/shorts")
                broll_count = 0
                
                for folder in sorted(shorts_dir.iterdir()):
                    if not folder.is_dir():
                        continue
                    
                    # Skip folders that already have broll
                    broll_dir = folder / "broll"
                    if broll_dir.exists() and list(broll_dir.glob("broll_*.mp4")):
                        continue
                    
                    # Check for voiceover (required)
                    voiceover_path = folder / "voiceover.mp3"
                    if not voiceover_path.exists():
                        continue
                    
                    # Generate shotlist if needed
                    shotlist_path = folder / "shotlist.csv"
                    if not shotlist_path.exists():
                        script_path = folder / "voiceover.txt"
                        if script_path.exists():
                            script_text = script_path.read_text(encoding="utf-8")
                            topic_name = folder.name.replace("_", " ").replace("-", " ")
                            
                            console.print(f"  Generating shotlist for {folder.name}...")
                            entries = asyncio.run(generate_shotlist_with_llm(
                                script=script_text,
                                topic=topic_name,
                                duration_hint=60.0,
                                num_shots=4,
                            ))
                            
                            if entries:
                                write_shotlist_csv(entries, shotlist_path)
                    
                    # Generate B-roll
                    if shotlist_path.exists():
                        console.print(f"  Generating B-roll for {folder.name}...")
                        try:
                            results = asyncio.run(veo.generate_broll_for_folder(folder, shotlist_path))
                            success = sum(1 for r in results if r.success)
                            broll_count += success
                            console.print(f"    -> {success}/{len(results)} clips generated")
                        except Exception as e:
                            console.print(f"    [red]Error: {e}[/red]")
                    
                    if broll_count >= limit:
                        break
                
                stats["broll_generated"] = broll_count
                console.print(f"  -> Generated {broll_count} B-roll clips\n")
        
        # Step 11: Compose video shorts (optional, after B-roll)
        if compose_video:
            current_step += 1
            console.print(f"[bold]Step {current_step}/{total_steps}: Composing video shorts...[/bold]")
            
            from pipeline.compose_short import compose_video_short, check_ffmpeg
            
            if not check_ffmpeg():
                console.print(f"  [red]FFmpeg not available![/red]")
                stats["video_composed"] = 0
            else:
                shorts_dir = Path("outputs/shorts")
                compose_count = 0
                
                for folder in sorted(shorts_dir.iterdir()):
                    if not folder.is_dir():
                        continue
                    
                    # Skip if already has final.mp4
                    final_path = folder / "final.mp4"
                    if final_path.exists():
                        continue
                    
                    # Check for required assets
                    voiceover_path = folder / "voiceover.mp3"
                    broll_dir = folder / "broll"
                    
                    if not voiceover_path.exists():
                        continue
                    
                    # Compose video (will use fallback if no B-roll)
                    console.print(f"  Composing {folder.name}...")
                    try:
                        result = compose_video_short(folder)
                        if result.success:
                            compose_count += 1
                            console.print(f"    -> Success: {result.final_path}")
                        else:
                            console.print(f"    [red]Error: {result.error}[/red]")
                    except Exception as e:
                        console.print(f"    [red]Error: {e}[/red]")
                    
                    if compose_count >= limit:
                        break
                
                stats["video_composed"] = compose_count
                console.print(f"  -> Composed {compose_count} video shorts\n")
    
    # Print summary
    _print_summary(stats, db_path, export_shorts=export_shorts, make_queue=make_queue, 
                   prepare_queue=prepare_queue, render_shorts=render_shorts,
                   generate_broll=generate_broll, compose_video=compose_video)


def _run_metagpt_pipeline(
    config: dict,
    db_path: Path,
    limit: int,
    min_score: int,
    topic: Optional[str],
    export_shorts: bool = False,
    make_queue: bool = False,
    prepare_queue: bool = False,
    render_shorts: bool = False,
    generate_broll: bool = False,
    compose_video: bool = False,
    veo_provider: VeoProviderType = VeoProviderType.GEMINI,
) -> None:
    """Run the pipeline using MetaGPT orchestration."""
    try:
        from metagpt_integration.orchestrator import process_claims_with_fallback
        from core.constants import ClaimStatusEnum
        from core.db import get_claims_by_status
        
        console.print("[bold]Running MetaGPT orchestrated pipeline...[/bold]")
        console.print("[dim]Note: MetaGPT requires valid LLM API key in config[/dim]\n")
        
        shorts_count = 0
        
        with get_connection(db_path) as conn:
            # First ingest data
            console.print("[bold]Step 1: Ingesting sources...[/bold]")
            from ingest.rss import ingest_all_rss
            from ingest.wiki import ingest_wiki_topics
            from ingest.factcheck import ingest_factcheck_sources
            
            raw_count = asyncio.run(ingest_all_rss(conn, config))
            raw_count += asyncio.run(ingest_wiki_topics(conn, config))
            raw_count += asyncio.run(ingest_factcheck_sources(conn, config))
            conn.commit()
            console.print(f"  -> Ingested {raw_count} new items\n")
            
            # Extract claims first (local, since MetaGPT handles evidence->script)
            console.print("[bold]Step 2: Extracting claims...[/bold]")
            from pipeline.extract_claims import process_raw_items
            claims_count = asyncio.run(process_raw_items(conn, limit=limit))
            conn.commit()
            console.print(f"  -> Extracted {claims_count} claims\n")
            
            # Run MetaGPT for evidence->verdict->script
            console.print("[bold]Step 3-5: MetaGPT Orchestration (Evidence -> Verdict -> Script)...[/bold]")
            processed = asyncio.run(process_claims_with_fallback(
                conn=conn,
                limit=limit,
                min_score=min_score,
                topic=topic,
                use_metagpt=True,
            ))
            conn.commit()
            console.print(f"  -> Processed {processed} claims with MetaGPT\n")
            
            # Export shorts (optional)
            if export_shorts:
                console.print("[bold]Step 6: Exporting shorts folders...[/bold]")
                from pipeline.export_shorts_pack import process_packets_for_shorts
                from pathlib import Path as PathLib
                
                shorts_output_dir = PathLib("outputs/shorts")
                shorts_count = asyncio.run(process_packets_for_shorts(
                    conn=conn,
                    limit=limit,
                    topic=topic,
                    from_files=False,
                    output_dir=shorts_output_dir,
                ))
                console.print(f"  -> Exported {shorts_count} shorts folders to {shorts_output_dir}\n")
            
            # Create queue (optional, after shorts)
            queue_result = None
            queue_count = 0
            queue_date = date.today().isoformat()
            if make_queue:
                console.print("[bold]Step 7: Creating shorts queue...[/bold]")
                from pipeline.select_for_shorts import create_daily_queue
                
                queue_result = create_daily_queue(
                    queue_date=queue_date,
                    limit=limit,
                    min_confidence=0.5,
                    topic_mix=True,
                    from_files=False,
                )
                queue_count = queue_result.get("count", 0)
                console.print(f"  -> Created queue with {queue_count} items\n")
            
            # Prepare queue items (optional)
            prepared_count = 0
            prepare_errors = 0
            if prepare_queue:
                console.print("[bold]Step 8: Preparing queue items...[/bold]")
                from pipeline.prepare_shorts import prepare_queue as do_prepare_queue
                
                prepare_result = asyncio.run(do_prepare_queue(
                    queue_date=queue_date,
                    limit=limit,
                ))
                prepared_count = prepare_result.get("processed", 0)
                prepare_errors = prepare_result.get("errors", 0)
                console.print(f"  -> Prepared {prepared_count} items ({prepare_errors} errors)\n")
            
            # Render shorts (optional)
            rendered_count = 0
            render_failed = 0
            if render_shorts:
                console.print("[bold]Step 9: Rendering shorts videos...[/bold]")
                from pipeline.render_basic_short import render_from_queue, check_ffmpeg, FFmpegNotFoundError
                
                try:
                    check_ffmpeg()
                    render_result = render_from_queue(
                        queue_date=queue_date,
                        limit=limit,
                    )
                    rendered_count = render_result.get("rendered", 0)
                    render_failed = render_result.get("failed", 0)
                    console.print(f"  -> Rendered {rendered_count} videos ({render_failed} failed)\n")
                except FFmpegNotFoundError as e:
                    console.print(f"[red]ffmpeg not available: {e}[/red]")
        
        stats = {
            "raw_items": raw_count,
            "claims": claims_count,
            "evidence": processed,
            "verdicts": processed,
            "packets": processed,
            "shorts": shorts_count,
            "queue": queue_count,
            "queue_path": queue_result.get("csv_path") if queue_result else None,
            "queue_topic_stats": queue_result.get("topic_stats", {}) if queue_result else {},
            "prepared": prepared_count,
            "prepare_errors": prepare_errors,
            "rendered": rendered_count,
            "render_failed": render_failed,
        }
        _print_summary(stats, db_path, export_shorts=export_shorts, make_queue=make_queue,
                       prepare_queue=prepare_queue, render_shorts=render_shorts,
                       generate_broll=generate_broll, compose_video=compose_video)
        
    except ImportError as e:
        console.print(f"[red]MetaGPT integration not available: {e}[/red]")
        console.print("[yellow]Falling back to local pipeline...[/yellow]")
        _run_local_pipeline(
            config=config,
            db_path=db_path,
            limit=limit,
            min_score=min_score,
            topic=topic,
            use_llm_extract=False,
            use_llm_judge=False,
            use_llm_script=False,
            skip_ingest=False,
            export_shorts=export_shorts,
            make_queue=make_queue,
            prepare_queue=prepare_queue,
            render_shorts=render_shorts,
            generate_broll=generate_broll,
            compose_video=compose_video,
            veo_provider=veo_provider,
        )
    except Exception as e:
        console.print(f"[red]MetaGPT pipeline failed: {e}[/red]")
        console.print("[yellow]Falling back to local pipeline...[/yellow]")
        _run_local_pipeline(
            config=config,
            db_path=db_path,
            limit=limit,
            min_score=min_score,
            topic=topic,
            use_llm_extract=False,
            use_llm_judge=False,
            use_llm_script=False,
            skip_ingest=False,
            export_shorts=export_shorts,
            make_queue=make_queue,
            prepare_queue=prepare_queue,
            render_shorts=render_shorts,
            generate_broll=generate_broll,
            compose_video=compose_video,
            veo_provider=veo_provider,
        )


def _print_summary(
    stats: dict, 
    db_path: Path, 
    export_shorts: bool = False, 
    make_queue: bool = False,
    prepare_queue: bool = False,
    render_shorts: bool = False,
    generate_broll: bool = False,
    compose_video: bool = False,
) -> None:
    """Print pipeline summary."""
    console.print("\n[bold cyan]=== Pipeline Summary ===[/bold cyan]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    
    table.add_row("New raw items", str(stats.get("raw_items", 0)))
    table.add_row("Claims extracted", str(stats.get("claims", 0)))
    table.add_row("Evidence gathered", str(stats.get("evidence", 0)))
    table.add_row("Verdicts generated", str(stats.get("verdicts", 0)))
    table.add_row("Packets created", str(stats.get("packets", 0)))
    
    if export_shorts:
        table.add_row("Shorts exported", str(stats.get("shorts", 0)))
    
    if make_queue:
        table.add_row("Queue items", str(stats.get("queue", 0)))
    
    if prepare_queue:
        table.add_row("Items prepared", str(stats.get("prepared", 0)))
        if stats.get("prepare_errors", 0) > 0:
            table.add_row("Prepare errors", str(stats.get("prepare_errors", 0)))
    
    if render_shorts:
        table.add_row("Videos rendered", str(stats.get("rendered", 0)))
        if stats.get("render_failed", 0) > 0:
            table.add_row("Render failed", str(stats.get("render_failed", 0)))
    
    if generate_broll:
        table.add_row("B-roll clips generated", str(stats.get("broll_generated", 0)))
    
    if compose_video:
        table.add_row("Videos composed", str(stats.get("video_composed", 0)))
    
    console.print(table)
    
    # Show top topics from claims
    with get_connection(db_path) as conn:
        from core.db import get_claims_by_topic
        topics = get_claims_by_topic(conn)
        
        if topics:
            console.print("\n[bold]Top Topics (from claims):[/bold]")
            for topic_name, count in list(topics.items())[:3]:
                console.print(f"  - {topic_name}: {count} claims")
    
    # Show shorts output location if exported
    if export_shorts and stats.get("shorts", 0) > 0:
        console.print(f"\n[bold]Shorts Output:[/bold]")
        console.print(f"  - Location: outputs/shorts/")
        console.print(f"  - Folders: {stats.get('shorts', 0)}")
    
    # Show queue info if created
    if make_queue and stats.get("queue", 0) > 0:
        console.print(f"\n[bold]Shorts Queue:[/bold]")
        queue_path = stats.get("queue_path")
        if queue_path:
            console.print(f"  - CSV: {queue_path}")
        console.print(f"  - Items: {stats.get('queue', 0)}")
        
        # Show topic mix in queue
        queue_topic_stats = stats.get("queue_topic_stats", {})
        if queue_topic_stats:
            console.print(f"  - Topic mix:")
            for topic_name, count in sorted(queue_topic_stats.items(), key=lambda x: -x[1]):
                console.print(f"      {topic_name}: {count}")
    
    # Show render results
    if render_shorts and stats.get("rendered", 0) > 0:
        console.print(f"\n[bold]Rendered Videos:[/bold]")
        console.print(f"  - Location: outputs/shorts/*/final.mp4")
        console.print(f"  - Count: {stats.get('rendered', 0)}")
    
    console.print(f"\n[green]OK[/green] Pipeline complete!")


@app.command()
def export(
    claim_id: Optional[int] = typer.Option(
        None,
        "--claim-id",
        "-c",
        help="Export specific claim by ID",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for packets",
    ),
) -> None:
    """
    Export packets to JSON and Markdown files.
    """
    from core.config import get_output_path
    from core.db import get_all_packets, get_packet_by_claim
    from pipeline.generate_scripts import export_packet_json, export_packet_md
    
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = get_output_path()
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    db_path = get_db_path()
    
    with get_connection(db_path) as conn:
        if claim_id:
            packet = get_packet_by_claim(conn, claim_id)
            if packet:
                packets = [packet]
            else:
                console.print(f"[red]No packet found for claim {claim_id}[/red]")
                return
        else:
            packets = get_all_packets(conn)
        
        for packet in packets:
            export_packet_json(packet, out_path)
            export_packet_md(packet, out_path)
            console.print(f"  Exported claim {packet.claim_id}")
    
    console.print(f"[green]OK[/green] Exported {len(packets)} packets to {out_path}")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
