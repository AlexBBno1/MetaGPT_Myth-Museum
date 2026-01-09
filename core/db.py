"""
Myth Museum - Database

SQLite database schema and helper functions.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from core.config import get_db_path
from core.logging import get_logger
from core.models import Claim, Evidence, Packet, RawItem, SourceConfig, Verdict

logger = get_logger(__name__)


# ============================================================================
# Schema Definitions
# ============================================================================


SCHEMA_SQL = """
-- Sources table
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config_json TEXT DEFAULT '{}',
    enabled INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Raw items table (ingested content)
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    hash TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(id),
    UNIQUE(url, hash)
);

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    topic TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    score INTEGER DEFAULT 0,
    status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (raw_item_id) REFERENCES raw_items(id)
);

-- Evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT DEFAULT 'rss',
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    credibility_score INTEGER DEFAULT 50,
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

-- Verdicts table
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL UNIQUE,
    verdict TEXT NOT NULL,
    explanation_json TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

-- Packets table
CREATE TABLE IF NOT EXISTS packets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL UNIQUE,
    packet_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_items_source ON raw_items(source_id);
CREATE INDEX IF NOT EXISTS idx_raw_items_hash ON raw_items(hash);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_topic ON claims(topic);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
"""


# ============================================================================
# Connection Management
# ============================================================================


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection.
    
    Args:
        db_path: Path to database file (uses config default if None)
    
    Returns:
        SQLite connection
    """
    if db_path is None:
        db_path = get_db_path()
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn


@contextmanager
def get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Args:
        db_path: Path to database file
    
    Yields:
        SQLite connection
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================================
# Initialization
# ============================================================================


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database with schema.
    
    Args:
        db_path: Path to database file (uses config default if None)
    """
    if db_path is None:
        db_path = get_db_path()
    
    if isinstance(db_path, str):
        db_path = Path(db_path)
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    
    logger.info(f"Database initialized at {db_path}")


# ============================================================================
# CRUD Operations - Sources
# ============================================================================


def insert_source(conn: sqlite3.Connection, source: SourceConfig) -> int:
    """Insert a source and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO sources (name, type, config_json, enabled)
        VALUES (?, ?, ?, ?)
        """,
        (source.name, source.type.value, json.dumps(source.config_json), int(source.enabled)),
    )
    return cursor.lastrowid


def get_sources_db(conn: sqlite3.Connection, enabled_only: bool = True) -> list[SourceConfig]:
    """Get all sources from database."""
    query = "SELECT * FROM sources"
    if enabled_only:
        query += " WHERE enabled = 1"
    
    rows = conn.execute(query).fetchall()
    sources = []
    for row in rows:
        from core.constants import SourceTypeEnum
        sources.append(SourceConfig(
            id=row["id"],
            name=row["name"],
            type=SourceTypeEnum(row["type"]),
            config_json=json.loads(row["config_json"]),
            enabled=bool(row["enabled"]),
        ))
    return sources


# ============================================================================
# CRUD Operations - Raw Items
# ============================================================================


def insert_raw_item(conn: sqlite3.Connection, item: RawItem) -> Optional[int]:
    """
    Insert a raw item. Returns ID or None if duplicate.
    """
    try:
        cursor = conn.execute(
            """
            INSERT INTO raw_items (source_id, url, title, content, published_at, fetched_at, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.source_id,
                item.url,
                item.title,
                item.content,
                item.published_at.isoformat() if item.published_at else None,
                item.fetched_at.isoformat(),
                item.hash,
            ),
        )
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate URL+hash
        return None


def get_raw_items_by_source(
    conn: sqlite3.Connection,
    source_id: int,
    limit: int = 100,
) -> list[RawItem]:
    """Get raw items for a source."""
    rows = conn.execute(
        "SELECT * FROM raw_items WHERE source_id = ? ORDER BY fetched_at DESC LIMIT ?",
        (source_id, limit),
    ).fetchall()
    return [_row_to_raw_item(row) for row in rows]


def get_unprocessed_raw_items(conn: sqlite3.Connection, limit: int = 100) -> list[RawItem]:
    """Get raw items that haven't been processed into claims yet."""
    rows = conn.execute(
        """
        SELECT ri.* FROM raw_items ri
        LEFT JOIN claims c ON ri.id = c.raw_item_id
        WHERE c.id IS NULL
        ORDER BY ri.fetched_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_row_to_raw_item(row) for row in rows]


def _row_to_raw_item(row: sqlite3.Row) -> RawItem:
    """Convert database row to RawItem."""
    return RawItem(
        id=row["id"],
        source_id=row["source_id"],
        url=row["url"],
        title=row["title"],
        content=row["content"],
        published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        hash=row["hash"],
    )


# ============================================================================
# CRUD Operations - Claims
# ============================================================================


def insert_claim(conn: sqlite3.Connection, claim: Claim) -> int:
    """Insert a claim and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO claims (raw_item_id, claim_text, topic, language, score, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim.raw_item_id,
            claim.claim_text,
            claim.topic,
            claim.language,
            claim.score,
            claim.status,
            claim.created_at.isoformat(),
        ),
    )
    return cursor.lastrowid


def update_claim_status(conn: sqlite3.Connection, claim_id: int, status: str) -> None:
    """Update claim status."""
    conn.execute(
        "UPDATE claims SET status = ? WHERE id = ?",
        (status, claim_id),
    )


def get_claims_by_status(
    conn: sqlite3.Connection,
    status: str,
    limit: int = 100,
    min_score: int = 0,
) -> list[Claim]:
    """Get claims with a specific status."""
    rows = conn.execute(
        """
        SELECT * FROM claims 
        WHERE status = ? AND score >= ?
        ORDER BY score DESC, created_at DESC
        LIMIT ?
        """,
        (status, min_score, limit),
    ).fetchall()
    return [_row_to_claim(row) for row in rows]


def get_pending_claims(
    conn: sqlite3.Connection,
    limit: int = 100,
    min_score: int = 0,
) -> list[Claim]:
    """Get claims ready for evidence gathering (status = 'new')."""
    return get_claims_by_status(conn, "new", limit, min_score)


def _row_to_claim(row: sqlite3.Row) -> Claim:
    """Convert database row to Claim."""
    return Claim(
        id=row["id"],
        raw_item_id=row["raw_item_id"],
        claim_text=row["claim_text"],
        topic=row["topic"],
        language=row["language"],
        score=row["score"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ============================================================================
# CRUD Operations - Evidence
# ============================================================================


def insert_evidence(conn: sqlite3.Connection, evidence: Evidence) -> int:
    """Insert evidence and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO evidence (claim_id, query, source_name, source_type, url, title, snippet, published_at, fetched_at, credibility_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence.claim_id,
            evidence.query,
            evidence.source_name,
            evidence.source_type,
            evidence.url,
            evidence.title,
            evidence.snippet,
            evidence.published_at.isoformat() if evidence.published_at else None,
            evidence.fetched_at.isoformat(),
            evidence.credibility_score,
        ),
    )
    return cursor.lastrowid


def get_evidence_by_claim(conn: sqlite3.Connection, claim_id: int) -> list[Evidence]:
    """Get all evidence for a claim."""
    rows = conn.execute(
        "SELECT * FROM evidence WHERE claim_id = ? ORDER BY credibility_score DESC",
        (claim_id,),
    ).fetchall()
    return [_row_to_evidence(row) for row in rows]


def _row_to_evidence(row: sqlite3.Row) -> Evidence:
    """Convert database row to Evidence."""
    return Evidence(
        id=row["id"],
        claim_id=row["claim_id"],
        query=row["query"],
        source_name=row["source_name"],
        source_type=row["source_type"],
        url=row["url"],
        title=row["title"],
        snippet=row["snippet"],
        published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        credibility_score=row["credibility_score"],
    )


# ============================================================================
# CRUD Operations - Verdicts
# ============================================================================


def insert_verdict(conn: sqlite3.Connection, verdict: Verdict) -> int:
    """Insert or update verdict and return its ID."""
    # Use INSERT OR REPLACE to handle updates
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO verdicts (claim_id, verdict, explanation_json, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            verdict.claim_id,
            verdict.verdict,
            json.dumps(verdict.explanation_json),
            verdict.confidence,
            verdict.created_at.isoformat(),
        ),
    )
    return cursor.lastrowid


def get_verdict_by_claim(conn: sqlite3.Connection, claim_id: int) -> Optional[Verdict]:
    """Get verdict for a claim."""
    row = conn.execute(
        "SELECT * FROM verdicts WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    
    if row is None:
        return None
    
    return _row_to_verdict(row)


def _row_to_verdict(row: sqlite3.Row) -> Verdict:
    """Convert database row to Verdict."""
    return Verdict(
        id=row["id"],
        claim_id=row["claim_id"],
        verdict=row["verdict"],
        explanation_json=json.loads(row["explanation_json"]),
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ============================================================================
# CRUD Operations - Packets
# ============================================================================


def insert_packet(conn: sqlite3.Connection, packet: Packet) -> int:
    """Insert or update packet and return its ID."""
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO packets (claim_id, packet_json, created_at)
        VALUES (?, ?, ?)
        """,
        (
            packet.claim_id,
            json.dumps(packet.packet_json),
            packet.created_at.isoformat(),
        ),
    )
    return cursor.lastrowid


def get_packet_by_claim(conn: sqlite3.Connection, claim_id: int) -> Optional[Packet]:
    """Get packet for a claim."""
    row = conn.execute(
        "SELECT * FROM packets WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    
    if row is None:
        return None
    
    return _row_to_packet(row)


def get_all_packets(conn: sqlite3.Connection, limit: int = 100) -> list[Packet]:
    """Get all packets."""
    rows = conn.execute(
        "SELECT * FROM packets ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_packet(row) for row in rows]


def _row_to_packet(row: sqlite3.Row) -> Packet:
    """Convert database row to Packet."""
    return Packet(
        id=row["id"],
        claim_id=row["claim_id"],
        packet_json=json.loads(row["packet_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ============================================================================
# Statistics
# ============================================================================


def get_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Get row counts for all tables."""
    tables = ["sources", "raw_items", "claims", "evidence", "verdicts", "packets"]
    counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()
        counts[table] = row["count"]
    return counts


def get_claims_by_topic(conn: sqlite3.Connection) -> dict[str, int]:
    """Get claim counts by topic."""
    rows = conn.execute(
        "SELECT topic, COUNT(*) as count FROM claims GROUP BY topic ORDER BY count DESC"
    ).fetchall()
    return {row["topic"]: row["count"] for row in rows}
