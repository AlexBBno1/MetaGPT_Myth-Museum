"""
Tests for core.db module.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from core.db import (
    get_connection,
    get_table_counts,
    init_db,
    insert_claim,
    insert_evidence,
    insert_raw_item,
    insert_verdict,
    get_claims_by_status,
    get_evidence_by_claim,
    get_verdict_by_claim,
)
from core.models import Claim, Evidence, RawItem, Verdict
from core.textnorm import compute_hash


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = Path(f.name)
    
    init_db(db_path)
    yield db_path
    
    # Cleanup - ignore errors on Windows due to file locking
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows file locking - will be cleaned up by OS


@pytest.fixture
def db_conn(temp_db):
    """Get a database connection."""
    conn = get_connection(temp_db)
    yield conn
    conn.close()


class TestInitDb:
    """Tests for init_db function."""
    
    def test_init_db_creates_tables(self, temp_db):
        """Test that init_db creates all required tables."""
        with get_connection(temp_db) as conn:
            counts = get_table_counts(conn)
        
        assert "sources" in counts
        assert "raw_items" in counts
        assert "claims" in counts
        assert "evidence" in counts
        assert "verdicts" in counts
        assert "packets" in counts
    
    def test_init_db_is_idempotent(self, temp_db):
        """Test that init_db can be called multiple times."""
        # Should not raise
        init_db(temp_db)
        init_db(temp_db)
        
        with get_connection(temp_db) as conn:
            counts = get_table_counts(conn)
        
        assert len(counts) == 6


class TestRawItems:
    """Tests for raw_items operations."""
    
    def test_insert_raw_item(self, db_conn):
        """Test inserting a raw item."""
        item = RawItem(
            source_id=1,
            url="https://example.com/article1",
            title="Test Article",
            content="This is test content.",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            hash=compute_hash("This is test content."),
        )
        
        item_id = insert_raw_item(db_conn, item)
        db_conn.commit()
        
        assert item_id is not None
        assert item_id > 0
    
    def test_duplicate_raw_item_rejected(self, db_conn):
        """Test that duplicate URL+hash is rejected."""
        content = "This is test content."
        item = RawItem(
            source_id=1,
            url="https://example.com/article1",
            title="Test Article",
            content=content,
            hash=compute_hash(content),
        )
        
        # First insert should succeed
        item_id1 = insert_raw_item(db_conn, item)
        db_conn.commit()
        assert item_id1 is not None
        
        # Second insert with same URL+hash should fail
        item_id2 = insert_raw_item(db_conn, item)
        db_conn.commit()
        assert item_id2 is None


class TestClaims:
    """Tests for claims operations."""
    
    def test_insert_claim(self, db_conn):
        """Test inserting a claim."""
        # First insert a raw item
        item = RawItem(
            source_id=1,
            url="https://example.com/article1",
            title="Test Article",
            content="Test content",
            hash=compute_hash("Test content"),
        )
        raw_item_id = insert_raw_item(db_conn, item)
        db_conn.commit()
        
        # Now insert a claim
        claim = Claim(
            raw_item_id=raw_item_id,
            claim_text="Drinking 8 glasses of water is necessary.",
            topic="health",
            language="en",
            score=75,
            status="new",
        )
        
        claim_id = insert_claim(db_conn, claim)
        db_conn.commit()
        
        assert claim_id is not None
        assert claim_id > 0
    
    def test_get_claims_by_status(self, db_conn):
        """Test querying claims by status."""
        # Insert raw item
        item = RawItem(
            source_id=1,
            url="https://example.com/article1",
            title="Test",
            content="Test",
            hash=compute_hash("Test"),
        )
        raw_item_id = insert_raw_item(db_conn, item)
        db_conn.commit()
        
        # Insert claims with different statuses
        claim1 = Claim(raw_item_id=raw_item_id, claim_text="Claim 1", topic="health", score=80, status="new")
        claim2 = Claim(raw_item_id=raw_item_id, claim_text="Claim 2", topic="health", score=60, status="new")
        claim3 = Claim(raw_item_id=raw_item_id, claim_text="Claim 3", topic="health", score=90, status="judged")
        
        insert_claim(db_conn, claim1)
        insert_claim(db_conn, claim2)
        insert_claim(db_conn, claim3)
        db_conn.commit()
        
        new_claims = get_claims_by_status(db_conn, "new")
        assert len(new_claims) == 2
        
        judged_claims = get_claims_by_status(db_conn, "judged")
        assert len(judged_claims) == 1


class TestEvidence:
    """Tests for evidence operations."""
    
    def test_insert_evidence(self, db_conn):
        """Test inserting evidence."""
        # Setup: insert raw item and claim
        item = RawItem(source_id=1, url="https://example.com", title="T", content="C", hash="h")
        raw_item_id = insert_raw_item(db_conn, item)
        claim = Claim(raw_item_id=raw_item_id, claim_text="Test", topic="health", score=80, status="new")
        claim_id = insert_claim(db_conn, claim)
        db_conn.commit()
        
        # Insert evidence
        evidence = Evidence(
            claim_id=claim_id,
            query="test query",
            source_name="Wikipedia",
            source_type="wikipedia",
            url="https://en.wikipedia.org/wiki/Test",
            title="Test Article",
            snippet="This is a test snippet.",
            credibility_score=70,
        )
        
        evidence_id = insert_evidence(db_conn, evidence)
        db_conn.commit()
        
        assert evidence_id is not None
        assert evidence_id > 0
        
        # Query back
        evidence_list = get_evidence_by_claim(db_conn, claim_id)
        assert len(evidence_list) == 1
        assert evidence_list[0].source_name == "Wikipedia"


class TestVerdicts:
    """Tests for verdicts operations."""
    
    def test_insert_verdict(self, db_conn):
        """Test inserting a verdict."""
        # Setup
        item = RawItem(source_id=1, url="https://example.com", title="T", content="C", hash="h")
        raw_item_id = insert_raw_item(db_conn, item)
        claim = Claim(raw_item_id=raw_item_id, claim_text="Test", topic="health", score=80, status="new")
        claim_id = insert_claim(db_conn, claim)
        db_conn.commit()
        
        # Insert verdict
        verdict = Verdict(
            claim_id=claim_id,
            verdict="False",
            explanation_json={"one_line_verdict": "This claim is false."},
            confidence=0.85,
        )
        
        verdict_id = insert_verdict(db_conn, verdict)
        db_conn.commit()
        
        assert verdict_id is not None
        
        # Query back
        result = get_verdict_by_claim(db_conn, claim_id)
        assert result is not None
        assert result.verdict == "False"
        assert result.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
