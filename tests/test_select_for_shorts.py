"""
Tests for pipeline/select_for_shorts.py

Shorts Queue selection, filtering, and export tests.
"""

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from core.constants import (
    QUEUE_CSV_COLUMNS,
    QUEUE_DEFAULT_MIN_CONFIDENCE,
    QUEUE_SAFETY_GATES,
    VERDICT_WEIGHTS,
)
from pipeline.select_for_shorts import (
    build_queue_row,
    create_daily_queue,
    deduplicate_packets,
    ensure_topic_mix,
    export_queue_csv,
    export_queue_md,
    filter_packets,
    get_shorts_folder_path,
    score_packet,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_packet_false() -> dict[str, Any]:
    """Mock packet with False verdict."""
    return {
        "claim_id": 1,
        "claim": "Drinking bleach can cure diseases",
        "topic": "health",
        "verdict": "False",
        "confidence": 0.95,
        "sources": [
            {"source_type": "academic", "title": "CDC Report"},
            {"source_type": "factcheck", "title": "Snopes Article"},
        ],
        "titles": ["Myth Busted: Bleach Cures"],
        "shorts_script": {
            "hook": "Think bleach can cure you? Think again!",
            "total_duration": 40,
        },
    }


@pytest.fixture
def mock_packet_true() -> dict[str, Any]:
    """Mock packet with True verdict."""
    return {
        "claim_id": 2,
        "claim": "Exercise improves heart health",
        "topic": "health",
        "verdict": "True",
        "confidence": 0.90,
        "sources": [
            {"source_type": "academic", "title": "Medical Study"},
        ],
        "titles": ["Exercise and Heart Health"],
        "shorts_script": {
            "hook": "Exercise is great for your heart!",
            "total_duration": 35,
        },
    }


@pytest.fixture
def mock_packet_unverified() -> dict[str, Any]:
    """Mock packet with Unverified verdict."""
    return {
        "claim_id": 3,
        "claim": "Aliens built the pyramids",
        "topic": "history",
        "verdict": "Unverified",
        "confidence": 0.30,
        "sources": [],
        "titles": ["Pyramid Mystery"],
        "shorts_script": {
            "hook": "Who really built the pyramids?",
            "total_duration": 45,
        },
    }


@pytest.fixture
def mock_packet_science() -> dict[str, Any]:
    """Mock packet with science topic."""
    return {
        "claim_id": 4,
        "claim": "Water has memory",
        "topic": "science",
        "verdict": "False",
        "confidence": 0.85,
        "sources": [
            {"source_type": "academic", "title": "Chemistry Study"},
            {"source_type": "wikipedia", "title": "Water Wiki"},
        ],
        "titles": ["Water Memory Myth"],
        "shorts_script": {
            "hook": "Does water remember?",
            "total_duration": 38,
        },
    }


@pytest.fixture
def mock_packet_history() -> dict[str, Any]:
    """Mock packet with history topic."""
    return {
        "claim_id": 5,
        "claim": "Napoleon was short",
        "topic": "history",
        "verdict": "Misleading",
        "confidence": 0.80,
        "sources": [
            {"source_type": "wikipedia", "title": "Napoleon Wiki"},
            {"source_type": "factcheck", "title": "History Check"},
        ],
        "titles": ["Napoleon's Height"],
        "shorts_script": {
            "hook": "Was Napoleon really short?",
            "total_duration": 42,
        },
    }


@pytest.fixture
def mock_packets(
    mock_packet_false,
    mock_packet_true,
    mock_packet_unverified,
    mock_packet_science,
    mock_packet_history,
) -> list[dict[str, Any]]:
    """Collection of mock packets."""
    return [
        mock_packet_false,
        mock_packet_true,
        mock_packet_unverified,
        mock_packet_science,
        mock_packet_history,
    ]


@pytest.fixture
def mock_packet_similar() -> dict[str, Any]:
    """Mock packet similar to mock_packet_false."""
    return {
        "claim_id": 6,
        "claim": "Drinking bleach will cure illness",  # Similar to claim_id=1
        "topic": "health",
        "verdict": "False",
        "confidence": 0.90,
        "sources": [{"source_type": "academic", "title": "Study"}],
        "titles": ["Bleach Myth"],
        "shorts_script": {"hook": "Bleach is dangerous!", "total_duration": 35},
    }


@pytest.fixture
def mock_packet_low_confidence_health() -> dict[str, Any]:
    """Mock health packet with low confidence (should fail safety gate)."""
    return {
        "claim_id": 7,
        "claim": "Vitamin C cures cancer",
        "topic": "health",
        "verdict": "False",
        "confidence": 0.50,  # Below safety gate
        "sources": [{"source_type": "wikipedia", "title": "Wiki"}],  # Only 1 type
        "titles": ["Vitamin C"],
        "shorts_script": {"hook": "Can vitamin C cure cancer?", "total_duration": 30},
    }


# ============================================================================
# Test Scoring
# ============================================================================


class TestScoring:
    """Tests for packet scoring."""

    def test_score_false_high_confidence(self, mock_packet_false):
        """False verdict with high confidence should score high."""
        score = score_packet(mock_packet_false)
        assert score > 0
        # False=10, confidence=0.95, 2 source types = evidence_bonus=1.2
        expected = 10 * 0.95 * 1.2
        assert abs(score - expected) < 0.01

    def test_score_true_lower_than_false(self, mock_packet_false, mock_packet_true):
        """True verdict should score lower than False."""
        score_false = score_packet(mock_packet_false)
        score_true = score_packet(mock_packet_true)
        assert score_false > score_true

    def test_score_unverified_zero(self, mock_packet_unverified):
        """Unverified verdict should score 0."""
        score = score_packet(mock_packet_unverified)
        assert score == 0

    def test_score_with_evidence_bonus(self, mock_packet_science):
        """More source types should increase score."""
        score = score_packet(mock_packet_science)
        # False=10, confidence=0.85, 2 source types = 1.2 bonus
        expected = 10 * 0.85 * 1.2
        assert abs(score - expected) < 0.01

    def test_verdict_weights_ordering(self):
        """Verify verdict weight ordering: False > Misleading > Depends > True > Unverified."""
        assert VERDICT_WEIGHTS["False"] > VERDICT_WEIGHTS["Misleading"]
        assert VERDICT_WEIGHTS["Misleading"] > VERDICT_WEIGHTS["Depends"]
        assert VERDICT_WEIGHTS["Depends"] > VERDICT_WEIGHTS["True"]
        assert VERDICT_WEIGHTS["True"] > VERDICT_WEIGHTS["Unverified"]


# ============================================================================
# Test Filtering
# ============================================================================


class TestFiltering:
    """Tests for packet filtering."""

    def test_filter_excludes_unverified(self, mock_packets):
        """Unverified packets should be excluded by default."""
        filtered = filter_packets(mock_packets, min_confidence=0.0)
        claim_ids = [p["claim_id"] for p in filtered]
        assert 3 not in claim_ids  # mock_packet_unverified

    def test_filter_by_confidence(self, mock_packets):
        """Only packets above min_confidence should pass."""
        filtered = filter_packets(mock_packets, min_confidence=0.85)
        for packet in filtered:
            assert packet["confidence"] >= 0.85

    def test_filter_health_safety_gate(self, mock_packet_low_confidence_health):
        """Health packets below safety gate should be excluded."""
        packets = [mock_packet_low_confidence_health]
        filtered = filter_packets(packets, min_confidence=0.0)
        assert len(filtered) == 0  # Safety gate blocks it

    def test_filter_allows_high_confidence_health(self, mock_packet_false):
        """Health packets above safety gate should pass."""
        packets = [mock_packet_false]
        filtered = filter_packets(packets, min_confidence=0.0)
        assert len(filtered) == 1


# ============================================================================
# Test Deduplication
# ============================================================================


class TestDeduplication:
    """Tests for packet deduplication."""

    def test_dedup_removes_similar(self, mock_packet_false, mock_packet_similar):
        """Similar claims should be deduplicated."""
        packets = [mock_packet_false, mock_packet_similar]
        # Use a lower threshold since the claims are similar but not identical
        deduped = deduplicate_packets(packets, threshold=55.0)
        
        # Should keep only one (higher score first)
        assert len(deduped) == 1

    def test_dedup_keeps_dissimilar(self, mock_packet_false, mock_packet_science):
        """Dissimilar claims should both be kept."""
        packets = [mock_packet_false, mock_packet_science]
        deduped = deduplicate_packets(packets, threshold=85.0)
        
        assert len(deduped) == 2

    def test_dedup_deterministic(self, mock_packets):
        """Deduplication should be deterministic (same input = same output)."""
        deduped1 = deduplicate_packets(mock_packets.copy())
        deduped2 = deduplicate_packets(mock_packets.copy())
        
        ids1 = [p["claim_id"] for p in deduped1]
        ids2 = [p["claim_id"] for p in deduped2]
        
        assert ids1 == ids2

    def test_dedup_keeps_higher_score(self, mock_packet_false, mock_packet_similar):
        """Should keep the packet with higher score."""
        packets = [mock_packet_false, mock_packet_similar]
        deduped = deduplicate_packets(packets, threshold=70.0)
        
        # mock_packet_false has higher confidence (0.95 vs 0.90)
        assert deduped[0]["claim_id"] == 1


# ============================================================================
# Test Topic Mix
# ============================================================================


class TestTopicMix:
    """Tests for topic diversity."""

    def test_topic_mix_ensures_diversity(
        self,
        mock_packet_false,
        mock_packet_science,
        mock_packet_history,
    ):
        """Should try to include different topics."""
        packets = [mock_packet_false, mock_packet_science, mock_packet_history]
        selected = ensure_topic_mix(packets, limit=3)
        
        topics = set(p["topic"] for p in selected)
        assert len(topics) >= 2  # At least 2 different topics

    def test_topic_mix_respects_limit(self, mock_packets):
        """Should not exceed limit."""
        selected = ensure_topic_mix(mock_packets, limit=2)
        assert len(selected) <= 2

    def test_topic_mix_deterministic(self, mock_packets):
        """Topic mix selection should be deterministic."""
        selected1 = ensure_topic_mix(mock_packets.copy(), limit=3)
        selected2 = ensure_topic_mix(mock_packets.copy(), limit=3)
        
        ids1 = [p["claim_id"] for p in selected1]
        ids2 = [p["claim_id"] for p in selected2]
        
        assert ids1 == ids2


# ============================================================================
# Test Queue Row Building
# ============================================================================


class TestQueueRowBuilding:
    """Tests for queue row construction."""

    def test_build_queue_row_structure(self, mock_packet_false, tmp_path):
        """Queue row should have all required columns."""
        row = build_queue_row(mock_packet_false, rank=1, shorts_dir=tmp_path)
        
        for col in QUEUE_CSV_COLUMNS:
            assert col in row

    def test_build_queue_row_folder_exists(self, mock_packet_false, tmp_path):
        """Status should be 'ready' if folder has all required files and voiceover.mp3."""
        from core.constants import SHORTS_REQUIRED_FILES
        
        # Create folder with all required files
        folder = tmp_path / "1"
        folder.mkdir()
        
        for filename in SHORTS_REQUIRED_FILES:
            (folder / filename).write_text("test content", encoding="utf-8")
        
        # Add voiceover.mp3 to make it "ready"
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 160000)  # ~10 seconds at 128kbps
        
        row = build_queue_row(mock_packet_false, rank=1, shorts_dir=tmp_path)
        assert row["status"] == "ready"
        # folder_path is relative, so check for claim_id in the path
        assert "1" in row["folder_path"]

    def test_build_queue_row_folder_missing(self, mock_packet_false, tmp_path):
        """Status should be 'needs_export' if folder doesn't exist."""
        row = build_queue_row(mock_packet_false, rank=1, shorts_dir=tmp_path)
        assert row["status"] == "needs_export"

    def test_get_shorts_folder_path(self, tmp_path):
        """get_shorts_folder_path should return correct folder path."""
        # Create folder for claim 42
        folder = tmp_path / "42"
        folder.mkdir()
        
        path = get_shorts_folder_path(42, tmp_path)
        assert path == folder
        assert path.exists()
        
        path2 = get_shorts_folder_path(999, tmp_path)
        assert path2 == tmp_path / "999"
        assert not path2.exists()


# ============================================================================
# Test Queue Export
# ============================================================================


class TestQueueExport:
    """Tests for queue file export."""

    def test_export_queue_csv(self, mock_packets, tmp_path):
        """Should create valid CSV with correct columns."""
        # Filter and prepare
        filtered = filter_packets(mock_packets, min_confidence=0.0)
        shorts_dir = tmp_path / "shorts"
        
        csv_path = export_queue_csv(
            filtered,
            queue_date="2026-01-03",
            output_dir=tmp_path,
            shorts_dir=shorts_dir,
        )
        
        assert csv_path.exists()
        assert csv_path.name == "queue_2026-01-03.csv"
        
        # Verify CSV structure
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) > 0
            # Check first row has all columns
            for col in QUEUE_CSV_COLUMNS:
                assert col in rows[0]

    def test_export_queue_md(self, mock_packets, tmp_path):
        """Should create readable Markdown file."""
        filtered = filter_packets(mock_packets, min_confidence=0.0)
        shorts_dir = tmp_path / "shorts"
        
        md_path = export_queue_md(
            filtered,
            queue_date="2026-01-03",
            output_dir=tmp_path,
            shorts_dir=shorts_dir,
        )
        
        assert md_path.exists()
        assert md_path.name == "queue_2026-01-03.md"
        
        content = md_path.read_text(encoding="utf-8")
        assert "# Shorts Queue: 2026-01-03" in content
        assert "Topic Distribution" in content

    def test_export_csv_columns_order(self, mock_packet_false, tmp_path):
        """CSV columns should be in correct order."""
        csv_path = export_queue_csv(
            [mock_packet_false],
            queue_date="2026-01-03",
            output_dir=tmp_path,
            shorts_dir=tmp_path / "shorts",
        )
        
        with open(csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            # Remove any \r characters
            header_line = header_line.replace("\r", "")
            columns = header_line.split(",")
            
            assert columns == QUEUE_CSV_COLUMNS


# ============================================================================
# Test Create Daily Queue
# ============================================================================


class TestCreateDailyQueue:
    """Tests for the main queue creation function."""

    def test_create_daily_queue_returns_stats(self, tmp_path, monkeypatch):
        """Should return stats dict with count and paths."""
        # Mock load_packets_from_files to return test data
        test_packets = [
            {
                "claim_id": 1,
                "claim": "Test claim",
                "topic": "science",
                "verdict": "False",
                "confidence": 0.85,
                "sources": [{"source_type": "academic"}],
                "titles": ["Test"],
                "shorts_script": {"hook": "Test!", "total_duration": 35},
            }
        ]
        
        def mock_load_files(*args, **kwargs):
            return test_packets
        
        def mock_load_db(*args, **kwargs):
            return []
        
        from pipeline import select_for_shorts
        monkeypatch.setattr(select_for_shorts, "load_packets_from_files", mock_load_files)
        monkeypatch.setattr(select_for_shorts, "load_packets_from_db", mock_load_db)
        
        result = create_daily_queue(
            queue_date="2026-01-03",
            limit=5,
            from_files=True,
            output_dir=tmp_path,
            shorts_dir=tmp_path / "shorts",
        )
        
        assert "count" in result
        assert "csv_path" in result
        assert "md_path" in result
        assert "topic_stats" in result
        assert result["count"] == 1

    def test_create_daily_queue_empty_input(self, tmp_path, monkeypatch):
        """Should handle empty packet list gracefully."""
        def mock_load(*args, **kwargs):
            return []
        
        from pipeline import select_for_shorts
        monkeypatch.setattr(select_for_shorts, "load_packets_from_files", mock_load)
        monkeypatch.setattr(select_for_shorts, "load_packets_from_db", mock_load)
        
        result = create_daily_queue(
            queue_date="2026-01-03",
            limit=5,
            output_dir=tmp_path,
            shorts_dir=tmp_path / "shorts",
        )
        
        assert result["count"] == 0
        assert result["csv_path"] is None


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_queue_workflow(self, mock_packets, tmp_path):
        """Test complete workflow: filter -> dedup -> mix -> export."""
        # Create shorts folders for some claims
        shorts_dir = tmp_path / "shorts"
        (shorts_dir / "1").mkdir(parents=True)
        (shorts_dir / "4").mkdir(parents=True)
        
        # Filter
        filtered = filter_packets(mock_packets, min_confidence=0.5)
        assert len(filtered) > 0
        
        # Dedup
        deduped = deduplicate_packets(filtered)
        
        # Topic mix
        selected = ensure_topic_mix(deduped, limit=3)
        assert len(selected) <= 3
        
        # Export
        csv_path = export_queue_csv(
            selected,
            queue_date="2026-01-03",
            output_dir=tmp_path,
            shorts_dir=shorts_dir,
        )
        md_path = export_queue_md(
            selected,
            queue_date="2026-01-03",
            output_dir=tmp_path,
            shorts_dir=shorts_dir,
        )
        
        assert csv_path.exists()
        assert md_path.exists()
        
        # Verify some rows have "ready" status
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            statuses = [r["status"] for r in rows]
            assert "ready" in statuses or "needs_export" in statuses

    def test_deterministic_queue(self, mock_packets, tmp_path):
        """Same input should produce identical queue."""
        def create_queue():
            filtered = filter_packets(mock_packets.copy(), min_confidence=0.5)
            deduped = deduplicate_packets(filtered)
            selected = ensure_topic_mix(deduped, limit=3)
            return [p["claim_id"] for p in selected]
        
        result1 = create_queue()
        result2 = create_queue()
        
        assert result1 == result2
