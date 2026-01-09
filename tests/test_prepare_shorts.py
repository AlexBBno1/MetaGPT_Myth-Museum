"""
Tests for pipeline/prepare_shorts.py

Shorts preparation and status transition tests.
"""

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.constants import (
    SHORTS_REQUIRED_FILES,
    ShortsStatus,
    VOICEOVER_MIN_ZH_CHARS,
    VOICEOVER_MIN_EN_WORDS,
    VOICEOVER_MIN_DURATION_SECONDS,
    determine_shorts_status,
)
from pipeline.prepare_shorts import (
    check_audio_duration,
    check_voiceover_text_length,
    format_srt_timestamp,
    generate_prepare_report,
    load_packet_for_claim,
    load_queue_csv,
    parse_srt_timestamp,
    prepare_single,
    repair_srt,
    update_queue_csv,
    validate_srt,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_packet() -> dict[str, Any]:
    """Create a mock packet for testing."""
    return {
        "claim_id": 42,
        "claim": "Test claim about something",
        "topic": "science",
        "verdict": "False",
        "confidence": 0.85,
        "sources": [{"title": "Test Source", "url": "https://example.com"}],
        "titles": ["Test Title"],
        "description": "Test description",
        "shorts_script": {
            "hook": "Did you know?",
            "segments": [
                {"voice_line": "Line 1", "visual": "Visual 1"},
                {"voice_line": "Line 2", "visual": "Visual 2"},
            ],
            "cta": "Comment below!",
            "total_duration": 35,
        },
        "visuals": ["Visual 1", "Visual 2"],
    }


@pytest.fixture
def mock_queue_csv(tmp_path) -> Path:
    """Create a mock queue CSV file."""
    queue_dir = tmp_path / "shorts_queue"
    queue_dir.mkdir(parents=True)
    
    csv_path = queue_dir / "queue_2026-01-03.csv"
    
    rows = [
        {
            "rank": "1",
            "claim_id": "42",
            "topic": "science",
            "verdict": "False",
            "confidence": "0.85",
            "title": "Test Title",
            "hook": "Did you know?",
            "estimated_seconds": "35",
            "folder_path": "outputs/shorts/42",
            "status": "needs_export",
        },
        {
            "rank": "2",
            "claim_id": "43",
            "topic": "health",
            "verdict": "Misleading",
            "confidence": "0.75",
            "title": "Health Myth",
            "hook": "You won't believe...",
            "estimated_seconds": "40",
            "folder_path": "outputs/shorts/43",
            "status": "needs_tts",
        },
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return csv_path


@pytest.fixture
def shorts_folder_needs_export(tmp_path) -> Path:
    """Create an empty shorts folder (needs_export status)."""
    folder = tmp_path / "shorts" / "42"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def shorts_folder_needs_tts(tmp_path) -> Path:
    """Create a shorts folder with all files except voiceover.mp3."""
    folder = tmp_path / "shorts" / "43"
    folder.mkdir(parents=True)
    
    for filename in SHORTS_REQUIRED_FILES:
        (folder / filename).write_text(f"Mock content for {filename}", encoding="utf-8")
    
    return folder


@pytest.fixture
def shorts_folder_ready(tmp_path) -> Path:
    """Create a ready shorts folder with all files including mp3."""
    folder = tmp_path / "shorts" / "44"
    folder.mkdir(parents=True)
    
    for filename in SHORTS_REQUIRED_FILES:
        (folder / filename).write_text(f"Mock content for {filename}", encoding="utf-8")
    
    (folder / "voiceover.mp3").write_bytes(b"\x00" * 1000)
    
    return folder


@pytest.fixture
def shorts_folder_rendered(tmp_path) -> Path:
    """Create a rendered shorts folder with final.mp4."""
    folder = tmp_path / "shorts" / "45"
    folder.mkdir(parents=True)
    
    for filename in SHORTS_REQUIRED_FILES:
        (folder / filename).write_text(f"Mock content for {filename}", encoding="utf-8")
    
    (folder / "voiceover.mp3").write_bytes(b"\x00" * 1000)
    (folder / "final.mp4").write_bytes(b"\x00" * 5000)
    
    return folder


@pytest.fixture
def valid_srt_content() -> str:
    """Create valid SRT content."""
    return """1
00:00:00,000 --> 00:00:05,000
First subtitle line.

2
00:00:05,000 --> 00:00:10,000
Second subtitle line.

3
00:00:10,000 --> 00:00:15,000
Third subtitle line.
"""


@pytest.fixture
def invalid_srt_content() -> str:
    """Create invalid SRT content with timing issues."""
    return """1
00:00:05,000 --> 00:00:03,000
End before start.

2
00:00:02,000 --> 00:00:08,000
Overlapping previous.

3
00:00:10,000 --> 00:00:15,000
Valid segment.
"""


# ============================================================================
# Test Status Determination
# ============================================================================


class TestStatusDetermination:
    """Tests for status state machine."""

    def test_needs_export_no_folder(self, tmp_path):
        """Missing folder should return NEEDS_EXPORT."""
        folder = tmp_path / "nonexistent"
        status = determine_shorts_status(folder)
        assert status == ShortsStatus.NEEDS_EXPORT

    def test_needs_export_missing_files(self, shorts_folder_needs_export):
        """Folder with missing files should return NEEDS_EXPORT."""
        # Create only some files
        (shorts_folder_needs_export / "voiceover.txt").write_text("test")
        
        status = determine_shorts_status(shorts_folder_needs_export)
        assert status == ShortsStatus.NEEDS_EXPORT

    def test_needs_tts_no_mp3(self, shorts_folder_needs_tts):
        """Folder with 6 files but no mp3 should return NEEDS_TTS."""
        status = determine_shorts_status(shorts_folder_needs_tts)
        assert status == ShortsStatus.NEEDS_TTS

    def test_ready_all_present(self, shorts_folder_ready):
        """Folder with all files should return READY."""
        status = determine_shorts_status(shorts_folder_ready)
        assert status == ShortsStatus.READY

    def test_rendered_has_mp4(self, shorts_folder_rendered):
        """Folder with final.mp4 should return RENDERED."""
        status = determine_shorts_status(shorts_folder_rendered)
        assert status == ShortsStatus.RENDERED


# ============================================================================
# Test Queue Loading
# ============================================================================


class TestQueueLoading:
    """Tests for queue CSV loading."""

    def test_load_queue_csv(self, mock_queue_csv, tmp_path):
        """Should load queue rows correctly."""
        queue_dir = tmp_path / "shorts_queue"
        rows = load_queue_csv("2026-01-03", queue_dir)
        
        assert len(rows) == 2
        assert rows[0]["claim_id"] == "42"
        assert rows[1]["claim_id"] == "43"

    def test_load_queue_csv_missing(self, tmp_path):
        """Should return empty list for missing queue."""
        rows = load_queue_csv("2099-01-01", tmp_path)
        assert rows == []


# ============================================================================
# Test SRT Validation
# ============================================================================


class TestSrtValidation:
    """Tests for SRT validation and repair."""

    def test_validate_srt_valid(self, tmp_path, valid_srt_content):
        """Should validate correct SRT file."""
        srt_path = tmp_path / "valid.srt"
        srt_path.write_text(valid_srt_content, encoding="utf-8")
        
        is_valid, issues = validate_srt(srt_path)
        
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_srt_invalid(self, tmp_path, invalid_srt_content):
        """Should detect issues in invalid SRT file."""
        srt_path = tmp_path / "invalid.srt"
        srt_path.write_text(invalid_srt_content, encoding="utf-8")
        
        is_valid, issues = validate_srt(srt_path)
        
        assert is_valid is False
        assert len(issues) > 0

    def test_validate_srt_missing(self, tmp_path):
        """Should handle missing file gracefully."""
        srt_path = tmp_path / "missing.srt"
        
        is_valid, issues = validate_srt(srt_path)
        
        assert is_valid is False
        assert "exist" in issues[0].lower()

    def test_repair_srt(self, tmp_path, invalid_srt_content):
        """Should repair invalid SRT timestamps."""
        srt_path = tmp_path / "repair.srt"
        srt_path.write_text(invalid_srt_content, encoding="utf-8")
        
        repaired = repair_srt(srt_path)
        
        assert repaired is True
        
        # Validate after repair
        is_valid, issues = validate_srt(srt_path)
        assert is_valid is True

    def test_parse_srt_timestamp(self):
        """Should parse SRT timestamps correctly."""
        assert parse_srt_timestamp("00:00:00,000") == 0.0
        assert parse_srt_timestamp("00:00:05,500") == 5.5
        assert parse_srt_timestamp("00:01:30,000") == 90.0
        assert parse_srt_timestamp("01:00:00,000") == 3600.0

    def test_format_srt_timestamp(self):
        """Should format timestamps correctly."""
        assert format_srt_timestamp(0.0) == "00:00:00,000"
        assert format_srt_timestamp(5.5) == "00:00:05,500"
        assert format_srt_timestamp(90.0) == "00:01:30,000"


# ============================================================================
# Test Prepare Single
# ============================================================================


class TestPrepareSingle:
    """Tests for single item preparation."""

    @pytest.mark.asyncio
    async def test_prepare_already_ready(self, shorts_folder_ready, tmp_path):
        """Should return existing status for ready folder."""
        result = await prepare_single(
            claim_id=44,
            packet=None,
            shorts_dir=tmp_path / "shorts",
            packets_dir=tmp_path / "packets",
        )
        
        assert result["initial_status"] == "ready"
        assert result["final_status"] == "ready"

    @pytest.mark.asyncio
    async def test_prepare_already_rendered(self, shorts_folder_rendered, tmp_path):
        """Should return existing status for rendered folder."""
        result = await prepare_single(
            claim_id=45,
            packet=None,
            shorts_dir=tmp_path / "shorts",
            packets_dir=tmp_path / "packets",
        )
        
        assert result["initial_status"] == "rendered"
        assert result["final_status"] == "rendered"

    @pytest.mark.asyncio
    async def test_prepare_needs_export_no_packet(self, tmp_path):
        """Should error when packet unavailable for export."""
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir(parents=True)
        
        result = await prepare_single(
            claim_id=999,
            packet=None,
            shorts_dir=shorts_dir,
            packets_dir=tmp_path / "packets",
        )
        
        assert result["initial_status"] == "needs_export"
        assert result["final_status"] == "needs_export"
        assert len(result["errors"]) > 0


# ============================================================================
# Test Queue Update
# ============================================================================


class TestQueueUpdate:
    """Tests for queue CSV update."""

    def test_update_queue_csv(self, mock_queue_csv, tmp_path):
        """Should update queue CSV with new statuses."""
        queue_dir = tmp_path / "shorts_queue"
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir(parents=True)
        
        # Create a ready folder for claim 42
        folder_42 = shorts_dir / "42"
        folder_42.mkdir()
        for f in SHORTS_REQUIRED_FILES:
            (folder_42 / f).write_text("test")
        (folder_42 / "voiceover.mp3").write_bytes(b"\x00" * 100)
        
        results = [
            {"claim_id": 42, "final_status": "ready"},
        ]
        
        update_queue_csv("2026-01-03", results, queue_dir, shorts_dir)
        
        # Read updated queue
        rows = load_queue_csv("2026-01-03", queue_dir)
        
        # Find claim 42
        row_42 = next(r for r in rows if r["claim_id"] == "42")
        assert row_42["status"] == "ready"


# ============================================================================
# Test Report Generation
# ============================================================================


class TestReportGeneration:
    """Tests for prepare report generation."""

    def test_generate_prepare_report(self, tmp_path):
        """Should generate markdown report."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        
        results = [
            {
                "claim_id": 42,
                "initial_status": "needs_export",
                "final_status": "ready",
                "actions": ["export_folder", "generate_tts"],
                "errors": [],
            },
            {
                "claim_id": 43,
                "initial_status": "needs_tts",
                "final_status": "needs_tts",
                "actions": [],
                "errors": ["TTS generation failed"],
            },
        ]
        
        report_path = generate_prepare_report("2026-01-03", results, queue_dir)
        
        assert report_path.exists()
        
        content = report_path.read_text(encoding="utf-8")
        assert "Prepare Report" in content
        assert "42" in content
        assert "43" in content
        assert "TTS generation failed" in content


# ============================================================================
# Test Determinism
# ============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_status_determination_deterministic(self, shorts_folder_ready):
        """Status determination should be deterministic."""
        status1 = determine_shorts_status(shorts_folder_ready)
        status2 = determine_shorts_status(shorts_folder_ready)
        status3 = determine_shorts_status(shorts_folder_ready)
        
        assert status1 == status2 == status3

    def test_srt_repair_deterministic(self, tmp_path, invalid_srt_content):
        """SRT repair should produce same result each time."""
        srt1 = tmp_path / "srt1.srt"
        srt2 = tmp_path / "srt2.srt"
        
        srt1.write_text(invalid_srt_content, encoding="utf-8")
        srt2.write_text(invalid_srt_content, encoding="utf-8")
        
        repair_srt(srt1)
        repair_srt(srt2)
        
        content1 = srt1.read_text(encoding="utf-8")
        content2 = srt2.read_text(encoding="utf-8")
        
        assert content1 == content2


# ============================================================================
# Test Voiceover Length Validation (too_short_voiceover gate)
# ============================================================================


class TestVoiceoverLengthValidation:
    """Tests for voiceover text length validation."""

    def test_check_voiceover_text_length_chinese_valid(self, tmp_path):
        """Should pass for Chinese text above minimum."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        # Create Chinese voiceover.txt with enough characters (>120)
        chinese_text = "這是一個測試文本。" * 20  # ~180 characters
        (folder / "voiceover.txt").write_text(chinese_text, encoding="utf-8")
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert is_valid
        assert reason == ""
        assert length >= VOICEOVER_MIN_ZH_CHARS

    def test_check_voiceover_text_length_chinese_too_short(self, tmp_path):
        """Should fail for Chinese text below minimum."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        # Create Chinese voiceover.txt with too few characters (<120)
        short_chinese = "這是短文。" * 5  # ~25 characters
        (folder / "voiceover.txt").write_text(short_chinese, encoding="utf-8")
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert not is_valid
        assert "too short" in reason.lower()
        assert str(VOICEOVER_MIN_ZH_CHARS) in reason

    def test_check_voiceover_text_length_english_valid(self, tmp_path):
        """Should pass for English text above minimum."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        # Create English voiceover.txt with enough words (>80)
        english_text = " ".join(["word"] * 100)
        (folder / "voiceover.txt").write_text(english_text, encoding="utf-8")
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert is_valid
        assert reason == ""
        assert length >= VOICEOVER_MIN_EN_WORDS

    def test_check_voiceover_text_length_english_too_short(self, tmp_path):
        """Should fail for English text below minimum."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        # Create English voiceover.txt with too few words (<80)
        short_english = " ".join(["word"] * 20)
        (folder / "voiceover.txt").write_text(short_english, encoding="utf-8")
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert not is_valid
        assert "too short" in reason.lower()
        assert str(VOICEOVER_MIN_EN_WORDS) in reason

    def test_check_voiceover_text_length_empty(self, tmp_path):
        """Should fail for empty voiceover.txt."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        (folder / "voiceover.txt").write_text("", encoding="utf-8")
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert not is_valid
        assert "empty" in reason.lower()

    def test_check_voiceover_text_length_missing(self, tmp_path):
        """Should fail for missing voiceover.txt."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        is_valid, reason, length = check_voiceover_text_length(folder / "voiceover.txt")
        
        assert not is_valid
        assert "not exist" in reason.lower()


# ============================================================================
# Test Audio Duration Validation (too_short_audio gate)
# ============================================================================


class TestAudioDurationValidation:
    """Tests for audio duration validation."""

    def test_check_audio_duration_valid(self, tmp_path):
        """Should pass for audio above minimum duration."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 1000)
        
        # Mock ffprobe to return valid duration
        import json
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "35.0"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, reason, duration = check_audio_duration(folder / "voiceover.mp3")
        
        assert is_valid
        assert reason == ""
        assert duration >= VOICEOVER_MIN_DURATION_SECONDS

    def test_check_audio_duration_too_short(self, tmp_path):
        """Should fail for audio below minimum duration."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 100)
        
        # Mock ffprobe to return short duration
        import json
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "5.0"},  # Too short!
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, reason, duration = check_audio_duration(folder / "voiceover.mp3")
        
        assert not is_valid
        assert "too short" in reason.lower()
        assert str(VOICEOVER_MIN_DURATION_SECONDS) in reason

    def test_check_audio_duration_missing(self, tmp_path):
        """Should fail for missing audio file."""
        folder = tmp_path / "shorts" / "100"
        folder.mkdir(parents=True)
        
        is_valid, reason, duration = check_audio_duration(folder / "voiceover.mp3")
        
        assert not is_valid
        assert "not exist" in reason.lower()


# ============================================================================
# Test New Status Values
# ============================================================================


class TestNewStatusValues:
    """Tests for new status enum values."""

    def test_status_enum_has_too_short_voiceover(self):
        """ShortsStatus should include TOO_SHORT_VOICEOVER."""
        assert hasattr(ShortsStatus, "TOO_SHORT_VOICEOVER")
        assert ShortsStatus.TOO_SHORT_VOICEOVER.value == "too_short_voiceover"

    def test_status_enum_has_too_short_audio(self):
        """ShortsStatus should include TOO_SHORT_AUDIO."""
        assert hasattr(ShortsStatus, "TOO_SHORT_AUDIO")
        assert ShortsStatus.TOO_SHORT_AUDIO.value == "too_short_audio"

    def test_status_enum_has_tts_failed(self):
        """ShortsStatus should include TTS_FAILED."""
        assert hasattr(ShortsStatus, "TTS_FAILED")
        assert ShortsStatus.TTS_FAILED.value == "tts_failed"

    def test_status_enum_has_render_failed(self):
        """ShortsStatus should include RENDER_FAILED."""
        assert hasattr(ShortsStatus, "RENDER_FAILED")
        assert ShortsStatus.RENDER_FAILED.value == "render_failed"


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_status_transition_flow(self, tmp_path):
        """Test status transitions through the pipeline."""
        shorts_dir = tmp_path / "shorts"
        folder = shorts_dir / "100"
        
        # Stage 1: Empty (needs_export)
        folder.mkdir(parents=True)
        assert determine_shorts_status(folder) == ShortsStatus.NEEDS_EXPORT
        
        # Stage 2: Add 6 files (needs_tts)
        for f in SHORTS_REQUIRED_FILES:
            (folder / f).write_text("test")
        assert determine_shorts_status(folder) == ShortsStatus.NEEDS_TTS
        
        # Stage 3: Add mp3 (ready)
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 100)
        assert determine_shorts_status(folder) == ShortsStatus.READY
        
        # Stage 4: Add mp4 (rendered)
        (folder / "final.mp4").write_bytes(b"\x00" * 1000)
        assert determine_shorts_status(folder) == ShortsStatus.RENDERED
