"""
Tests for Shorts Export Pack Module

Tests the generation of YouTube Shorts production folders including
voiceover scripts, shotlists, captions, metadata, sources, and assets.
"""

import csv
import io
import json
import re
from pathlib import Path

import pytest

from pipeline.export_shorts_pack import (
    export_shorts_pack,
    generate_assets_needed_md,
    generate_captions_srt,
    generate_metadata_json,
    generate_shotlist_csv,
    generate_sources_md,
    generate_voiceover,
)
from core.constants import (
    SHOTLIST_CSV_COLUMNS,
    VoiceoverLimits,
    OnScreenTextLimits,
    pick_hashtags,
    needs_disclaimer,
    get_disclaimer,
    truncate_onscreen_text,
)


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def mock_packet_basic():
    """Basic mock packet for testing."""
    return {
        "claim_id": 999,
        "claim": "Test claim about science",
        "topic": "science",
        "language": "en",
        "verdict": "False",
        "confidence": 0.8,
        "one_line_verdict": "This claim is false based on scientific evidence.",
        "shorts_script": {
            "hook": "Did you know this common belief is completely wrong?",
            "segments": [
                {
                    "time_start": 0,
                    "time_end": 6,
                    "narration": "Did you know this common belief is completely wrong?",
                    "visual_suggestion": "Hook visual with text overlay",
                    "on_screen_text": "MYTH OR FACT?",
                },
                {
                    "time_start": 6,
                    "time_end": 14,
                    "narration": "Based on multiple scientific studies, this claim is false.",
                    "visual_suggestion": "Verdict reveal animation",
                    "on_screen_text": "FALSE",
                },
                {
                    "time_start": 14,
                    "time_end": 26,
                    "narration": "Here's what the evidence actually shows about this topic.",
                    "visual_suggestion": "Evidence data visualization",
                    "on_screen_text": "THE TRUTH",
                },
                {
                    "time_start": 26,
                    "time_end": 35,
                    "narration": "So remember, always check the facts before believing everything you hear.",
                    "visual_suggestion": "Summary graphic",
                    "on_screen_text": "REMEMBER",
                },
            ],
            "cta": "Follow for more fact-checks!",
            "total_duration": 35,
        },
        "titles": [
            "DEBUNKED: This Common Belief is Wrong",
            "Is This Really True? Scientists Say NO",
            "The Truth About This Myth",
            "Why Everyone Believes This (They're Wrong)",
        ],
        "description": "We fact-check a common misconception.\n\nSOURCES:\n- Study 1\n- Study 2",
        "sources": [
            {
                "evidence_id": 1,
                "title": "Scientific Study on the Topic",
                "url": "https://example.com/study1",
                "source_type": "academic",
                "credibility_score": 90,
            },
            {
                "evidence_id": 2,
                "title": "Wikipedia Article",
                "url": "https://en.wikipedia.org/wiki/Topic",
                "source_type": "wikipedia",
                "credibility_score": 70,
            },
            {
                "evidence_id": 3,
                "title": "Fact-Check Report",
                "url": "https://example.com/factcheck",
                "source_type": "factcheck",
                "credibility_score": 80,
            },
        ],
        "citation_map": {
            "one_line_verdict": [1, 2],
            "why_believed": [2],
            "what_wrong": [1, 2, 3],
            "truth": [1, 3],
        },
        "visuals": [
            "Opening hook animation",
            "Evidence presentation graphics",
            "Verdict reveal animation",
            "Summary infographic",
        ],
        "created_at": "2026-01-03T12:00:00",
    }


@pytest.fixture
def mock_packet_health():
    """Mock packet with health topic (requires disclaimer)."""
    packet = {
        "claim_id": 888,
        "claim": "Drinking lemon water cures cancer",
        "topic": "health",
        "language": "en",
        "verdict": "False",
        "confidence": 0.95,
        "one_line_verdict": "This health claim is false and potentially dangerous.",
        "shorts_script": {
            "hook": "Can lemon water really cure cancer? Let's find out.",
            "segments": [
                {
                    "time_start": 0,
                    "time_end": 5,
                    "narration": "Can lemon water really cure cancer? Let's find out.",
                    "visual_suggestion": "Hook with question",
                    "on_screen_text": "LEMON WATER CURE?",
                },
                {
                    "time_start": 5,
                    "time_end": 12,
                    "narration": "The verdict is clear: FALSE.",
                    "visual_suggestion": "Verdict reveal",
                    "on_screen_text": "FALSE",
                },
                {
                    "time_start": 12,
                    "time_end": 25,
                    "narration": "No food or drink can cure cancer by itself.",
                    "visual_suggestion": "Medical evidence",
                    "on_screen_text": "NO CURE",
                },
                {
                    "time_start": 25,
                    "time_end": 33,
                    "narration": "Always consult a doctor for medical advice.",
                    "visual_suggestion": "Summary",
                    "on_screen_text": "SEE A DOCTOR",
                },
            ],
            "cta": "Follow for more health fact-checks!",
            "total_duration": 33,
        },
        "titles": ["DEBUNKED: Lemon Water Cancer Myth"],
        "description": "Fact-checking a dangerous health myth.",
        "sources": [
            {
                "evidence_id": 10,
                "title": "Cancer Research Institute",
                "url": "https://example.com/cancer",
                "source_type": "official",
                "credibility_score": 95,
            },
        ],
        "citation_map": {"truth": [10]},
        "visuals": ["Medical charts", "Doctor interview"],
        "created_at": "2026-01-03T12:00:00",
    }
    return packet


@pytest.fixture
def mock_packet_chinese():
    """Mock packet with Chinese language."""
    return {
        "claim_id": 777,
        "claim": "吃蘿蔔能治感冒",
        "topic": "health",
        "language": "zh",
        "verdict": "Misleading",
        "confidence": 0.7,
        "shorts_script": {
            "hook": "吃蘿蔔真的能治感冒嗎？",
            "segments": [
                {
                    "time_start": 0,
                    "time_end": 5,
                    "narration": "吃蘿蔔真的能治感冒嗎？讓我們來看看真相。",
                    "visual_suggestion": "開場動畫",
                    "on_screen_text": "蘿蔔治感冒？",
                },
                {
                    "time_start": 5,
                    "time_end": 12,
                    "narration": "結論：這個說法是誤導性的。",
                    "visual_suggestion": "結論揭曉",
                    "on_screen_text": "誤導",
                },
                {
                    "time_start": 12,
                    "time_end": 25,
                    "narration": "蘿蔔確實含有維生素C，但不能直接治療感冒。",
                    "visual_suggestion": "營養成分圖",
                    "on_screen_text": "維生素C",
                },
            ],
            "cta": "追蹤更多健康知識！",
            "total_duration": 25,
        },
        "titles": ["蘿蔔治感冒？真相揭曉"],
        "description": "事實查核：蘿蔔與感冒",
        "sources": [
            {
                "evidence_id": 20,
                "title": "醫學期刊",
                "url": "https://example.com/journal",
                "source_type": "academic",
                "credibility_score": 90,
            },
        ],
        "citation_map": {"truth": [20]},
        "visuals": ["營養圖表", "醫學解說"],
        "created_at": "2026-01-03T12:00:00",
    }


# ============================================================================
# Constants Helper Tests
# ============================================================================


class TestConstantsHelpers:
    """Test helper functions in constants.py."""
    
    def test_pick_hashtags_returns_list(self):
        """pick_hashtags should return a list of strings."""
        hashtags = pick_hashtags("science")
        assert isinstance(hashtags, list)
        assert len(hashtags) > 0
        assert all(isinstance(h, str) for h in hashtags)
    
    def test_pick_hashtags_count(self):
        """pick_hashtags should return approximately count items."""
        hashtags = pick_hashtags("health", count=10)
        assert len(hashtags) <= 10
        assert len(hashtags) >= 5  # At least some hashtags
    
    def test_pick_hashtags_includes_topic_tags(self):
        """pick_hashtags should include topic-specific tags."""
        hashtags = pick_hashtags("science")
        # Should have at least one science-related tag
        assert any("science" in h.lower() for h in hashtags)
    
    def test_pick_hashtags_includes_generic(self):
        """pick_hashtags should include generic tags."""
        hashtags = pick_hashtags("unknown")
        # Should include generic tags like #shorts
        assert any("shorts" in h.lower() for h in hashtags)
    
    def test_pick_hashtags_with_extra(self):
        """pick_hashtags should include extra tags."""
        extra = ["#custom1", "#custom2"]
        hashtags = pick_hashtags("science", extra=extra, count=12)
        assert "#custom1" in hashtags or "#custom2" in hashtags
    
    def test_needs_disclaimer_health(self):
        """Health topic should need disclaimer."""
        assert needs_disclaimer("health") is True
        assert needs_disclaimer("HEALTH") is True
        assert needs_disclaimer("Health") is True
    
    def test_needs_disclaimer_other(self):
        """Non-sensitive topics should not need disclaimer."""
        assert needs_disclaimer("science") is False
        assert needs_disclaimer("history") is False
        assert needs_disclaimer("unknown") is False
    
    def test_get_disclaimer_health(self):
        """get_disclaimer should return disclaimer for health."""
        disclaimer = get_disclaimer("health", "en")
        assert disclaimer is not None
        assert "medical" in disclaimer.lower() or "advice" in disclaimer.lower()
    
    def test_get_disclaimer_none(self):
        """get_disclaimer should return None for non-sensitive topics."""
        disclaimer = get_disclaimer("science", "en")
        assert disclaimer is None
    
    def test_truncate_onscreen_text_english(self):
        """truncate_onscreen_text should limit English words."""
        long_text = "This is a very long on-screen text that needs truncation"
        truncated = truncate_onscreen_text(long_text, "en")
        words = truncated.replace("...", "").split()
        assert len(words) <= OnScreenTextLimits.EN_MAX_WORDS + 1
    
    def test_truncate_onscreen_text_chinese(self):
        """truncate_onscreen_text should limit Chinese characters."""
        long_text = "這是一個非常長的螢幕文字需要被截斷才能顯示在手機上"
        truncated = truncate_onscreen_text(long_text, "zh")
        # Should be truncated
        assert len(truncated) <= OnScreenTextLimits.ZH_MAX_CHARS + 1
    
    def test_truncate_onscreen_text_short(self):
        """Short text should not be truncated."""
        short_text = "SHORT"
        truncated = truncate_onscreen_text(short_text, "en")
        assert truncated == short_text


# ============================================================================
# Voiceover Generation Tests
# ============================================================================


class TestGenerateVoiceover:
    """Test voiceover generation."""
    
    def test_voiceover_not_empty(self, mock_packet_basic):
        """Voiceover should not be empty."""
        voiceover = generate_voiceover(mock_packet_basic)
        assert len(voiceover) > 0
    
    def test_voiceover_contains_hook(self, mock_packet_basic):
        """Voiceover should start with hook."""
        voiceover = generate_voiceover(mock_packet_basic)
        hook = mock_packet_basic["shorts_script"]["hook"]
        assert hook in voiceover
    
    def test_voiceover_contains_cta(self, mock_packet_basic):
        """Voiceover should contain CTA."""
        voiceover = generate_voiceover(mock_packet_basic)
        # Should contain engagement CTA
        assert "comment" in voiceover.lower() or "myth" in voiceover.lower()
    
    def test_voiceover_health_has_disclaimer(self, mock_packet_health):
        """Health topic voiceover should include disclaimer."""
        voiceover = generate_voiceover(mock_packet_health)
        # Should have some form of medical disclaimer
        assert "medical" in voiceover.lower() or "advice" in voiceover.lower() or "professional" in voiceover.lower()
    
    def test_voiceover_science_no_disclaimer(self, mock_packet_basic):
        """Science topic should not have health disclaimer."""
        voiceover = generate_voiceover(mock_packet_basic)
        # Science topic should not have medical disclaimer
        assert "not medical advice" not in voiceover.lower()
    
    def test_voiceover_length_reasonable(self, mock_packet_basic):
        """Voiceover length should be reasonable for Shorts."""
        voiceover = generate_voiceover(mock_packet_basic)
        # Should be under ~1500 characters (reasonable for 60s video)
        assert len(voiceover) < 1500
    
    def test_voiceover_chinese(self, mock_packet_chinese):
        """Chinese voiceover should work."""
        voiceover = generate_voiceover(mock_packet_chinese)
        assert len(voiceover) > 0
        # Should contain Chinese CTA
        assert "迷思" in voiceover or "留言" in voiceover


# ============================================================================
# Shotlist CSV Tests
# ============================================================================


class TestGenerateShotlistCsv:
    """Test shotlist CSV generation."""
    
    def test_shotlist_has_header(self, mock_packet_basic):
        """Shotlist should have proper CSV header."""
        csv_content = generate_shotlist_csv(mock_packet_basic)
        reader = csv.DictReader(io.StringIO(csv_content))
        fieldnames = reader.fieldnames
        
        # Check all required columns exist
        for col in SHOTLIST_CSV_COLUMNS:
            assert col in fieldnames, f"Missing column: {col}"
    
    def test_shotlist_columns_order(self, mock_packet_basic):
        """Shotlist columns should be in correct order."""
        csv_content = generate_shotlist_csv(mock_packet_basic)
        first_line = csv_content.split("\n")[0].strip()
        columns = [col.strip() for col in first_line.split(",")]
        
        assert columns == SHOTLIST_CSV_COLUMNS
    
    def test_shotlist_has_rows(self, mock_packet_basic):
        """Shotlist should have data rows."""
        csv_content = generate_shotlist_csv(mock_packet_basic)
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        
        # Should have at least 3 rows (for a basic shorts)
        assert len(rows) >= 3
    
    def test_shotlist_time_increasing(self, mock_packet_basic):
        """Shotlist time should be increasing."""
        csv_content = generate_shotlist_csv(mock_packet_basic)
        reader = csv.DictReader(io.StringIO(csv_content))
        
        prev_end = 0
        for row in reader:
            time_start = float(row["time_start"])
            time_end = float(row["time_end"])
            
            assert time_start >= prev_end - 1, "Times should not overlap significantly"
            assert time_end > time_start, "End time should be after start time"
            prev_end = time_end
    
    def test_shotlist_onscreen_text_short(self, mock_packet_basic):
        """On-screen text should be short."""
        csv_content = generate_shotlist_csv(mock_packet_basic)
        reader = csv.DictReader(io.StringIO(csv_content))
        
        for row in reader:
            on_screen = row["on_screen_text"]
            words = on_screen.split()
            # Should be max 6 words or truncated
            assert len(words) <= OnScreenTextLimits.EN_MAX_WORDS + 2  # Allow small buffer


# ============================================================================
# SRT Caption Tests
# ============================================================================


class TestGenerateCaptionsSrt:
    """Test SRT caption generation."""
    
    def test_srt_not_empty(self, mock_packet_basic):
        """SRT should not be empty."""
        srt = generate_captions_srt(mock_packet_basic)
        assert len(srt) > 0
    
    def test_srt_has_sequences(self, mock_packet_basic):
        """SRT should have numbered sequences."""
        srt = generate_captions_srt(mock_packet_basic)
        
        # Find sequence numbers
        numbers = re.findall(r"^(\d+)$", srt, re.MULTILINE)
        assert len(numbers) >= 3, "Should have at least 3 subtitle sequences"
    
    def test_srt_has_timestamps(self, mock_packet_basic):
        """SRT should have valid timestamps."""
        srt = generate_captions_srt(mock_packet_basic)
        
        # Check for timestamp format: 00:00:00,000 --> 00:00:05,000
        timestamp_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        timestamps = re.findall(timestamp_pattern, srt)
        
        assert len(timestamps) >= 3, "Should have at least 3 timestamps"
    
    def test_srt_timestamps_increasing(self, mock_packet_basic):
        """SRT timestamps should be increasing."""
        srt = generate_captions_srt(mock_packet_basic)
        
        # Extract all start times
        timestamp_pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) -->"
        matches = re.findall(timestamp_pattern, srt)
        
        prev_time = 0
        for h, m, s, ms in matches:
            current_time = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            assert current_time >= prev_time, "Timestamps should be increasing"
            prev_time = current_time
    
    def test_srt_format_valid(self, mock_packet_basic):
        """SRT format should be valid."""
        srt = generate_captions_srt(mock_packet_basic)
        blocks = srt.strip().split("\n\n")
        
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            # Each block should have: number, timestamp, text (1+ lines)
            assert len(lines) >= 3, f"Invalid SRT block: {block}"
            assert lines[0].strip().isdigit(), f"First line should be number: {lines[0]}"
            assert "-->" in lines[1], f"Second line should be timestamp: {lines[1]}"


# ============================================================================
# Metadata JSON Tests
# ============================================================================


class TestGenerateMetadataJson:
    """Test metadata JSON generation."""
    
    def test_metadata_has_title(self, mock_packet_basic):
        """Metadata should have title."""
        metadata = generate_metadata_json(mock_packet_basic)
        assert "title" in metadata
        assert len(metadata["title"]) > 0
    
    def test_metadata_has_alternatives(self, mock_packet_basic):
        """Metadata should have alternative titles."""
        metadata = generate_metadata_json(mock_packet_basic)
        assert "alternatives" in metadata
        assert isinstance(metadata["alternatives"], list)
    
    def test_metadata_hashtags_count(self, mock_packet_basic):
        """Metadata should have 8-12 hashtags."""
        metadata = generate_metadata_json(mock_packet_basic)
        assert "hashtags" in metadata
        hashtags = metadata["hashtags"]
        assert 8 <= len(hashtags) <= 12, f"Should have 8-12 hashtags, got {len(hashtags)}"
    
    def test_metadata_hashtags_not_empty(self, mock_packet_basic):
        """Hashtags should not be empty strings."""
        metadata = generate_metadata_json(mock_packet_basic)
        for tag in metadata["hashtags"]:
            assert len(tag) > 0
            assert tag.startswith("#")
    
    def test_metadata_has_required_fields(self, mock_packet_basic):
        """Metadata should have all required fields."""
        metadata = generate_metadata_json(mock_packet_basic)
        
        required_fields = ["title", "description", "hashtags", "claim_id", "topic", "verdict", "confidence"]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_metadata_is_json_serializable(self, mock_packet_basic):
        """Metadata should be JSON serializable."""
        metadata = generate_metadata_json(mock_packet_basic)
        # This should not raise
        json_str = json.dumps(metadata)
        assert len(json_str) > 0


# ============================================================================
# Sources Markdown Tests
# ============================================================================


class TestGenerateSourcesMd:
    """Test sources markdown generation."""
    
    def test_sources_not_empty(self, mock_packet_basic):
        """Sources markdown should not be empty."""
        sources = generate_sources_md(mock_packet_basic)
        assert len(sources) > 0
    
    def test_sources_has_header(self, mock_packet_basic):
        """Sources should have header."""
        sources = generate_sources_md(mock_packet_basic)
        assert "# Sources" in sources
    
    def test_sources_lists_urls(self, mock_packet_basic):
        """Sources should list URLs."""
        sources = generate_sources_md(mock_packet_basic)
        assert "https://" in sources
    
    def test_sources_has_citation_map(self, mock_packet_basic):
        """Sources should include citation map."""
        sources = generate_sources_md(mock_packet_basic)
        assert "Citation Map" in sources or "citation" in sources.lower()


# ============================================================================
# Assets Needed Markdown Tests
# ============================================================================


class TestGenerateAssetsNeededMd:
    """Test assets needed markdown generation."""
    
    def test_assets_not_empty(self, mock_packet_basic):
        """Assets markdown should not be empty."""
        assets = generate_assets_needed_md(mock_packet_basic)
        assert len(assets) > 0
    
    def test_assets_has_header(self, mock_packet_basic):
        """Assets should have header."""
        assets = generate_assets_needed_md(mock_packet_basic)
        assert "# Assets" in assets
    
    def test_assets_lists_visuals(self, mock_packet_basic):
        """Assets should list visual requirements."""
        assets = generate_assets_needed_md(mock_packet_basic)
        # Should mention at least one category
        categories = ["B-roll", "Chart", "Text", "Animation", "Original"]
        assert any(cat in assets for cat in categories)


# ============================================================================
# Full Export Tests
# ============================================================================


class TestExportShortsPack:
    """Test full export functionality."""
    
    def test_export_creates_folder(self, mock_packet_basic, tmp_path):
        """Export should create claim folder."""
        folder = export_shorts_pack(mock_packet_basic, tmp_path)
        
        assert folder is not None
        assert folder.exists()
        assert folder.is_dir()
    
    def test_export_creates_all_files(self, mock_packet_basic, tmp_path):
        """Export should create all 6 files."""
        folder = export_shorts_pack(mock_packet_basic, tmp_path)
        
        expected_files = [
            "voiceover.txt",
            "shotlist.csv",
            "captions.srt",
            "metadata.json",
            "sources.md",
            "assets_needed.md",
        ]
        
        for filename in expected_files:
            filepath = folder / filename
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"
    
    def test_export_no_overwrite_by_default(self, mock_packet_basic, tmp_path):
        """Export should not overwrite existing files by default."""
        # First export
        folder = export_shorts_pack(mock_packet_basic, tmp_path)
        assert folder is not None
        
        # Modify a file to check if it gets overwritten
        voiceover_path = folder / "voiceover.txt"
        original_content = voiceover_path.read_text(encoding="utf-8")
        voiceover_path.write_text("MODIFIED", encoding="utf-8")
        
        # Second export (should skip)
        folder2 = export_shorts_pack(mock_packet_basic, tmp_path, overwrite=False)
        
        # Should skip (return None) since files exist
        assert folder2 is None
        
        # File should still be modified (not overwritten)
        assert voiceover_path.read_text(encoding="utf-8") == "MODIFIED"
    
    def test_export_overwrite_when_flag_set(self, mock_packet_basic, tmp_path):
        """Export should overwrite when flag is set."""
        # First export
        folder = export_shorts_pack(mock_packet_basic, tmp_path)
        voiceover_path = folder / "voiceover.txt"
        original_content = voiceover_path.read_text(encoding="utf-8")
        
        # Modify file
        voiceover_path.write_text("MODIFIED", encoding="utf-8")
        
        # Export with overwrite
        folder2 = export_shorts_pack(mock_packet_basic, tmp_path, overwrite=True)
        
        # File should be restored to original
        new_content = voiceover_path.read_text(encoding="utf-8")
        assert new_content == original_content
    
    def test_export_deterministic(self, mock_packet_basic, tmp_path):
        """Same input should produce same output."""
        folder1 = tmp_path / "run1"
        folder2 = tmp_path / "run2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Export twice
        export_shorts_pack(mock_packet_basic, folder1)
        export_shorts_pack(mock_packet_basic, folder2)
        
        # Compare outputs
        claim_id = mock_packet_basic["claim_id"]
        files_to_check = ["voiceover.txt", "shotlist.csv", "metadata.json"]
        
        for filename in files_to_check:
            content1 = (folder1 / str(claim_id) / filename).read_text(encoding="utf-8")
            content2 = (folder2 / str(claim_id) / filename).read_text(encoding="utf-8")
            
            # Metadata might have different timestamps, so we check other files
            if filename != "metadata.json":
                assert content1 == content2, f"Output differs for {filename}"
    
    def test_export_handles_missing_claim_id(self, tmp_path):
        """Export should handle packet without claim_id."""
        packet = {"topic": "science", "shorts_script": {"segments": []}}
        result = export_shorts_pack(packet, tmp_path)
        assert result is None
    
    def test_export_health_topic(self, mock_packet_health, tmp_path):
        """Health topic export should include disclaimer."""
        folder = export_shorts_pack(mock_packet_health, tmp_path)
        
        voiceover_path = folder / "voiceover.txt"
        voiceover = voiceover_path.read_text(encoding="utf-8")
        
        # Should contain disclaimer
        assert "medical" in voiceover.lower() or "advice" in voiceover.lower()
    
    def test_export_chinese_packet(self, mock_packet_chinese, tmp_path):
        """Chinese packet should export correctly."""
        folder = export_shorts_pack(mock_packet_chinese, tmp_path)
        
        assert folder is not None
        
        # Check voiceover has Chinese content
        voiceover = (folder / "voiceover.txt").read_text(encoding="utf-8")
        assert "蘿蔔" in voiceover or "迷思" in voiceover


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the export workflow."""
    
    def test_full_workflow(self, mock_packet_basic, tmp_path):
        """Test complete export workflow."""
        # Export packet
        folder = export_shorts_pack(mock_packet_basic, tmp_path)
        
        # Verify all files
        assert (folder / "voiceover.txt").exists()
        assert (folder / "shotlist.csv").exists()
        assert (folder / "captions.srt").exists()
        assert (folder / "metadata.json").exists()
        assert (folder / "sources.md").exists()
        assert (folder / "assets_needed.md").exists()
        
        # Verify metadata is valid JSON
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["claim_id"] == 999
        assert metadata["topic"] == "science"
        assert len(metadata["hashtags"]) >= 8
        
        # Verify CSV is valid
        csv_content = (folder / "shotlist.csv").read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) >= 3
        
        # Verify SRT is valid
        srt_content = (folder / "captions.srt").read_text(encoding="utf-8")
        assert "00:00:" in srt_content
        assert "-->" in srt_content
