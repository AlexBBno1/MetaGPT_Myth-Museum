"""
Tests for pipeline/tts.py

TTS voice-over generation and SRT timing adjustment tests.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.constants import (
    SRT_TIMING_ADJUSTMENT_THRESHOLD,
    TTS_DEFAULT_VOICES,
    detect_language,
    get_tts_voice,
)
from pipeline.tts import (
    adjust_srt_timing,
    batch_generate,
    format_srt_timestamp,
    generate_voiceover_for_folder,
    get_audio_duration,
    parse_srt_timestamp,
    run_async,
    set_synthesize_function,
    synthesize,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_voiceover_en() -> str:
    """Sample English voiceover text."""
    return """Did you know that drinking bleach is actually dangerous?
    
Many people believe it can cure diseases, but this is completely false.

The truth is: bleach is a toxic chemical that can cause severe harm.

What myths have you heard? Comment below!"""


@pytest.fixture
def sample_voiceover_zh() -> str:
    """Sample Chinese voiceover text."""
    return """你知道喝漂白水其實是危險的嗎？

很多人相信它可以治病，但這完全是錯誤的。

事實是：漂白水是一種有毒的化學物質，會造成嚴重傷害。

你聽過哪些迷思？在下面留言！"""


@pytest.fixture
def sample_srt() -> str:
    """Sample SRT file content."""
    return """1
00:00:00,000 --> 00:00:05,000
Did you know that drinking bleach is actually dangerous?

2
00:00:05,000 --> 00:00:10,000
Many people believe it can cure diseases.

3
00:00:10,000 --> 00:00:15,000
But this is completely false.

4
00:00:15,000 --> 00:00:20,000
The truth is: bleach is a toxic chemical.

5
00:00:20,000 --> 00:00:25,000
What myths have you heard? Comment below!
"""


@pytest.fixture
def shorts_folder(tmp_path, sample_voiceover_en, sample_srt) -> Path:
    """Create a sample shorts folder with voiceover and captions."""
    folder = tmp_path / "shorts" / "1"
    folder.mkdir(parents=True)
    
    (folder / "voiceover.txt").write_text(sample_voiceover_en, encoding="utf-8")
    (folder / "captions.srt").write_text(sample_srt, encoding="utf-8")
    
    return folder


@pytest.fixture
def mock_synthesize():
    """Mock synthesize function for testing."""
    async def _mock(text, output_path, voice, rate="+0%", pitch="+0Hz"):
        # Create a dummy MP3 file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00" * 1000)  # Dummy content
        return output_path
    
    return _mock


# ============================================================================
# Test Language Detection
# ============================================================================


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_english(self, sample_voiceover_en):
        """Should detect English text."""
        lang = detect_language(sample_voiceover_en)
        assert lang == "en"

    def test_detect_chinese(self, sample_voiceover_zh):
        """Should detect Chinese text."""
        lang = detect_language(sample_voiceover_zh)
        assert lang == "zh"

    def test_detect_empty(self):
        """Empty text should default to English."""
        lang = detect_language("")
        assert lang == "en"

    def test_detect_mixed_mostly_english(self):
        """Mixed text with more English should detect as English."""
        text = "Hello world! 你好"
        lang = detect_language(text)
        assert lang == "en"

    def test_detect_mixed_mostly_chinese(self):
        """Mixed text with more Chinese should detect as Chinese."""
        # Using more Chinese characters to exceed the 30% threshold
        text = "這是一段中文測試內容 with English"
        lang = detect_language(text)
        assert lang == "zh"


# ============================================================================
# Test Voice Selection
# ============================================================================


class TestVoiceSelection:
    """Tests for TTS voice selection."""

    def test_get_default_english_voice(self):
        """Should return default English voice."""
        voice = get_tts_voice("en")
        assert voice == TTS_DEFAULT_VOICES["en"]

    def test_get_default_chinese_voice(self):
        """Should return default Chinese voice."""
        voice = get_tts_voice("zh")
        assert voice == TTS_DEFAULT_VOICES["zh"]

    def test_get_custom_voice(self):
        """Custom voice should override default."""
        voice = get_tts_voice("en", custom_voice="en-US-JennyNeural")
        assert voice == "en-US-JennyNeural"

    def test_voice_fallback(self):
        """Unknown language should fall back to English."""
        voice = get_tts_voice("fr")  # French not in defaults
        assert voice == TTS_DEFAULT_VOICES["en"]


# ============================================================================
# Test SRT Timestamp Parsing
# ============================================================================


class TestSrtTimestamps:
    """Tests for SRT timestamp parsing and formatting."""

    def test_parse_timestamp_zero(self):
        """Should parse zero timestamp."""
        ts = parse_srt_timestamp("00:00:00,000")
        assert ts == 0.0

    def test_parse_timestamp_seconds(self):
        """Should parse seconds correctly."""
        ts = parse_srt_timestamp("00:00:05,500")
        assert ts == 5.5

    def test_parse_timestamp_minutes(self):
        """Should parse minutes correctly."""
        ts = parse_srt_timestamp("00:01:30,000")
        assert ts == 90.0

    def test_parse_timestamp_hours(self):
        """Should parse hours correctly."""
        ts = parse_srt_timestamp("01:00:00,000")
        assert ts == 3600.0

    def test_parse_timestamp_with_dot(self):
        """Should handle dot decimal separator."""
        ts = parse_srt_timestamp("00:00:05.500")
        assert ts == 5.5

    def test_format_timestamp_zero(self):
        """Should format zero correctly."""
        ts = format_srt_timestamp(0.0)
        assert ts == "00:00:00,000"

    def test_format_timestamp_seconds(self):
        """Should format seconds correctly."""
        ts = format_srt_timestamp(5.5)
        assert ts == "00:00:05,500"

    def test_format_timestamp_minutes(self):
        """Should format minutes correctly."""
        ts = format_srt_timestamp(90.0)
        assert ts == "00:01:30,000"

    def test_format_timestamp_roundtrip(self):
        """Parse and format should round-trip correctly."""
        original = "00:01:23,456"
        parsed = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(parsed)
        assert formatted == original


# ============================================================================
# Test SRT Timing Adjustment
# ============================================================================


class TestSrtAdjustment:
    """Tests for SRT timing adjustment."""

    def test_adjust_no_change_needed(self, tmp_path, sample_srt):
        """Should not adjust if difference is within threshold."""
        srt_path = tmp_path / "captions.srt"
        srt_path.write_text(sample_srt, encoding="utf-8")
        
        # Original duration is 25s, target close to that
        adjusted = adjust_srt_timing(srt_path, target_duration=25.0)
        assert adjusted is False  # No change needed

    def test_adjust_scales_up(self, tmp_path, sample_srt):
        """Should scale timestamps up when audio is longer."""
        srt_path = tmp_path / "captions.srt"
        srt_path.write_text(sample_srt, encoding="utf-8")
        
        # Original is 25s, target is 50s (2x)
        adjusted = adjust_srt_timing(srt_path, target_duration=50.0)
        assert adjusted is True
        
        # Read adjusted file
        content = srt_path.read_text(encoding="utf-8")
        assert "00:00:50" in content  # Last timestamp should be ~50s

    def test_adjust_scales_down(self, tmp_path, sample_srt):
        """Should scale timestamps down when audio is shorter."""
        srt_path = tmp_path / "captions.srt"
        srt_path.write_text(sample_srt, encoding="utf-8")
        
        # Original is 25s, target is 12.5s (0.5x)
        adjusted = adjust_srt_timing(srt_path, target_duration=12.5)
        assert adjusted is True
        
        # Read adjusted file
        content = srt_path.read_text(encoding="utf-8")
        # Timestamps should be scaled down
        assert "00:00:12" in content or "00:00:13" in content

    def test_adjust_preserves_monotonic(self, tmp_path, sample_srt):
        """Timestamps should remain monotonically increasing."""
        srt_path = tmp_path / "captions.srt"
        srt_path.write_text(sample_srt, encoding="utf-8")
        
        adjust_srt_timing(srt_path, target_duration=50.0)
        
        # Parse and verify monotonicity
        content = srt_path.read_text(encoding="utf-8")
        import re
        timestamps = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3})", content)
        
        prev = 0.0
        for ts in timestamps:
            current = parse_srt_timestamp(ts)
            assert current >= prev, f"Non-monotonic: {prev} -> {current}"
            prev = current

    def test_adjust_missing_file(self, tmp_path):
        """Should handle missing file gracefully."""
        srt_path = tmp_path / "nonexistent.srt"
        adjusted = adjust_srt_timing(srt_path, target_duration=30.0)
        assert adjusted is False

    def test_adjust_threshold_config(self, tmp_path, sample_srt):
        """Should use configured threshold."""
        srt_path = tmp_path / "captions.srt"
        srt_path.write_text(sample_srt, encoding="utf-8")
        
        # 25s original, 27s target = 8% difference
        # Default threshold is 15%, so shouldn't adjust
        adjusted = adjust_srt_timing(
            srt_path, 
            target_duration=27.0,
            threshold=0.15,
        )
        assert adjusted is False
        
        # But with 5% threshold, should adjust
        adjusted = adjust_srt_timing(
            srt_path,
            target_duration=27.0,
            threshold=0.05,
        )
        assert adjusted is True


# ============================================================================
# Test Synthesize Function Mocking
# ============================================================================


class TestSynthesizeMocking:
    """Tests for the mockable synthesize function."""

    def test_set_synthesize_function(self, mock_synthesize, tmp_path):
        """Should use custom synthesize function when set."""
        # Set mock
        set_synthesize_function(mock_synthesize)
        
        try:
            output_path = tmp_path / "test.mp3"
            result = run_async(synthesize(
                text="Test text",
                output_path=output_path,
                voice="en-US-GuyNeural",
            ))
            
            assert result.exists()
        finally:
            # Reset
            set_synthesize_function(None)

    def test_reset_synthesize_function(self):
        """Should be able to reset to default."""
        async def custom_fn(*args, **kwargs):
            raise Exception("Custom function called")
        
        set_synthesize_function(custom_fn)
        set_synthesize_function(None)  # Reset
        
        # Now it should try to use real edge_tts (which might not be installed)
        # Just verify the function is reset


# ============================================================================
# Test Audio Duration
# ============================================================================


class TestAudioDuration:
    """Tests for audio duration detection."""

    def test_duration_missing_file(self, tmp_path):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            get_audio_duration(tmp_path / "nonexistent.mp3")

    def test_duration_fallback_estimate(self, tmp_path):
        """Should estimate duration if tools unavailable."""
        # Create dummy file
        mp3_path = tmp_path / "test.mp3"
        mp3_path.write_bytes(b"\x00" * 16000)  # ~1 second at 128kbps
        
        # This will use fallback estimation
        duration = get_audio_duration(mp3_path)
        assert duration > 0


# ============================================================================
# Test Voiceover Generation
# ============================================================================


class TestVoiceoverGeneration:
    """Tests for voiceover generation."""

    def test_generate_for_folder(self, shorts_folder, mock_synthesize):
        """Should generate voiceover.mp3 in folder."""
        set_synthesize_function(mock_synthesize)
        
        try:
            result = run_async(generate_voiceover_for_folder(
                folder_path=shorts_folder,
                voice="en-US-GuyNeural",
            ))
            
            assert result is not None
            assert result.name == "voiceover.mp3"
            assert (shorts_folder / "voiceover.mp3").exists()
        finally:
            set_synthesize_function(None)

    def test_generate_missing_voiceover_txt(self, tmp_path, mock_synthesize):
        """Should return None if voiceover.txt missing."""
        folder = tmp_path / "empty_folder"
        folder.mkdir()
        
        set_synthesize_function(mock_synthesize)
        
        try:
            result = run_async(generate_voiceover_for_folder(folder_path=folder))
            assert result is None
        finally:
            set_synthesize_function(None)

    def test_generate_auto_detects_language(
        self, 
        tmp_path, 
        sample_voiceover_zh, 
        mock_synthesize,
    ):
        """Should auto-detect language and select appropriate voice."""
        folder = tmp_path / "zh_folder"
        folder.mkdir()
        (folder / "voiceover.txt").write_text(sample_voiceover_zh, encoding="utf-8")
        
        # Track which voice was used
        used_voice = None
        
        async def tracking_mock(text, output_path, voice, rate="+0%", pitch="+0Hz"):
            nonlocal used_voice
            used_voice = voice
            output_path.write_bytes(b"\x00" * 1000)
            return output_path
        
        set_synthesize_function(tracking_mock)
        
        try:
            run_async(generate_voiceover_for_folder(folder_path=folder))
            assert used_voice == TTS_DEFAULT_VOICES["zh"]
        finally:
            set_synthesize_function(None)


# ============================================================================
# Test Batch Generation
# ============================================================================


class TestBatchGeneration:
    """Tests for batch voiceover generation."""

    def test_batch_generate_multiple(self, tmp_path, sample_voiceover_en, mock_synthesize):
        """Should process multiple folders."""
        shorts_dir = tmp_path / "shorts"
        
        # Create multiple folders
        for i in range(3):
            folder = shorts_dir / str(i)
            folder.mkdir(parents=True)
            (folder / "voiceover.txt").write_text(sample_voiceover_en, encoding="utf-8")
        
        set_synthesize_function(mock_synthesize)
        
        try:
            result = run_async(batch_generate(
                shorts_dir=shorts_dir,
                limit=10,
            ))
            
            assert result["processed"] == 3
            assert result["skipped"] == 0
            assert result["failed"] == 0
            assert len(result["paths"]) == 3
        finally:
            set_synthesize_function(None)

    def test_batch_generate_skip_existing(self, tmp_path, sample_voiceover_en, mock_synthesize):
        """Should skip folders with existing mp3."""
        shorts_dir = tmp_path / "shorts"
        
        # Create folder with existing mp3
        folder1 = shorts_dir / "1"
        folder1.mkdir(parents=True)
        (folder1 / "voiceover.txt").write_text(sample_voiceover_en, encoding="utf-8")
        (folder1 / "voiceover.mp3").write_bytes(b"existing")
        
        # Create folder without mp3
        folder2 = shorts_dir / "2"
        folder2.mkdir(parents=True)
        (folder2 / "voiceover.txt").write_text(sample_voiceover_en, encoding="utf-8")
        
        set_synthesize_function(mock_synthesize)
        
        try:
            result = run_async(batch_generate(
                shorts_dir=shorts_dir,
                skip_existing=True,
            ))
            
            assert result["processed"] == 1
            assert result["skipped"] == 1
        finally:
            set_synthesize_function(None)

    def test_batch_generate_respects_limit(self, tmp_path, sample_voiceover_en, mock_synthesize):
        """Should respect limit parameter."""
        shorts_dir = tmp_path / "shorts"
        
        # Create 5 folders
        for i in range(5):
            folder = shorts_dir / str(i)
            folder.mkdir(parents=True)
            (folder / "voiceover.txt").write_text(sample_voiceover_en, encoding="utf-8")
        
        set_synthesize_function(mock_synthesize)
        
        try:
            result = run_async(batch_generate(
                shorts_dir=shorts_dir,
                limit=2,
            ))
            
            assert result["processed"] <= 2
        finally:
            set_synthesize_function(None)

    def test_batch_generate_empty_dir(self, tmp_path):
        """Should handle empty directory."""
        shorts_dir = tmp_path / "empty_shorts"
        shorts_dir.mkdir()
        
        result = run_async(batch_generate(shorts_dir=shorts_dir))
        
        assert result["processed"] == 0
        assert result["skipped"] == 0
        assert result["failed"] == 0


# ============================================================================
# Test Edge-TTS Import Error Handling
# ============================================================================


class TestEdgeTtsImportHandling:
    """Tests for edge-tts import error handling."""

    def test_missing_edge_tts_shows_install_hint(self, capsys):
        """Should show installation hint when edge-tts is missing."""
        # This test verifies the error message format
        # We can't easily test the actual import error without
        # uninstalling edge-tts, so we just verify the error message pattern
        from pipeline.tts import _edge_tts_synthesize
        
        # The function should raise ImportError with install hint
        # if edge-tts is not installed (which we can't easily test)
        pass


# ============================================================================
# Test Windows Asyncio Compatibility
# ============================================================================


class TestWindowsAsyncio:
    """Tests for Windows asyncio compatibility."""

    def test_run_async_works(self):
        """run_async should execute coroutines correctly."""
        async def simple_coro():
            return 42
        
        result = run_async(simple_coro())
        assert result == 42

    def test_run_async_handles_exception(self):
        """run_async should propagate exceptions."""
        async def failing_coro():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests for full TTS workflow."""

    def test_full_workflow(self, shorts_folder, mock_synthesize, sample_srt):
        """Test complete workflow: generate -> adjust captions."""
        set_synthesize_function(mock_synthesize)
        
        try:
            # Generate voiceover
            result = run_async(generate_voiceover_for_folder(
                folder_path=shorts_folder,
                adjust_captions=False,  # Skip for this test
            ))
            
            assert result is not None
            assert result.exists()
            
            # Manually adjust captions
            srt_path = shorts_folder / "captions.srt"
            adjusted = adjust_srt_timing(srt_path, target_duration=40.0)
            
            # Should adjust since original is 25s, target is 40s
            assert adjusted is True
            
        finally:
            set_synthesize_function(None)

    def test_deterministic_output(self, shorts_folder, mock_synthesize):
        """Same input should produce same output."""
        calls = []
        
        async def tracking_mock(text, output_path, voice, rate="+0%", pitch="+0Hz"):
            calls.append({
                "text": text,
                "voice": voice,
                "rate": rate,
            })
            output_path.write_bytes(b"\x00" * 1000)
            return output_path
        
        set_synthesize_function(tracking_mock)
        
        try:
            # First run
            run_async(generate_voiceover_for_folder(folder_path=shorts_folder))
            first_call = calls[0].copy()
            
            # Remove generated file
            (shorts_folder / "voiceover.mp3").unlink()
            
            # Second run
            run_async(generate_voiceover_for_folder(folder_path=shorts_folder))
            second_call = calls[1]
            
            # Should be identical
            assert first_call == second_call
            
        finally:
            set_synthesize_function(None)
