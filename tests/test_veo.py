"""
Tests for Veo video generation pipeline.

Uses mocks to test without calling actual APIs.
"""

import csv
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.veo import (
    VeoResult,
    ShotlistEntry,
    GeminiVeoProvider,
    VertexVeoProvider,
    UnifiedVeoProvider,
    VeoProvider,
    parse_shotlist,
    write_shotlist,
)
from pipeline.shotlist_generator import (
    generate_shotlist_with_llm,
    write_shotlist_csv,
    read_shotlist_csv,
    _generate_template_shotlist,
    _parse_llm_response,
    ShotlistEntry as ShotlistEntry2,
)
from pipeline.compose_short import (
    escape_ffmpeg_path,
    check_ffmpeg,
    ComposeResult,
)


# ============================================================================
# VeoResult Tests
# ============================================================================

def test_veo_result_default():
    """Test VeoResult default values."""
    result = VeoResult()
    
    assert result.success is False
    assert result.local_path is None
    assert result.prompt == ""
    assert result.error is None
    assert result.provider == ""


def test_veo_result_to_dict():
    """Test VeoResult serialization."""
    result = VeoResult(
        success=True,
        local_path=Path("/test/video.mp4"),
        prompt="Test prompt",
        duration_sec=8.0,
        latency_ms=5000,
        provider="gemini",
        model="veo-3",
    )
    
    d = result.to_dict()
    
    assert d["success"] is True
    # Path serialization may differ on Windows vs Unix
    assert "video.mp4" in d["local_path"]
    assert d["prompt"] == "Test prompt"
    assert d["duration_sec"] == 8.0
    assert d["provider"] == "gemini"


# ============================================================================
# ShotlistEntry Tests
# ============================================================================

def test_shotlist_entry_duration():
    """Test ShotlistEntry duration calculation."""
    entry = ShotlistEntry(
        shot_id=1,
        segment="hook",
        start_time=0.0,
        end_time=5.0,
        visual_description="Test visual",
    )
    
    assert entry.duration == 5.0


def test_shotlist_entry_to_veo_prompt():
    """Test ShotlistEntry Veo prompt generation."""
    entry = ShotlistEntry(
        shot_id=1,
        segment="hook",
        start_time=0.0,
        end_time=5.0,
        visual_description="Golden statue in morning light",
    )
    
    prompt = entry.to_veo_prompt()
    
    assert "Golden statue in morning light" in prompt
    assert "vertical video 9:16" in prompt
    assert "no text overlays" in prompt
    assert "no logos or watermarks" in prompt


def test_shotlist_entry_to_veo_prompt_custom_style():
    """Test ShotlistEntry with custom style."""
    entry = ShotlistEntry(
        shot_id=1,
        segment="hook",
        start_time=0.0,
        end_time=5.0,
        visual_description="Test",
    )
    
    prompt = entry.to_veo_prompt(style="anime style")
    
    assert "anime style" in prompt


# ============================================================================
# Shotlist I/O Tests
# ============================================================================

def test_parse_shotlist():
    """Test parsing shotlist CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("shot_id,segment,start_time,end_time,visual_description,notes\n")
        f.write("1,hook,0,5,Test visual 1,Note 1\n")
        f.write("2,context,5,15,Test visual 2,Note 2\n")
        csv_path = Path(f.name)
    
    try:
        entries = parse_shotlist(csv_path)
        
        assert len(entries) == 2
        assert entries[0].shot_id == 1
        assert entries[0].segment == "hook"
        assert entries[0].start_time == 0.0
        assert entries[0].end_time == 5.0
        assert entries[0].visual_description == "Test visual 1"
        assert entries[1].shot_id == 2
    finally:
        csv_path.unlink()


def test_write_shotlist():
    """Test writing shotlist CSV."""
    entries = [
        ShotlistEntry(1, "hook", 0, 5, "Visual 1", "Note 1"),
        ShotlistEntry(2, "context", 5, 15, "Visual 2", "Note 2"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "shotlist.csv"
        write_shotlist(entries, csv_path)
        
        assert csv_path.exists()
        
        # Read back and verify
        read_entries = parse_shotlist(csv_path)
        assert len(read_entries) == 2
        assert read_entries[0].segment == "hook"


# ============================================================================
# Shotlist Generator Tests
# ============================================================================

def test_generate_template_shotlist():
    """Test template-based shotlist generation."""
    entries = _generate_template_shotlist(
        script="A script about ancient Rome",
        topic="Roman Colosseum history",
        duration_hint=60.0,
        num_shots=4,
    )
    
    assert len(entries) == 4
    assert entries[0].segment == "hook"
    assert entries[1].segment == "context"
    assert entries[2].segment == "contrast"
    assert entries[3].segment == "resolution"
    
    # Check durations
    assert entries[0].start_time == 0.0
    assert entries[0].end_time == 15.0
    assert entries[3].end_time == 60.0


def test_generate_template_shotlist_mythology():
    """Test template generation for mythology topic."""
    entries = _generate_template_shotlist(
        script="A story about Zeus",
        topic="Greek myths Zeus thunder",
        duration_hint=60.0,
        num_shots=4,
    )
    
    # Should use mythology templates
    assert "statue" in entries[0].visual_description.lower() or "storm" in entries[0].visual_description.lower()


def test_parse_llm_response_valid():
    """Test parsing valid LLM response."""
    response = json.dumps({
        "shots": [
            {
                "shot_id": 1,
                "segment": "hook",
                "start_time": 0,
                "end_time": 5,
                "visual_description": "Test visual",
                "notes": "Test note",
            }
        ]
    })
    
    entries = _parse_llm_response(response)
    
    assert len(entries) == 1
    assert entries[0].shot_id == 1
    assert entries[0].visual_description == "Test visual"


def test_parse_llm_response_with_markdown():
    """Test parsing LLM response with markdown code blocks."""
    response = """```json
{
    "shots": [
        {"shot_id": 1, "segment": "hook", "start_time": 0, "end_time": 5, "visual_description": "Test"}
    ]
}
```"""
    
    entries = _parse_llm_response(response)
    
    assert len(entries) == 1


def test_parse_llm_response_invalid():
    """Test parsing invalid LLM response."""
    entries = _parse_llm_response("Not valid JSON")
    
    assert len(entries) == 0


# ============================================================================
# Provider Tests
# ============================================================================

def test_gemini_provider_not_available():
    """Test Gemini provider without API key."""
    with patch.dict('os.environ', {'GEMINI_API_KEY': ''}, clear=False):
        # Need to reimport to pick up new env
        provider = GeminiVeoProvider()
        provider.api_key = ""
        
        assert provider.is_available() is False


def test_vertex_provider_not_available():
    """Test Vertex provider without credentials."""
    with patch.dict('os.environ', {'VERTEX_PROJECT_ID': '', 'GOOGLE_APPLICATION_CREDENTIALS': ''}, clear=False):
        provider = VertexVeoProvider()
        provider.project_id = ""
        
        assert provider.is_available() is False


def test_unified_provider_empty():
    """Test unified provider with no providers available."""
    with patch.dict('os.environ', {'GEMINI_API_KEY': '', 'VERTEX_PROJECT_ID': ''}, clear=False):
        provider = UnifiedVeoProvider()
        provider.gemini.api_key = ""
        provider.vertex.project_id = ""
        
        available = provider.get_available_providers()
        assert available == []


@pytest.mark.asyncio
async def test_unified_provider_generate_no_providers():
    """Test unified provider generate with no providers."""
    provider = UnifiedVeoProvider()
    provider.gemini.api_key = ""
    provider.vertex.project_id = ""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await provider.generate_broll(
            prompt="Test prompt",
            output_path=Path(tmpdir) / "test.mp4",
        )
        
        assert result.success is False
        assert "unavailable" in result.error.lower() or "failed" in result.error.lower()


# ============================================================================
# Compose Tests
# ============================================================================

def test_escape_ffmpeg_path_windows():
    """Test FFmpeg path escaping for Windows."""
    path = Path("C:/Users/test/video.mp4")
    escaped = escape_ffmpeg_path(path)
    
    # On Windows, colons are escaped with backslash for FFmpeg filter compatibility
    # The result should have forward slashes for path separators
    assert "/" in escaped  # Uses forward slashes
    assert "video.mp4" in escaped


def test_escape_ffmpeg_path_unix():
    """Test FFmpeg path escaping for Unix-style paths."""
    # Note: On Windows, Path("/home/test/video.mp4") may be converted
    # to absolute Windows path. Test the escaping behavior instead.
    path = Path("test/subfolder/video.mp4")
    escaped = escape_ffmpeg_path(path)
    
    # Should have forward slashes
    assert "/" in escaped or "video.mp4" in escaped


def test_compose_result_default():
    """Test ComposeResult default values."""
    result = ComposeResult()
    
    assert result.success is False
    assert result.background_path is None
    assert result.final_path is None
    assert result.clips_used == 0
    assert result.fallback_used is False


def test_compose_result_to_dict():
    """Test ComposeResult serialization."""
    result = ComposeResult(
        success=True,
        background_path=Path("/test/bg.mp4"),
        final_path=Path("/test/final.mp4"),
        duration=60.0,
        clips_used=4,
        fallback_used=False,
    )
    
    d = result.to_dict()
    
    assert d["success"] is True
    assert d["duration"] == 60.0
    assert d["clips_used"] == 4


# ============================================================================
# Integration Tests (with mocks)
# ============================================================================

@pytest.mark.asyncio
async def test_generate_shotlist_with_llm_fallback():
    """Test shotlist generation falls back to template."""
    with patch.dict('os.environ', {'GEMINI_API_KEY': ''}, clear=False):
        entries = await generate_shotlist_with_llm(
            script="Test script about history",
            topic="Ancient Rome",
            duration_hint=60.0,
            num_shots=4,
        )
        
        # Should fall back to template
        assert len(entries) == 4
        assert entries[0].segment == "hook"


@pytest.mark.asyncio
async def test_gemini_provider_generate_mock():
    """Test Gemini provider with mocked response."""
    provider = GeminiVeoProvider()
    provider.api_key = "test_key"
    provider._available_model = "veo-2"
    
    # Mock the generate_content call
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = []
    
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await provider.generate_video(
                    prompt="Test prompt",
                    output_path=Path(tmpdir) / "test.mp4",
                )
                
                # Should fail gracefully since no video in response
                assert result.success is False
                assert result.provider == "gemini"


# ============================================================================
# Shotlist Generator CSV I/O Tests
# ============================================================================

def test_write_and_read_shotlist_csv():
    """Test roundtrip write/read of shotlist CSV."""
    entries = [
        ShotlistEntry2(
            shot_id=1,
            segment="hook",
            start_time=0.0,
            end_time=5.0,
            visual_description="Golden man in lake",
            notes="Opening",
        ),
        ShotlistEntry2(
            shot_id=2,
            segment="context",
            start_time=5.0,
            end_time=20.0,
            visual_description="Spanish conquistadors",
            notes="History",
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_shotlist.csv"
        
        write_shotlist_csv(entries, csv_path)
        assert csv_path.exists()
        
        read_entries = read_shotlist_csv(csv_path)
        
        assert len(read_entries) == 2
        assert read_entries[0].shot_id == 1
        assert read_entries[0].visual_description == "Golden man in lake"
        assert read_entries[1].segment == "context"


# ============================================================================
# Prompt Builder Tests
# ============================================================================

def test_shotlist_prompt_builder():
    """Test that shotlist prompts are correctly built."""
    entry = ShotlistEntry(
        shot_id=1,
        segment="hook",
        start_time=0.0,
        end_time=5.0,
        visual_description="Close-up of ancient artifact",
    )
    
    prompt = entry.to_veo_prompt(style="cinematic documentary")
    
    # Check required elements
    assert "Close-up of ancient artifact" in prompt
    assert "vertical video 9:16 aspect ratio" in prompt
    assert "smooth camera movement" in prompt
    assert "no text overlays" in prompt
    assert "no logos or watermarks" in prompt
    assert "no subtitles" in prompt
    assert "cinematic documentary" in prompt
    assert "1080p" in prompt


# ============================================================================
# FFmpeg Command Builder Tests (compose_short)
# ============================================================================

def test_ffmpeg_concat_command_builder():
    """Test that FFmpeg concat command is correctly structured."""
    # This is a conceptual test - actual command building is in compose_short.py
    # We just verify the path escaping produces valid FFmpeg filter paths
    
    test_paths = [
        Path("relative/video1.mp4"),
        Path("another/path/video2.mp4"),
    ]
    
    for path in test_paths:
        escaped = escape_ffmpeg_path(path)
        # Escaped paths should use forward slashes for FFmpeg compatibility
        assert "/" in escaped or "video" in escaped
        # Should contain the filename
        assert path.name in escaped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
