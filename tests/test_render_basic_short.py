"""
Tests for pipeline/render_basic_short.py

FFmpeg rendering tests with mocked subprocess calls.
Includes tests for explicit audio mapping, audio normalization, and validation.
"""

import csv
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.constants import (
    RENDER_AUDIO_CODEC,
    RENDER_DEFAULT_BG_COLOR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_SUBTITLE_STYLE,
    RENDER_VIDEO_CODEC,
    RENDER_WIDTH,
    SHORTS_REQUIRED_FILES,
    ShortsStatus,
    VOICEOVER_MIN_DURATION_SECONDS,
    determine_shorts_status,
)
from pipeline.render_basic_short import (
    AudioInfo,
    FFmpegNotFoundError,
    RenderValidationError,
    VideoInfo,
    build_ffmpeg_command,
    check_ffmpeg,
    copy_srt_to_temp,
    escape_ffmpeg_path,
    get_audio_duration,
    probe_audio_file,
    probe_video_file,
    render_from_queue,
    render_single,
    update_queue_after_render,
    validate_before_render,
    validate_after_render,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ready_shorts_folder(tmp_path) -> Path:
    """Create a ready shorts folder with all required files."""
    folder = tmp_path / "shorts" / "42"
    folder.mkdir(parents=True)
    
    # Create required files
    for filename in SHORTS_REQUIRED_FILES:
        if filename == "captions.srt":
            content = """1
00:00:00,000 --> 00:00:05,000
First line.

2
00:00:05,000 --> 00:00:10,000
Second line.
"""
            (folder / filename).write_text(content, encoding="utf-8")
        else:
            (folder / filename).write_text(f"Mock {filename}", encoding="utf-8")
    
    # Create voiceover.mp3 (mock audio file)
    (folder / "voiceover.mp3").write_bytes(b"\x00" * 10000)
    
    return folder


@pytest.fixture
def mock_queue_csv(tmp_path) -> Path:
    """Create a mock queue CSV with ready items."""
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
            "status": "ready",
        },
        {
            "rank": "2",
            "claim_id": "43",
            "topic": "health",
            "verdict": "True",
            "confidence": "0.9",
            "title": "Health Fact",
            "hook": "Scientists found...",
            "estimated_seconds": "40",
            "folder_path": "outputs/shorts/43",
            "status": "needs_tts",  # Not ready
        },
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return csv_path


# ============================================================================
# Test FFmpeg Check
# ============================================================================


class TestFFmpegCheck:
    """Tests for ffmpeg availability checking."""

    def test_check_ffmpeg_available(self):
        """Should return version when ffmpeg is available."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright (c) 2000-2022"
        
        with patch("subprocess.run", return_value=mock_result):
            version = check_ffmpeg()
            assert "ffmpeg" in version.lower()

    def test_check_ffmpeg_not_found(self):
        """Should raise error with install instructions when not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FFmpegNotFoundError) as exc_info:
                check_ffmpeg()
            
            error_msg = str(exc_info.value)
            assert "not found" in error_msg.lower()
            assert "install" in error_msg.lower()

    def test_check_ffmpeg_timeout(self):
        """Should handle timeout gracefully."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10)):
            with pytest.raises(FFmpegNotFoundError) as exc_info:
                check_ffmpeg()
            
            assert "timeout" in str(exc_info.value).lower()


# ============================================================================
# Test Path Escaping
# ============================================================================


class TestPathEscaping:
    """Tests for Windows path escaping in ffmpeg commands."""

    def test_escape_simple_path(self, tmp_path):
        """Should escape simple paths correctly."""
        path = tmp_path / "test.srt"
        escaped = escape_ffmpeg_path(path)
        
        # Should use forward slashes
        assert "\\" not in escaped or "\\:" in escaped
        # Should escape drive letter colon on Windows
        if ":" in str(path):
            assert "\\:" in escaped

    def test_escape_path_with_spaces(self, tmp_path):
        """Should handle paths with spaces."""
        path = tmp_path / "path with spaces" / "test.srt"
        escaped = escape_ffmpeg_path(path)
        
        # Forward slashes should be used
        assert "/" in escaped

    def test_escape_windows_path(self):
        """Should properly escape Windows-style paths."""
        # Simulate a Windows path
        path = Path("C:/Users/test/shorts/42/captions.srt")
        escaped = escape_ffmpeg_path(path)
        
        # Colon after drive letter should be escaped
        # Note: The exact escaping depends on the path format
        assert "/" in escaped  # Forward slashes used


# ============================================================================
# Test FFmpeg Command Building
# ============================================================================


class TestFFmpegCommand:
    """Tests for ffmpeg command construction."""

    def test_command_structure_black_background(self, tmp_path):
        """Should build correct command with black background."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
        )
        
        # Check basic structure
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd  # Overwrite flag
        
        # Check video/audio inputs
        assert str(audio) in cmd
        
        # Check output
        assert str(output) in cmd

    def test_command_includes_resolution(self, tmp_path):
        """Should include correct resolution in command."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
            width=1080,
            height=1920,
        )
        
        cmd_str = " ".join(cmd)
        assert "1080" in cmd_str
        assert "1920" in cmd_str

    def test_command_includes_subtitles_filter(self, tmp_path):
        """Should include subtitles filter in command."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
        )
        
        # Find the -vf argument and check for subtitles
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        
        assert "subtitles" in vf_value

    def test_command_includes_shortest_flag(self, tmp_path):
        """Should include -shortest flag."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
        )
        
        assert "-shortest" in cmd

    def test_command_includes_codecs(self, tmp_path):
        """Should include correct video and audio codecs."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
        )
        
        # Video codec
        cv_idx = cmd.index("-c:v")
        assert cmd[cv_idx + 1] == RENDER_VIDEO_CODEC
        
        # Audio codec
        ca_idx = cmd.index("-c:a")
        assert cmd[ca_idx + 1] == RENDER_AUDIO_CODEC

    def test_command_with_background_image(self, tmp_path):
        """Should handle background image input."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x00" * 100)
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
            background_path=bg,
        )
        
        # Should include loop for image
        assert "-loop" in cmd

    def test_command_has_explicit_audio_map(self, tmp_path):
        """Should explicitly map audio from input 1 to ensure audio is included."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
        )
        
        # Must have explicit -map for both video and audio
        assert "-map" in cmd
        
        # Find all -map arguments
        map_args = []
        for i, arg in enumerate(cmd):
            if arg == "-map":
                map_args.append(cmd[i + 1])
        
        # Should have video from input 0 and audio from input 1
        assert len(map_args) >= 2
        assert any("0:v" in m for m in map_args), "Should map video from input 0"
        assert any("1:a" in m for m in map_args), "Should map audio from input 1"

    def test_command_has_audio_filter_loudnorm(self, tmp_path):
        """Should include loudnorm audio filter by default."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
            use_loudnorm=True,
        )
        
        # Should have -af with loudnorm
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        af_value = cmd[af_idx + 1]
        
        assert "loudnorm" in af_value
        assert "I=-16" in af_value  # Target loudness
        assert "TP=-1.5" in af_value  # True peak

    def test_command_has_audio_filter_volume_gain(self, tmp_path):
        """Should use volume gain when loudnorm is disabled."""
        audio = tmp_path / "audio.mp3"
        srt = tmp_path / "subs.srt"
        output = tmp_path / "out.mp4"
        
        cmd = build_ffmpeg_command(
            audio_path=audio,
            srt_path=srt,
            output_path=output,
            duration=35.0,
            use_loudnorm=False,
            audio_gain_db=8.0,
        )
        
        # Should have -af with volume
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        af_value = cmd[af_idx + 1]
        
        assert "volume" in af_value
        assert "8" in af_value  # Gain value


# ============================================================================
# Test Audio/Video Probing
# ============================================================================


class TestAudioVideoProbing:
    """Tests for ffprobe-based file probing."""

    def test_probe_audio_file_valid(self, tmp_path):
        """Should probe valid audio file."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "35.5"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            info = probe_audio_file(audio)
        
        assert info.exists
        assert abs(info.duration - 35.5) < 0.1
        assert info.has_audio_stream
        assert info.codec == "mp3"
        assert info.error is None

    def test_probe_audio_file_not_exists(self, tmp_path):
        """Should handle non-existent audio file."""
        audio = tmp_path / "nonexistent.mp3"
        
        info = probe_audio_file(audio)
        
        assert not info.exists
        assert info.duration == 0
        assert not info.has_audio_stream
        assert "not exist" in info.error.lower()

    def test_probe_video_file_with_audio(self, tmp_path):
        """Should detect audio stream in video file."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"}
            ]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            info = probe_video_file(video)
        
        assert info.exists
        assert abs(info.duration - 30.0) < 0.1
        assert info.has_video_stream
        assert info.has_audio_stream
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"

    def test_probe_video_file_no_audio(self, tmp_path):
        """Should detect missing audio stream."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"}
                # No audio stream!
            ]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            info = probe_video_file(video)
        
        assert info.has_video_stream
        assert not info.has_audio_stream  # Critical: no audio!


# ============================================================================
# Test Pre/Post Render Validation
# ============================================================================


class TestRenderValidation:
    """Tests for pre/post render validation."""

    def test_validate_before_render_audio_missing(self, tmp_path):
        """Should fail if voiceover.mp3 is missing."""
        folder = tmp_path / "shorts" / "99"
        folder.mkdir(parents=True)
        
        is_valid, error, info = validate_before_render(folder)
        
        assert not is_valid
        assert "not exist" in error.lower()

    def test_validate_before_render_audio_too_short(self, tmp_path):
        """Should fail if audio duration < minimum."""
        folder = tmp_path / "shorts" / "99"
        folder.mkdir(parents=True)
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 100)
        
        # Mock ffprobe to return short duration
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "5.0"},  # Too short!
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, error, info = validate_before_render(folder, min_duration=20.0)
        
        assert not is_valid
        assert "too short" in error.lower()
        assert "5.0s" in error

    def test_validate_before_render_audio_valid(self, tmp_path):
        """Should pass with valid audio."""
        folder = tmp_path / "shorts" / "99"
        folder.mkdir(parents=True)
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "35.0"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, error, info = validate_before_render(folder, min_duration=20.0)
        
        assert is_valid
        assert error == ""
        assert info.duration >= 20.0

    def test_validate_after_render_no_audio_stream(self, tmp_path):
        """Should fail if output has no audio stream (critical bug)."""
        video = tmp_path / "final.mp4"
        video.write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"}
                # NO AUDIO STREAM - this is the bug we're catching!
            ]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, error, info = validate_after_render(video)
        
        assert not is_valid
        assert "no audio" in error.lower()

    def test_validate_after_render_with_audio(self, tmp_path):
        """Should pass with valid video containing audio."""
        video = tmp_path / "final.mp4"
        video.write_bytes(b"\x00" * 1000)
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"}
            ]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            is_valid, error, info = validate_after_render(video)
        
        assert is_valid
        assert info.has_audio_stream
        assert info.audio_codec == "aac"


# ============================================================================
# Test Audio Duration
# ============================================================================


class TestAudioDuration:
    """Tests for audio duration detection."""

    def test_get_audio_duration_ffprobe(self, tmp_path):
        """Should get duration from ffprobe."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00" * 1000)
        
        # ffprobe returns JSON with format and streams
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"duration": "35.5"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}]
        })
        
        with patch("subprocess.run", return_value=mock_result):
            duration = get_audio_duration(audio)
            assert abs(duration - 35.5) < 0.1

    def test_get_audio_duration_fallback(self, tmp_path):
        """Should fallback to estimate on ffprobe failure."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00" * 160000)  # ~10 seconds at 128kbps
        
        with patch("subprocess.run", side_effect=FileNotFoundError):
            duration = get_audio_duration(audio)
            # Should return an estimate based on file size
            assert duration > 0


# ============================================================================
# Test Render Single
# ============================================================================


class TestRenderSingle:
    """Tests for single folder rendering."""

    def test_render_single_skips_already_rendered(self, tmp_path):
        """Should skip folders that are already rendered."""
        folder = tmp_path / "shorts" / "45"
        folder.mkdir(parents=True)
        
        for f in SHORTS_REQUIRED_FILES:
            (folder / f).write_text("test")
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 100)
        (folder / "final.mp4").write_bytes(b"\x00" * 1000)
        
        result = render_single(folder)
        
        assert result == folder / "final.mp4"

    def test_render_single_skips_not_ready(self, tmp_path):
        """Should skip folders that are not ready."""
        folder = tmp_path / "shorts" / "46"
        folder.mkdir(parents=True)
        
        # Only create some files (not ready)
        (folder / "voiceover.txt").write_text("test")
        
        result = render_single(folder)
        
        assert result is None

    def test_render_single_mock_ffmpeg(self, ready_shorts_folder):
        """Should call ffmpeg correctly for ready folder."""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = ""
        mock_subprocess_result.stderr = ""
        
        # Track ffmpeg calls (filtering out ffprobe calls)
        ffmpeg_calls = []
        
        def mock_run(cmd, *args, **kwargs):
            cmd_name = cmd[0] if isinstance(cmd, list) else cmd
            
            if "ffprobe" in str(cmd_name):
                # Return valid ffprobe JSON for audio/video probing
                probe_result = MagicMock()
                probe_result.returncode = 0
                probe_result.stdout = json.dumps({
                    "format": {"duration": "35.0"},
                    "streams": [
                        {"codec_type": "audio", "codec_name": "aac"},
                        {"codec_type": "video", "codec_name": "h264"}
                    ]
                })
                return probe_result
            elif "ffmpeg" in str(cmd_name):
                ffmpeg_calls.append(cmd)
                # Create the output file to simulate success
                output_path = ready_shorts_folder / "final.mp4"
                output_path.write_bytes(b"\x00" * 1000)
                return mock_subprocess_result
            
            return mock_subprocess_result
        
        with patch("subprocess.run", side_effect=mock_run):
            result = render_single(ready_shorts_folder)
        
        # Should have called ffmpeg at least once
        assert len(ffmpeg_calls) >= 1
        
        # The ffmpeg command should be correct
        ffmpeg_cmd = ffmpeg_calls[0]
        assert ffmpeg_cmd[0] == "ffmpeg"


# ============================================================================
# Test Render From Queue
# ============================================================================


class TestRenderFromQueue:
    """Tests for queue-based rendering."""

    def test_render_from_queue_skips_non_ready(self, mock_queue_csv, tmp_path):
        """Should only render ready items."""
        queue_dir = tmp_path / "shorts_queue"
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir(parents=True)
        
        # Create folder for claim 42 (ready in queue)
        folder_42 = shorts_dir / "42"
        folder_42.mkdir()
        for f in SHORTS_REQUIRED_FILES:
            (folder_42 / f).write_text("test")
        (folder_42 / "voiceover.mp3").write_bytes(b"\x00" * 100)
        
        # Mock ffmpeg success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        
        with patch("subprocess.run", return_value=mock_result):
            with patch("pipeline.render_basic_short.get_audio_duration", return_value=35.0):
                # Create output on render
                def create_output(*args, **kwargs):
                    (folder_42 / "final.mp4").write_bytes(b"\x00" * 1000)
                    return mock_result
                
                with patch("subprocess.run", side_effect=create_output):
                    result = render_from_queue(
                        queue_date="2026-01-03",
                        limit=10,
                        queue_dir=queue_dir,
                        shorts_dir=shorts_dir,
                    )
        
        # Claim 43 should be skipped (needs_tts)
        assert result["skipped"] >= 1


# ============================================================================
# Test Queue Update After Render
# ============================================================================


class TestQueueUpdateAfterRender:
    """Tests for queue status update after rendering."""

    def test_update_queue_after_render(self, mock_queue_csv, tmp_path):
        """Should update queue CSV with rendered status."""
        queue_dir = tmp_path / "shorts_queue"
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir(parents=True)
        
        # Create rendered folder
        folder_42 = shorts_dir / "42"
        folder_42.mkdir()
        for f in SHORTS_REQUIRED_FILES:
            (folder_42 / f).write_text("test")
        (folder_42 / "voiceover.mp3").write_bytes(b"\x00" * 100)
        (folder_42 / "final.mp4").write_bytes(b"\x00" * 1000)
        
        update_queue_after_render("2026-01-03", queue_dir, shorts_dir)
        
        # Read updated queue
        csv_path = queue_dir / "queue_2026-01-03.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        row_42 = next(r for r in rows if r["claim_id"] == "42")
        assert row_42["status"] == "rendered"


# ============================================================================
# Test SRT Copy
# ============================================================================


class TestSrtCopy:
    """Tests for SRT temp file copying."""

    def test_copy_srt_to_temp(self, tmp_path):
        """Should copy SRT to temp with simple filename."""
        srt = tmp_path / "original.srt"
        srt.write_text("test content", encoding="utf-8")
        
        temp_srt = copy_srt_to_temp(srt)
        
        assert temp_srt.exists()
        assert temp_srt.read_text(encoding="utf-8") == "test content"
        
        # Filename should not have spaces
        assert " " not in temp_srt.name

    def test_copy_srt_to_temp_deterministic(self, tmp_path):
        """Same source should produce same temp filename."""
        srt = tmp_path / "test.srt"
        srt.write_text("content", encoding="utf-8")
        
        temp1 = copy_srt_to_temp(srt)
        temp2 = copy_srt_to_temp(srt)
        
        assert temp1 == temp2


# ============================================================================
# Test Constants
# ============================================================================


class TestRenderConstants:
    """Tests for render constants."""

    def test_render_dimensions(self):
        """Should have correct vertical video dimensions."""
        assert RENDER_WIDTH == 1080
        assert RENDER_HEIGHT == 1920
        assert RENDER_HEIGHT > RENDER_WIDTH  # Vertical

    def test_render_fps(self):
        """Should have reasonable FPS."""
        assert RENDER_FPS in (30, 60)

    def test_render_codecs(self):
        """Should use standard codecs."""
        assert RENDER_VIDEO_CODEC == "libx264"
        assert RENDER_AUDIO_CODEC == "aac"


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_render_handles_ffmpeg_failure(self, ready_shorts_folder):
        """Should handle ffmpeg returning error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: something went wrong"
        
        with patch("subprocess.run", return_value=mock_result):
            with patch("pipeline.render_basic_short.get_audio_duration", return_value=35.0):
                result = render_single(ready_shorts_folder)
        
        assert result is None

    def test_render_handles_timeout(self, ready_shorts_folder):
        """Should handle ffmpeg timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 300)):
            with patch("pipeline.render_basic_short.get_audio_duration", return_value=35.0):
                result = render_single(ready_shorts_folder)
        
        assert result is None


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests for render workflow."""

    def test_full_render_workflow_mock(self, tmp_path):
        """Test complete render workflow with mocks."""
        shorts_dir = tmp_path / "shorts"
        queue_dir = tmp_path / "queue"
        shorts_dir.mkdir()
        queue_dir.mkdir()
        
        # Create ready folder
        folder = shorts_dir / "100"
        folder.mkdir()
        for f in SHORTS_REQUIRED_FILES:
            if f == "captions.srt":
                content = """1
00:00:00,000 --> 00:00:10,000
Test subtitle.
"""
                (folder / f).write_text(content, encoding="utf-8")
            else:
                (folder / f).write_text("test")
        (folder / "voiceover.mp3").write_bytes(b"\x00" * 1000)
        
        # Create queue CSV
        csv_path = queue_dir / "queue_2026-01-03.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "rank", "claim_id", "topic", "verdict", "confidence",
                "title", "hook", "estimated_seconds", "folder_path", "status"
            ])
            writer.writeheader()
            writer.writerow({
                "rank": "1",
                "claim_id": "100",
                "topic": "science",
                "verdict": "False",
                "confidence": "0.9",
                "title": "Test",
                "hook": "Hook",
                "estimated_seconds": "30",
                "folder_path": f"outputs/shorts/100",
                "status": "ready",
            })
        
        # Mock subprocess for both ffprobe and ffmpeg
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        
        def mock_run(cmd, *args, **kwargs):
            cmd_name = cmd[0] if isinstance(cmd, list) else cmd
            
            if "ffprobe" in str(cmd_name):
                # Return valid ffprobe JSON for audio/video probing
                probe_result = MagicMock()
                probe_result.returncode = 0
                probe_result.stdout = json.dumps({
                    "format": {"duration": "30.0"},
                    "streams": [
                        {"codec_type": "audio", "codec_name": "aac"},
                        {"codec_type": "video", "codec_name": "h264"}
                    ]
                })
                return probe_result
            elif "ffmpeg" in str(cmd_name):
                # Find output path in command (last argument)
                output_idx = len(cmd) - 1
                output_path = Path(cmd[output_idx])
                output_path.write_bytes(b"\x00" * 5000)
                return mock_result
            
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            result = render_from_queue(
                queue_date="2026-01-03",
                queue_dir=queue_dir,
                shorts_dir=shorts_dir,
            )
        
        # Should have rendered 1 video
        assert result["rendered"] == 1
        
        # final.mp4 should exist
        assert (folder / "final.mp4").exists()
