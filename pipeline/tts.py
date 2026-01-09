"""
Myth Museum - TTS Voice-over Generation

Generate MP3 audio from voiceover scripts using multiple TTS providers:
- edge-tts (default, free, Microsoft Edge TTS)
- http (OpenAI-compatible API, requires API key)

Also handles SRT caption timing adjustment.
"""

import asyncio
import json
import os
import platform
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from core.constants import (
    SRT_TIMING_ADJUSTMENT_THRESHOLD,
    TTS_DEFAULT_PITCH,
    TTS_DEFAULT_RATE,
    TTS_DEFAULT_VOICES,
    detect_language,
    get_tts_voice,
)
from core.logging import get_logger

logger = get_logger(__name__)
console = Console()


# ============================================================================
# TTS Provider Abstraction
# ============================================================================


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Path:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Output MP3 path
            voice: Voice name (provider-specific)
            rate: Speech rate adjustment
            pitch: Pitch adjustment
        
        Returns:
            Path to generated MP3 file
        
        Raises:
            Exception: If synthesis fails
        """
        pass


# Global synthesize function override (can be replaced for testing)
# Must be defined before the provider classes that reference it
_synthesize_fn: Optional[Callable] = None


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS provider (free, no API key required)."""
    
    @property
    def name(self) -> str:
        return "edge-tts"
    
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Path:
        """Synthesize using edge-tts."""
        global _synthesize_fn
        
        # Use override function if set (for testing)
        if _synthesize_fn is not None:
            return await _synthesize_fn(text, output_path, voice, rate, pitch)
        
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts is not installed. Install it with: pip install edge-tts"
            )
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create communicate object
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        
        # Save to file
        await communicate.save(str(output_path))
        logger.debug(f"edge-tts saved to {output_path}")
        
        return output_path


class HttpTTSProvider(TTSProvider):
    """
    OpenAI-compatible HTTP TTS provider.
    
    Supports:
    - OpenAI TTS API (tts-1, tts-1-hd)
    - Any OpenAI-compatible endpoint
    
    Environment variables:
    - OPENAI_API_KEY: Required API key
    - OPENAI_BASE_URL: API base URL (default: https://api.openai.com/v1)
    - OPENAI_MODEL_TTS: TTS model name (default: tts-1)
    - OPENAI_TTS_VOICE: Voice name (default: alloy)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TTS_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL_TTS", "tts-1")
        self.default_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
        
        if not self.api_key:
            raise ValueError(
                "HTTP TTS provider requires OPENAI_API_KEY environment variable.\n"
                "Set it in your environment or .env file:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "Or use edge-tts provider (free, no API key): --provider edge"
            )
    
    @property
    def name(self) -> str:
        return f"http ({self.base_url})"
    
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Path:
        """
        Synthesize using OpenAI-compatible TTS API.
        
        Note: rate and pitch are not supported by OpenAI TTS API.
        They are ignored for HTTP provider.
        """
        import aiohttp
        
        # Map voice names if needed
        # edge-tts voices like "en-US-AriaNeural" -> OpenAI voices like "alloy"
        openai_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in openai_voices:
            voice = self.default_voice
            logger.debug(f"Using default voice '{voice}' for HTTP provider")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
        }
        
        logger.debug(f"HTTP TTS request to {url}, model={self.model}, voice={voice}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"HTTP TTS failed (status {response.status}): {error_text}"
                    )
                
                # Stream response to file
                with open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        
        logger.debug(f"HTTP TTS saved to {output_path}")
        return output_path


# Google Cloud TTS voice mapping
GOOGLE_TTS_VOICES: dict[str, str] = {
    "en": "en-US-Neural2-D",      # English (US) - Male
    "zh": "cmn-TW-Wavenet-A",     # Chinese (Taiwan) - Female
}


def text_to_ssml_with_marks(text: str) -> tuple[str, list[str]]:
    """
    Convert plain text to SSML with sentence marks for timing.
    
    Splits text into sentences using BOTH paragraph breaks AND punctuation
    to ensure accurate timepoints for each text segment.
    
    Args:
        text: Plain text to convert
    
    Returns:
        Tuple of (SSML string, list of sentence texts)
    """
    import re
    
    # Step 1: Split on paragraph breaks (double newlines or single newlines)
    # This ensures lines without punctuation become separate segments
    paragraphs = re.split(r'\n\s*\n|\n', text)
    
    # Step 2: For each paragraph, split on sentence-ending punctuation
    sentences = []
    sentence_pattern = r'([^.!?。！？]+[.!?。！？])'
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Preserve ellipsis - replace temporarily
        para_normalized = para.replace('...', '\x00ELLIPSIS\x00')
        para_normalized = para_normalized.replace('…', '\x00ELLIPSIS\x00')
        
        # Try to split on punctuation
        para_sentences = re.findall(sentence_pattern, para_normalized)
        
        if para_sentences:
            # Restore ellipsis and add sentences
            for s in para_sentences:
                restored = s.replace('\x00ELLIPSIS\x00', '...').strip()
                if restored:
                    sentences.append(restored)
        else:
            # Paragraph has no sentence-ending punctuation
            # Treat the whole paragraph as one sentence
            restored = para_normalized.replace('\x00ELLIPSIS\x00', '...').strip()
            if restored:
                sentences.append(restored)
    
    # Step 3: Build SSML with marks BEFORE each sentence
    # This gives us the exact START time of each sentence for subtitle sync
    ssml_parts = ['<speak>']
    for i, sentence in enumerate(sentences):
        # Escape special XML characters
        escaped = (sentence
                   .replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;'))
        # Mark BEFORE sentence = timepoint when sentence STARTS
        ssml_parts.append(f'<mark name="s{i}"/>{escaped}')
    ssml_parts.append('</speak>')
    
    return ''.join(ssml_parts), sentences


def generate_srt_from_timepoints(
    sentences: list[str],
    timepoints: list[dict],
    total_duration: float,
) -> str:
    """
    Generate SRT content from sentences and their timepoints.
    
    Args:
        sentences: List of sentence texts
        timepoints: List of {"markName": "s0", "timeSeconds": 3.5} from Google TTS
        total_duration: Total audio duration in seconds
    
    Returns:
        SRT formatted string
    """
    # Create timing map from timepoints
    timing_map = {}
    for tp in timepoints:
        mark_name = tp.get("markName", "")
        time_seconds = tp.get("timeSeconds", 0)
        timing_map[mark_name] = time_seconds
    
    srt_blocks = []
    
    # Minimum display duration for subtitles (prevents flash-subtitle effect)
    # BUT never extend beyond the next sentence's actual start time!
    MIN_SUBTITLE_DURATION = 1.5  # seconds
    
    for i, sentence in enumerate(sentences):
        # Mark is placed BEFORE sentence, so s{i} = START time of sentence i
        start_time = timing_map.get(f"s{i}", 0.0)
        
        # Get the ACTUAL next sentence start time (not modified)
        next_sentence_start = None
        if i + 1 < len(sentences):
            next_sentence_start = timing_map.get(f"s{i+1}", total_duration)
            end_time = next_sentence_start
        else:
            end_time = total_duration
        
        # Only apply minimum duration if it doesn't overlap with next sentence
        # This prevents the SRT repair function from needing to shift timestamps
        if end_time - start_time < MIN_SUBTITLE_DURATION:
            desired_end = start_time + MIN_SUBTITLE_DURATION
            # Don't extend beyond next sentence start or total duration
            max_end = next_sentence_start if next_sentence_start else total_duration
            end_time = min(desired_end, max_end)
        
        # Format timestamps
        def to_srt_time(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int((s % 1) * 1000)
            return f'{h:02d}:{m:02d}:{sec:02d},{ms:03d}'
        
        # Split long sentences into display lines (max 32 chars per line to avoid overflow)
        MAX_LINE_LENGTH = 32
        words = sentence.split()
        display_lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > MAX_LINE_LENGTH:
                if len(current_line) > 1:
                    display_lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
                else:
                    display_lines.append(' '.join(current_line))
                    current_line = []
        if current_line:
            display_lines.append(' '.join(current_line))
        
        subtitle_text = '\n'.join(display_lines[:3])
        
        srt_block = f"{i+1}\n{to_srt_time(start_time)} --> {to_srt_time(end_time)}\n{subtitle_text}\n"
        srt_blocks.append(srt_block)
    
    return '\n'.join(srt_blocks)


class GoogleTTSProvider(TTSProvider):
    """
    Google Cloud Text-to-Speech provider.
    
    Uses Google Cloud TTS REST API with Service Account authentication.
    
    Environment variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file
    - GOOGLE_ACCESS_TOKEN: Direct OAuth2 access token (alternative)
    
    Pricing: https://cloud.google.com/text-to-speech/pricing
    - Neural2 voices: $16 per 1M characters
    - Wavenet voices: $16 per 1M characters  
    - Standard voices: $4 per 1M characters
    """
    
    def __init__(self):
        # Use v1beta1 for enableTimePointing feature
        self.base_url = "https://texttospeech.googleapis.com/v1beta1"
        self.service_account_file = None
        self.access_token = None
        
        # Try service account JSON file first
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and Path(creds_path).exists():
            self.service_account_file = Path(creds_path)
            logger.debug(f"Using service account file: {self.service_account_file}")
        else:
            # Look for JSON file in current directory
            json_files = list(Path(".").glob("*-*.json"))
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)
                        if data.get("type") == "service_account":
                            self.service_account_file = jf
                            logger.debug(f"Found service account file: {jf}")
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Fall back to direct access token
        if not self.service_account_file:
            self.access_token = os.getenv("GOOGLE_ACCESS_TOKEN")
        
        if not self.service_account_file and not self.access_token:
            raise ValueError(
                "Google TTS provider requires either:\n"
                "1. Service account JSON file (set GOOGLE_APPLICATION_CREDENTIALS or place in current dir)\n"
                "2. OAuth2 access token (set GOOGLE_ACCESS_TOKEN)\n"
                "Or use edge-tts provider (free, no API key): --provider edge"
            )
    
    async def _get_access_token(self) -> str:
        """Get OAuth2 access token from service account credentials."""
        if self.access_token:
            return self.access_token
        
        if not self.service_account_file:
            raise ValueError("No service account file configured")
        
        import time
        import jwt
        import aiohttp
        
        # Load service account credentials
        with open(self.service_account_file, "r") as f:
            creds = json.load(f)
        
        # Create JWT for OAuth2 token exchange
        now = int(time.time())
        payload = {
            "iss": creds["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": creds["token_uri"],
            "iat": now,
            "exp": now + 3600,  # 1 hour expiry
        }
        
        # Sign JWT with private key
        token = jwt.encode(payload, creds["private_key"], algorithm="RS256")
        
        # Exchange JWT for access token
        token_url = creds["token_uri"]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_url,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to get access token: {error_text}")
                
                result = await response.json()
                return result["access_token"]
    
    @property
    def name(self) -> str:
        return "google-tts"
    
    def _get_google_voice(self, lang: str) -> tuple[str, str]:
        """
        Get Google Cloud TTS voice name and language code.
        
        Args:
            lang: Language code ("en" or "zh")
        
        Returns:
            Tuple of (voice_name, language_code)
        """
        voice_name = GOOGLE_TTS_VOICES.get(lang, GOOGLE_TTS_VOICES["en"])
        
        # Extract language code from voice name (e.g., "en-US-Neural2-D" -> "en-US")
        parts = voice_name.split("-")
        if len(parts) >= 2:
            language_code = f"{parts[0]}-{parts[1]}"
        else:
            language_code = "en-US"
        
        return voice_name, language_code
    
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Path:
        """
        Synthesize using Google Cloud TTS REST API.
        
        Note: rate and pitch are converted to Google's speakingRate and pitch format.
        """
        import aiohttp
        import base64
        
        # Detect language and get appropriate Google voice
        lang = detect_language(text)
        google_voice, language_code = self._get_google_voice(lang)
        
        logger.debug(f"Google TTS: lang={lang}, voice={google_voice}, language_code={language_code}")
        
        # Parse rate (e.g., "+10%" -> 1.1, "-5%" -> 0.95)
        speaking_rate = 1.0
        if rate and rate != "+0%":
            try:
                rate_pct = float(rate.replace("%", "").replace("+", ""))
                speaking_rate = 1.0 + (rate_pct / 100.0)
                speaking_rate = max(0.25, min(4.0, speaking_rate))  # Clamp to valid range
            except ValueError:
                pass
        
        # Parse pitch (e.g., "+5Hz" -> 5.0, "-2Hz" -> -2.0)
        pitch_value = 0.0
        if pitch and pitch != "+0Hz":
            try:
                pitch_value = float(pitch.replace("Hz", "").replace("+", ""))
                pitch_value = max(-20.0, min(20.0, pitch_value))  # Clamp to valid range
            except ValueError:
                pass
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get access token (from service account or direct token)
        access_token = await self._get_access_token()
        
        # Convert text to SSML with sentence marks for timing
        ssml, sentences = text_to_ssml_with_marks(text)
        logger.debug(f"SSML generated with {len(sentences)} sentences")
        
        url = f"{self.base_url}/text:synthesize"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {"ssml": ssml},
            "voice": {
                "languageCode": language_code,
                "name": google_voice,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
                "pitch": pitch_value,
            },
            "enableTimePointing": ["SSML_MARK"],
        }
        
        logger.debug(f"Google TTS request to {self.base_url}, voice={google_voice}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Google TTS failed (status {response.status}): {error_text}"
                    )
                
                # Google returns base64-encoded audio in JSON response
                result = await response.json()
                audio_content = result.get("audioContent", "")
                timepoints = result.get("timepoints", [])
                
                if not audio_content:
                    raise RuntimeError("Google TTS returned empty audio content")
                
                # Decode base64 and save to file
                audio_bytes = base64.b64decode(audio_content)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                
                logger.debug(f"Google TTS saved to {output_path}")
                logger.debug(f"Received {len(timepoints)} timepoints")
                
                # Generate SRT from timepoints if available
                if timepoints and sentences:
                    # Get audio duration
                    try:
                        audio_duration = get_audio_duration(output_path)
                    except Exception:
                        audio_duration = 30.0  # Default fallback
                    
                    srt_content = generate_srt_from_timepoints(
                        sentences, timepoints, audio_duration
                    )
                    
                    # Save SRT alongside audio
                    srt_path = output_path.parent / "captions.srt"
                    srt_path.write_text(srt_content, encoding="utf-8")
                    logger.info(f"Generated synced SRT: {srt_path}")
        
        return output_path


def get_tts_provider(provider: str = "edge") -> TTSProvider:
    """
    Get TTS provider instance.
    
    Args:
        provider: Provider name ("edge", "http", or "google")
    
    Returns:
        TTSProvider instance
    
    Raises:
        ValueError: If provider is unknown or misconfigured
    """
    provider = provider.lower().strip()
    
    if provider == "edge":
        return EdgeTTSProvider()
    elif provider == "http" or provider == "openai":
        return HttpTTSProvider()
    elif provider == "google" or provider == "gcloud":
        return GoogleTTSProvider()
    else:
        raise ValueError(
            f"Unknown TTS provider: {provider}\n"
            f"Supported providers: edge (default), http, google"
        )

# Typer CLI app
app = typer.Typer(
    name="tts",
    help="Generate TTS voice-over audio from voiceover scripts",
    add_completion=False,
)


# ============================================================================
# Windows asyncio compatibility
# ============================================================================


def _setup_asyncio_windows() -> None:
    """
    Set up asyncio event loop policy for Windows.
    
    Windows requires WindowsSelectorEventLoopPolicy for subprocess operations.
    """
    if platform.system() == "Windows":
        # Use WindowsSelectorEventLoopPolicy to avoid issues with subprocess
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_async(coro: Any) -> Any:
    """
    Run async coroutine with proper Windows handling.
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Result of coroutine
    """
    _setup_asyncio_windows()
    return asyncio.run(coro)


# ============================================================================
# TTS Core Functions (mockable)
# ============================================================================


async def _edge_tts_synthesize(
    text: str,
    output_path: Path,
    voice: str,
    rate: str = TTS_DEFAULT_RATE,
    pitch: str = TTS_DEFAULT_PITCH,
) -> Path:
    """
    Internal edge-tts synthesis function.
    
    This is the actual implementation that calls edge-tts.
    Separated for easier mocking in tests.
    
    Args:
        text: Text to synthesize
        output_path: Output MP3 path
        voice: Voice name
        rate: Speech rate (e.g., "+10%", "-5%")
        pitch: Pitch adjustment (e.g., "+0Hz")
    
    Returns:
        Path to generated MP3 file
    
    Raises:
        ImportError: If edge-tts is not installed
        RuntimeError: If synthesis fails
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is not installed. Install it with: pip install edge-tts"
        )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create communicate object
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    
    # Save to file
    await communicate.save(str(output_path))
    
    return output_path


def set_synthesize_function(fn: Optional[Callable]) -> None:
    """
    Set custom synthesize function (for testing).
    
    Args:
        fn: Custom async function with same signature as _edge_tts_synthesize
            Pass None to reset to default
    """
    global _synthesize_fn
    _synthesize_fn = fn


async def synthesize(
    text: str,
    output_path: Path,
    voice: str,
    rate: str = TTS_DEFAULT_RATE,
    pitch: str = TTS_DEFAULT_PITCH,
) -> Path:
    """
    Synthesize text to speech.
    
    This is the main entry point for TTS. It can be mocked via
    set_synthesize_function() for testing.
    
    Args:
        text: Text to synthesize
        output_path: Output MP3 path
        voice: Voice name
        rate: Speech rate (e.g., "+10%", "-5%")
        pitch: Pitch adjustment (e.g., "+0Hz")
    
    Returns:
        Path to generated MP3 file
    """
    global _synthesize_fn
    
    if _synthesize_fn is not None:
        return await _synthesize_fn(text, output_path, voice, rate, pitch)
    
    return await _edge_tts_synthesize(text, output_path, voice, rate, pitch)


# ============================================================================
# Audio Duration Detection
# ============================================================================


def get_audio_duration(audio_path: Path) -> float:
    """
    Get duration of audio file in seconds.
    
    Tries pydub first, falls back to ffprobe.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    
    Raises:
        RuntimeError: If duration cannot be determined
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Try pydub first
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0  # milliseconds to seconds
    except ImportError:
        logger.debug("pydub not available, trying ffprobe")
    except Exception as e:
        logger.debug(f"pydub failed: {e}, trying ffprobe")
    
    # Fall back to ffprobe
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError) as e:
        logger.debug(f"ffprobe failed: {e}")
    
    # Estimate based on file size and bitrate (fallback)
    # Typical MP3 at 128kbps: ~16KB per second
    file_size = audio_path.stat().st_size
    estimated = file_size / 16000  # Rough estimate
    logger.warning(f"Using estimated duration: {estimated:.1f}s")
    return estimated


# ============================================================================
# SRT Timing Adjustment
# ============================================================================


def parse_srt_timestamp(ts: str) -> float:
    """
    Parse SRT timestamp to seconds.
    
    Format: HH:MM:SS,mmm
    
    Args:
        ts: SRT timestamp string
    
    Returns:
        Seconds as float
    """
    # Handle both , and . as decimal separator
    ts = ts.replace(",", ".")
    
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    return hours * 3600 + minutes * 60 + seconds


def format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as SRT timestamp.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        SRT timestamp string (HH:MM:SS,mmm)
    """
    seconds = max(0, seconds)
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def adjust_srt_timing(
    srt_path: Path,
    target_duration: float,
    output_path: Optional[Path] = None,
    threshold: float = SRT_TIMING_ADJUSTMENT_THRESHOLD,
) -> bool:
    """
    Adjust SRT timestamps to match audio duration.
    
    Uses linear scaling if difference exceeds threshold.
    Ensures timestamps are always increasing.
    
    Args:
        srt_path: Path to SRT file
        target_duration: Target duration in seconds
        output_path: Output path (default: overwrite original)
        threshold: Adjustment threshold (0.15 = 15%)
    
    Returns:
        True if adjusted, False if no adjustment needed
    """
    if not srt_path.exists():
        logger.warning(f"SRT file not found: {srt_path}")
        return False
    
    if output_path is None:
        output_path = srt_path
    
    # Read SRT content
    content = srt_path.read_text(encoding="utf-8")
    
    # Parse SRT entries
    # Format: index\nstart --> end\ntext\n\n
    pattern = r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(.*?)(?=\n\n|\n*$)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        logger.warning(f"No valid SRT entries found in {srt_path}")
        return False
    
    # Find current duration (last end timestamp)
    last_end = 0.0
    entries = []
    
    for match in matches:
        index = int(match.group(1))
        start = parse_srt_timestamp(match.group(2))
        end = parse_srt_timestamp(match.group(3))
        text = match.group(4).strip()
        
        entries.append({
            "index": index,
            "start": start,
            "end": end,
            "text": text,
        })
        
        last_end = max(last_end, end)
    
    current_duration = last_end
    
    # Check if adjustment needed
    if current_duration == 0:
        logger.warning("SRT has zero duration, cannot adjust")
        return False
    
    difference = abs(target_duration - current_duration) / current_duration
    
    if difference <= threshold:
        logger.debug(f"SRT timing difference {difference:.1%} <= {threshold:.0%}, no adjustment needed")
        return False
    
    # Calculate scale factor
    scale_factor = target_duration / current_duration
    logger.info(f"Adjusting SRT timing: {current_duration:.1f}s -> {target_duration:.1f}s (scale: {scale_factor:.2f}x)")
    
    # Scale timestamps and ensure monotonically increasing
    prev_end = 0.0
    adjusted_entries = []
    
    for entry in entries:
        new_start = entry["start"] * scale_factor
        new_end = entry["end"] * scale_factor
        
        # Ensure start >= previous end (monotonically increasing)
        if new_start < prev_end:
            new_start = prev_end + 0.001
        
        # Ensure end > start
        if new_end <= new_start:
            new_end = new_start + 0.5
        
        adjusted_entries.append({
            "index": entry["index"],
            "start": new_start,
            "end": new_end,
            "text": entry["text"],
        })
        
        prev_end = new_end
    
    # Write adjusted SRT
    lines = []
    for entry in adjusted_entries:
        lines.append(str(entry["index"]))
        lines.append(f"{format_srt_timestamp(entry['start'])} --> {format_srt_timestamp(entry['end'])}")
        lines.append(entry["text"])
        lines.append("")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Adjusted SRT saved to {output_path}")
    
    return True


# ============================================================================
# Batch Processing
# ============================================================================


async def generate_voiceover_for_folder(
    folder_path: Path,
    voice: Optional[str] = None,
    rate: str = TTS_DEFAULT_RATE,
    pitch: str = TTS_DEFAULT_PITCH,
    adjust_captions: bool = True,
    provider: str = "edge",
) -> Optional[Path]:
    """
    Generate voiceover MP3 for a shorts folder.
    
    Args:
        folder_path: Path to shorts folder (e.g., outputs/shorts/42)
        voice: Voice name (auto-detected from language if None)
        rate: Speech rate
        pitch: Pitch adjustment
        adjust_captions: Whether to adjust SRT timing after generation
        provider: TTS provider ("edge" or "http")
    
    Returns:
        Path to generated MP3, or None if failed
    """
    voiceover_txt = folder_path / "voiceover.txt"
    voiceover_mp3 = folder_path / "voiceover.mp3"
    captions_srt = folder_path / "captions.srt"
    
    if not voiceover_txt.exists():
        logger.warning(f"voiceover.txt not found in {folder_path}")
        return None
    
    # Read voiceover text - ENSURE FULL TEXT IS READ
    text = voiceover_txt.read_text(encoding="utf-8").strip()
    
    # Clean up whitespace but PRESERVE NEWLINES for paragraph detection
    # Only collapse multiple spaces on the same line, not newlines
    text = re.sub(r'[^\S\n]+', ' ', text)  # Collapse spaces but keep newlines
    text = re.sub(r' *\n *', '\n', text)   # Clean whitespace around newlines
    text = text.replace(' .', '.').replace(' ,', ',')  # Fix punctuation
    
    if not text:
        logger.warning(f"voiceover.txt is empty in {folder_path}")
        return None
    
    logger.debug(f"Voiceover text length: {len(text)} chars")
    
    # Get TTS provider
    try:
        tts_provider = get_tts_provider(provider)
        logger.info(f"Using TTS provider: {tts_provider.name}")
    except ValueError as e:
        logger.error(str(e))
        raise
    
    # Auto-detect language and select voice
    if voice is None:
        lang = detect_language(text)
        voice = get_tts_voice(lang)
        logger.debug(f"Auto-detected language: {lang}, using voice: {voice}")
    
    # Generate TTS using provider
    try:
        await tts_provider.synthesize(
            text=text,
            output_path=voiceover_mp3,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        logger.info(f"Generated {voiceover_mp3}")
    except ImportError as e:
        error_msg = str(e)
        console.print(f"[red]Error: {error_msg}[/red]")
        if "edge" in provider.lower():
            console.print("[yellow]To install edge-tts, run: pip install edge-tts[/yellow]")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"TTS failed for {folder_path}: {error_msg}")
        
        # Provide helpful error messages
        if "SSL" in error_msg or "certificate" in error_msg.lower():
            logger.info("SSL error detected. Try: export MYTH_MUSEUM_SKIP_SSL=1")
            logger.info("Or use HTTP provider: --provider http (requires OPENAI_API_KEY)")
        
        raise
    
    # Adjust captions if requested
    if adjust_captions and captions_srt.exists() and voiceover_mp3.exists():
        try:
            audio_duration = get_audio_duration(voiceover_mp3)
            adjust_srt_timing(captions_srt, audio_duration)
        except Exception as e:
            logger.warning(f"Failed to adjust captions: {e}")
    
    return voiceover_mp3


async def batch_generate(
    shorts_dir: Path,
    limit: int = 100,
    voice: Optional[str] = None,
    rate: str = TTS_DEFAULT_RATE,
    pitch: str = TTS_DEFAULT_PITCH,
    adjust_captions: bool = True,
    skip_existing: bool = True,
    provider: str = "edge",
) -> dict[str, Any]:
    """
    Generate voiceovers for multiple shorts folders.
    
    Args:
        shorts_dir: Base directory containing shorts folders
        limit: Maximum folders to process
        voice: Voice name (auto-detected if None)
        rate: Speech rate
        pitch: Pitch adjustment
        adjust_captions: Whether to adjust SRT timing
        skip_existing: Skip folders that already have voiceover.mp3
        provider: TTS provider ("edge" or "http")
    
    Returns:
        Dict with stats: processed, skipped, failed, paths
    """
    if not shorts_dir.exists():
        logger.warning(f"Shorts directory not found: {shorts_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0, "paths": []}
    
    # Find all shorts folders
    folders = sorted([
        d for d in shorts_dir.iterdir()
        if d.is_dir() and (d / "voiceover.txt").exists()
    ])[:limit]
    
    processed = 0
    skipped = 0
    failed = 0
    paths = []
    
    for folder in folders:
        mp3_path = folder / "voiceover.mp3"
        
        if skip_existing and mp3_path.exists():
            logger.debug(f"Skipping {folder.name}: voiceover.mp3 already exists")
            skipped += 1
            continue
        
        try:
            result = await generate_voiceover_for_folder(
                folder_path=folder,
                voice=voice,
                rate=rate,
                pitch=pitch,
                adjust_captions=adjust_captions,
                provider=provider,
            )
            
            if result:
                processed += 1
                paths.append(result)
            else:
                failed += 1
        except Exception as e:
            logger.error(f"TTS failed for {folder.name}: {e}")
            failed += 1
    
    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "paths": paths,
    }


# ============================================================================
# Voice Listing
# ============================================================================


async def list_voices_async(language: Optional[str] = None) -> list[dict[str, str]]:
    """
    List available edge-tts voices.
    
    Args:
        language: Filter by language code (e.g., "en", "zh")
    
    Returns:
        List of voice info dicts
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is not installed. Install it with: pip install edge-tts"
        )
    
    voices = await edge_tts.list_voices()
    
    result = []
    for voice in voices:
        voice_lang = voice.get("Locale", "").split("-")[0].lower()
        
        if language and not voice_lang.startswith(language.lower()):
            continue
        
        result.append({
            "name": voice.get("ShortName", ""),
            "locale": voice.get("Locale", ""),
            "gender": voice.get("Gender", ""),
        })
    
    return result


# ============================================================================
# CLI Commands
# ============================================================================


@app.command("batch")
def batch_cmd(
    in_dir: str = typer.Option(
        "outputs/shorts",
        "--in-dir",
        "-i",
        help="Input directory containing shorts folders",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum folders to process",
    ),
    voice: Optional[str] = typer.Option(
        None,
        "--voice",
        "-v",
        help="TTS voice name (auto-detected if not specified)",
    ),
    rate: str = typer.Option(
        TTS_DEFAULT_RATE,
        "--rate",
        "-r",
        help="Speech rate (e.g., '+10%%', '-5%%')",
    ),
    pitch: str = typer.Option(
        TTS_DEFAULT_PITCH,
        "--pitch",
        "-p",
        help="Pitch adjustment (e.g., '+0Hz')",
    ),
    provider: str = typer.Option(
        "edge",
        "--provider",
        help="TTS provider: 'edge' (free) or 'http' (OpenAI-compatible, requires OPENAI_API_KEY)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--overwrite",
        help="Skip folders with existing voiceover.mp3",
    ),
    no_adjust: bool = typer.Option(
        False,
        "--no-adjust",
        help="Skip SRT timing adjustment",
    ),
) -> None:
    """
    Generate voiceover audio for multiple shorts folders.
    
    Reads voiceover.txt from each folder and generates voiceover.mp3.
    Optionally adjusts captions.srt timing to match audio duration.
    
    TTS Providers:
    - edge: Microsoft Edge TTS (free, no API key, default)
    - http: OpenAI-compatible API (requires OPENAI_API_KEY env var)
    """
    shorts_dir = Path(in_dir)
    
    console.print("[bold cyan]=== TTS Batch Generation ===[/bold cyan]\n")
    console.print(f"Input directory: {shorts_dir}")
    console.print(f"Limit: {limit}")
    console.print(f"Voice: {voice or 'auto-detect'}")
    console.print(f"Provider: {provider}")
    console.print(f"Rate: {rate}")
    console.print(f"Skip existing: {skip_existing}")
    console.print(f"Adjust captions: {not no_adjust}")
    console.print("")
    
    if not shorts_dir.exists():
        console.print(f"[red]Directory not found: {shorts_dir}[/red]")
        raise typer.Exit(1)
    
    # Run batch generation
    try:
        result = run_async(batch_generate(
            shorts_dir=shorts_dir,
            limit=limit,
            voice=voice,
            rate=rate,
            pitch=pitch,
            adjust_captions=not no_adjust,
            skip_existing=skip_existing,
            provider=provider,
        ))
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]To install edge-tts, run:[/yellow]")
        console.print("  pip install edge-tts")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Print results
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Processed: {result['processed']}")
    console.print(f"  Skipped: {result['skipped']}")
    console.print(f"  Failed: {result['failed']}")
    
    if result["processed"] > 0:
        console.print(f"\n[green]OK[/green] Generated {result['processed']} voiceover files")
    else:
        console.print("\n[yellow]No new voiceovers generated[/yellow]")


@app.command("single")
def single_cmd(
    claim_id: int = typer.Argument(..., help="Claim ID of the shorts folder"),
    shorts_dir: str = typer.Option(
        "outputs/shorts",
        "--dir",
        "-d",
        help="Base shorts directory",
    ),
    voice: Optional[str] = typer.Option(
        None,
        "--voice",
        "-v",
        help="TTS voice name",
    ),
    rate: str = typer.Option(
        TTS_DEFAULT_RATE,
        "--rate",
        "-r",
        help="Speech rate",
    ),
    pitch: str = typer.Option(
        TTS_DEFAULT_PITCH,
        "--pitch",
        "-p",
        help="Pitch adjustment",
    ),
    provider: str = typer.Option(
        "edge",
        "--provider",
        help="TTS provider: 'edge' (free) or 'http' (OpenAI-compatible)",
    ),
) -> None:
    """
    Generate voiceover for a single shorts folder.
    
    TTS Providers:
    - edge: Microsoft Edge TTS (free, no API key, default)
    - http: OpenAI-compatible API (requires OPENAI_API_KEY env var)
    """
    folder = Path(shorts_dir) / str(claim_id)
    
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]=== TTS Single Generation ===[/bold cyan]\n")
    console.print(f"Folder: {folder}")
    console.print(f"Provider: {provider}")
    
    try:
        result = run_async(generate_voiceover_for_folder(
            folder_path=folder,
            voice=voice,
            rate=rate,
            pitch=pitch,
            adjust_captions=True,
            provider=provider,
        ))
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]To install edge-tts, run:[/yellow]")
        console.print("  pip install edge-tts")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    if result:
        console.print(f"\n[green]OK[/green] Generated: {result}")
    else:
        console.print(f"\n[red]Failed to generate voiceover[/red]")
        raise typer.Exit(1)


@app.command("list-voices")
def list_voices_cmd(
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Filter by language code (e.g., 'en', 'zh')",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        help="Maximum voices to show",
    ),
) -> None:
    """
    List available edge-tts voices.
    """
    console.print("[bold cyan]=== Available TTS Voices ===[/bold cyan]\n")
    
    try:
        voices = run_async(list_voices_async(language=language))
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]To install edge-tts, run:[/yellow]")
        console.print("  pip install edge-tts")
        raise typer.Exit(1)
    
    from rich.table import Table
    table = Table()
    table.add_column("Name")
    table.add_column("Locale")
    table.add_column("Gender")
    
    for voice in voices[:limit]:
        table.add_row(voice["name"], voice["locale"], voice["gender"])
    
    console.print(table)
    console.print(f"\nShowing {min(limit, len(voices))} of {len(voices)} voices")
    
    if not language:
        console.print("\n[dim]Tip: Use --language en or --language zh to filter[/dim]")


@app.command("adjust-srt")
def adjust_srt_cmd(
    srt_path: str = typer.Argument(..., help="Path to SRT file"),
    duration: float = typer.Argument(..., help="Target duration in seconds"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (default: overwrite original)",
    ),
) -> None:
    """
    Adjust SRT timestamps to match target duration.
    """
    srt_file = Path(srt_path)
    output_file = Path(output) if output else None
    
    if not srt_file.exists():
        console.print(f"[red]SRT file not found: {srt_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Adjusting {srt_file} to {duration:.1f}s...")
    
    adjusted = adjust_srt_timing(srt_file, duration, output_file)
    
    if adjusted:
        console.print("[green]OK[/green] SRT timing adjusted")
    else:
        console.print("[yellow]No adjustment needed[/yellow]")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
