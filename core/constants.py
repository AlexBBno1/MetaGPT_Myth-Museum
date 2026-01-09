"""
Myth Museum - Constants

Enums, mappings, and constant values used throughout the pipeline.
"""

from enum import Enum


class TopicEnum(str, Enum):
    """Topic categories for claims."""
    
    HEALTH = "health"
    HISTORY = "history"
    LANGUAGE = "language"
    PSYCHOLOGY = "psychology"
    SCIENCE = "science"
    INTERNET_RUMOR = "internet_rumor"
    UNKNOWN = "unknown"


class VerdictEnum(str, Enum):
    """Verdict categories for fact-checking."""
    
    FALSE = "False"
    MISLEADING = "Misleading"
    DEPENDS = "Depends"
    TRUE = "True"
    UNVERIFIED = "Unverified"


class ClaimStatusEnum(str, Enum):
    """Status of a claim in the pipeline."""
    
    NEW = "new"
    EXTRACTING = "extracting"
    HAS_EVIDENCE = "has_evidence"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    JUDGED = "judged"
    PACKAGED = "packaged"
    FAILED = "failed"


class SourceTypeEnum(str, Enum):
    """Types of data sources."""
    
    RSS = "rss"
    WIKIPEDIA = "wikipedia"
    FACTCHECK = "factcheck"
    ACADEMIC = "academic"
    OFFICIAL = "official"
    NEWS = "news"


# Credibility scores by source type (0-100)
CREDIBILITY_MAP: dict[str, int] = {
    SourceTypeEnum.ACADEMIC.value: 90,
    SourceTypeEnum.OFFICIAL.value: 85,
    SourceTypeEnum.FACTCHECK.value: 80,
    SourceTypeEnum.WIKIPEDIA.value: 70,
    SourceTypeEnum.NEWS.value: 50,
    SourceTypeEnum.RSS.value: 40,
}


# Topic keywords for classification
TOPIC_KEYWORDS: dict[TopicEnum, list[str]] = {
    TopicEnum.HEALTH: [
        "health", "medical", "medicine", "doctor", "病", "醫", "健康",
        "vaccine", "疫苗", "drug", "藥", "treatment", "治療", "cure",
        "disease", "疾病", "symptom", "症狀", "cancer", "癌", "diet",
        "nutrition", "營養", "exercise", "運動", "sleep", "睡眠",
        "virus", "病毒", "bacteria", "細菌", "immune", "免疫",
    ],
    TopicEnum.HISTORY: [
        "history", "historical", "ancient", "歷史", "古代", "war", "戰爭",
        "emperor", "皇帝", "dynasty", "朝代", "civilization", "文明",
        "archaeology", "考古", "artifact", "文物", "century", "世紀",
    ],
    TopicEnum.LANGUAGE: [
        "language", "語言", "word", "詞", "etymology", "語源",
        "grammar", "語法", "dialect", "方言", "pronunciation", "發音",
        "writing", "文字", "alphabet", "字母",
    ],
    TopicEnum.PSYCHOLOGY: [
        "psychology", "心理", "mental", "精神", "brain", "腦",
        "behavior", "行為", "emotion", "情緒", "memory", "記憶",
        "cognitive", "認知", "therapy", "治療", "anxiety", "焦慮",
        "depression", "憂鬱", "stress", "壓力",
    ],
    TopicEnum.SCIENCE: [
        "science", "科學", "physics", "物理", "chemistry", "化學",
        "biology", "生物", "astronomy", "天文", "earth", "地球",
        "experiment", "實驗", "research", "研究", "discovery", "發現",
        "evolution", "演化", "climate", "氣候", "space", "太空",
    ],
    TopicEnum.INTERNET_RUMOR: [
        "viral", "rumor", "謠言", "fake", "假", "hoax", "騙局",
        "conspiracy", "陰謀", "scam", "詐騙", "urban legend", "都市傳說",
        "forwarded", "轉發", "share", "分享",
    ],
}


# Health and legal disclaimer text
HEALTH_LEGAL_DISCLAIMER: str = """
⚠️ 免責聲明 / Disclaimer:
本內容僅供教育與資訊參考，不構成醫療、法律或專業建議。
如有健康疑慮，請諮詢合格醫療專業人員。
如需法律協助，請諮詢持牌律師。

This content is for educational and informational purposes only and does not 
constitute medical, legal, or professional advice. For health concerns, please 
consult a qualified healthcare professional. For legal matters, please consult 
a licensed attorney.
"""


# Topics that require disclaimer
DISCLAIMER_TOPICS: set[TopicEnum] = {
    TopicEnum.HEALTH,
}


# Default score thresholds
DEFAULT_MIN_SCORE: int = 50
DEFAULT_SIMILARITY_THRESHOLD: float = 85.0
DEFAULT_MIN_EVIDENCE_SOURCES: int = 2


# Rate limiting
DEFAULT_RATE_LIMIT_SECONDS: float = 1.0
DEFAULT_MAX_RETRIES: int = 3


# Claim extraction limits
MAX_CLAIMS_PER_ITEM: int = 3
MAX_CLAIM_LENGTH: int = 140


# Script generation parameters
SHORTS_SEGMENT_SECONDS: tuple[int, int] = (5, 8)
SHORTS_TOTAL_SECONDS: tuple[int, int] = (30, 60)
LONG_OUTLINE_CHAPTERS: tuple[int, int] = (6, 10)
LONG_VIDEO_MINUTES: tuple[int, int] = (6, 10)


# ============================================================================
# Shorts Export Constants
# ============================================================================


# Voiceover limits for Shorts (target duration and word/character counts)
class VoiceoverLimits:
    """Voiceover length limits for YouTube Shorts."""
    
    TARGET_SECONDS: tuple[int, int] = (30, 45)
    ZH_CHARS: tuple[int, int] = (120, 180)  # Chinese characters
    EN_WORDS: tuple[int, int] = (80, 120)   # English words


# On-screen text limits (must be short for mobile viewing)
class OnScreenTextLimits:
    """On-screen text length limits for readability."""
    
    ZH_MAX_CHARS: int = 12
    EN_MAX_WORDS: int = 6


# Shotlist CSV columns (fixed order)
SHOTLIST_CSV_COLUMNS: list[str] = [
    "time_start",
    "time_end",
    "scene",
    "voice_line",
    "on_screen_text",
    "visual_suggestion",
    "sfx",
]


# Short disclaimers for voiceovers (single line, non-intrusive)
SHORT_DISCLAIMERS: dict[str, dict[str, str]] = {
    "health": {
        "en": "(Not medical advice - consult a healthcare professional.)",
        "zh": "（非醫療建議，請諮詢專業醫師）",
    },
    "law": {
        "en": "(Not legal advice - consult a licensed attorney.)",
        "zh": "（非法律建議，請諮詢專業律師）",
    },
    "general": {
        "en": "(For informational purposes only.)",
        "zh": "（僅供參考）",
    },
}


# Topics that require short disclaimer in voiceover
DISCLAIMER_REQUIRED_TOPICS: set[str] = {"health", "law"}


# Default hashtags by topic (safe, no medical claims or exaggerations)
DEFAULT_HASHTAGS_BY_TOPIC: dict[str, list[str]] = {
    "health": [
        "#health", "#factcheck", "#wellness", "#science", "#mythbusted",
        "#debunked", "#facts", "#education", "#learning", "#didyouknow",
    ],
    "history": [
        "#history", "#factcheck", "#historical", "#education", "#mythbusted",
        "#debunked", "#facts", "#learning", "#didyouknow", "#ancient",
    ],
    "science": [
        "#science", "#factcheck", "#physics", "#education", "#mythbusted",
        "#debunked", "#facts", "#research", "#learning", "#didyouknow",
    ],
    "psychology": [
        "#psychology", "#factcheck", "#mind", "#education", "#mythbusted",
        "#debunked", "#facts", "#brain", "#learning", "#mentalhealth",
    ],
    "language": [
        "#language", "#factcheck", "#linguistics", "#education", "#mythbusted",
        "#debunked", "#facts", "#words", "#learning", "#didyouknow",
    ],
    "internet_rumor": [
        "#viral", "#factcheck", "#fakenews", "#education", "#mythbusted",
        "#debunked", "#facts", "#truth", "#learning", "#didyouknow",
    ],
    "unknown": [
        "#factcheck", "#mythbusted", "#debunked", "#facts", "#education",
        "#learning", "#truth", "#viral", "#didyouknow", "#shorts",
    ],
}


# Generic hashtags to add to all topics
GENERIC_HASHTAGS: list[str] = ["#shorts", "#fyp"]


# ============================================================================
# Shorts Export Helper Functions
# ============================================================================


def pick_hashtags(
    topic: str,
    extra: list[str] | None = None,
    count: int = 10,
) -> list[str]:
    """
    Pick hashtags for a given topic.
    
    Args:
        topic: The topic category (health, science, etc.)
        extra: Additional hashtags to include (prioritized)
        count: Target number of hashtags (8-12 recommended)
    
    Returns:
        List of hashtags (deduplicated, capped at count)
    """
    topic_lower = topic.lower() if topic else "unknown"
    
    hashtags = []
    
    # Add extra hashtags first (prioritized)
    if extra:
        hashtags.extend(extra)
    
    # Get topic-specific hashtags
    topic_tags = DEFAULT_HASHTAGS_BY_TOPIC.get(
        topic_lower, 
        DEFAULT_HASHTAGS_BY_TOPIC["unknown"]
    )
    hashtags.extend(topic_tags)
    
    # Add generic hashtags
    hashtags.extend(GENERIC_HASHTAGS)
    
    # Deduplicate while preserving order
    seen = set()
    unique_hashtags = []
    for tag in hashtags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_hashtags.append(tag)
    
    return unique_hashtags[:count]


def needs_disclaimer(topic: str) -> bool:
    """
    Check if a topic requires a disclaimer in voiceover.
    
    Args:
        topic: The topic category
    
    Returns:
        True if disclaimer is required
    """
    if not topic:
        return False
    return topic.lower() in DISCLAIMER_REQUIRED_TOPICS


def get_disclaimer(topic: str, language: str = "en") -> str | None:
    """
    Get the short disclaimer for a topic.
    
    Args:
        topic: The topic category
        language: Language code ('en' or 'zh')
    
    Returns:
        Disclaimer string or None if not required
    """
    if not needs_disclaimer(topic):
        return None
    
    topic_lower = topic.lower()
    lang = "zh" if language.lower().startswith("zh") else "en"
    
    if topic_lower in SHORT_DISCLAIMERS:
        return SHORT_DISCLAIMERS[topic_lower].get(lang)
    
    return SHORT_DISCLAIMERS["general"].get(lang)


def truncate_onscreen_text(text: str, language: str = "en") -> str:
    """
    Truncate on-screen text to fit display limits.
    
    Args:
        text: Original text
        language: Language code ('en' or 'zh')
    
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Detect language if auto
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if is_chinese or language.lower().startswith("zh"):
        # Chinese: limit by characters
        if len(text) > OnScreenTextLimits.ZH_MAX_CHARS:
            return text[:OnScreenTextLimits.ZH_MAX_CHARS - 1] + "…"
        return text
    else:
        # English: limit by words
        words = text.split()
        if len(words) > OnScreenTextLimits.EN_MAX_WORDS:
            return " ".join(words[:OnScreenTextLimits.EN_MAX_WORDS]) + "..."
        return text


# ============================================================================
# Shorts Queue Selection Constants
# ============================================================================


# Verdict weights for queue scoring (higher = more likely to be selected)
VERDICT_WEIGHTS: dict[str, int] = {
    "False": 10,
    "Misleading": 8,
    "Depends": 5,
    "True": 3,
    "Unverified": 0,  # Excluded by default
}


# Safety gates for sensitive topics (require higher confidence/evidence)
QUEUE_SAFETY_GATES: dict[str, dict[str, float | int]] = {
    "health": {"min_confidence": 0.7, "min_evidence_types": 2},
    "law": {"min_confidence": 0.7, "min_evidence_types": 2},
}


# Topics to ensure diversity in queue (at least 1 from each if available)
QUEUE_TOPIC_MIX_TARGETS: list[str] = ["health", "history", "science", "unknown"]


# Queue CSV columns (fixed order)
QUEUE_CSV_COLUMNS: list[str] = [
    "rank",
    "claim_id",
    "topic",
    "verdict",
    "confidence",
    "title",
    "hook",
    "estimated_seconds",
    "folder_path",
    "status",
]


# Queue selection defaults
QUEUE_DEFAULT_MIN_CONFIDENCE: float = 0.5
QUEUE_DEFAULT_SIMILARITY_THRESHOLD: float = 85.0


# ============================================================================
# TTS Voice-over Constants
# ============================================================================


# Default TTS voices by language (edge-tts voice names)
TTS_DEFAULT_VOICES: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "zh": "zh-CN-YunxiNeural",
}


# TTS default parameters
TTS_DEFAULT_RATE: str = "+0%"
TTS_DEFAULT_PITCH: str = "+0Hz"


# SRT timing adjustment threshold (scale if difference > this percentage)
SRT_TIMING_ADJUSTMENT_THRESHOLD: float = 0.15  # 15%


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English.
    
    Args:
        text: Text to analyze
    
    Returns:
        "zh" for Chinese, "en" for English
    """
    if not text:
        return "en"
    
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text.replace(" ", ""))
    
    if total_chars == 0:
        return "en"
    
    # If more than 30% Chinese characters, consider it Chinese
    if chinese_chars / total_chars > 0.3:
        return "zh"
    
    return "en"


def get_tts_voice(language: str, custom_voice: str | None = None) -> str:
    """
    Get TTS voice for a language.
    
    Args:
        language: Language code ("en" or "zh")
        custom_voice: Override voice name
    
    Returns:
        Voice name for edge-tts
    """
    if custom_voice:
        return custom_voice
    
    lang = "zh" if language.lower().startswith("zh") else "en"
    return TTS_DEFAULT_VOICES.get(lang, TTS_DEFAULT_VOICES["en"])


# ============================================================================
# Shorts Production Pipeline Constants
# ============================================================================


class ShortsStatus(str, Enum):
    """Status of a shorts folder in the production pipeline."""
    
    NEEDS_EXPORT = "needs_export"           # Folder missing or < 6 required files
    NEEDS_TTS = "needs_tts"                 # Has 6 files but missing voiceover.mp3
    TOO_SHORT_VOICEOVER = "too_short_voiceover"  # voiceover.txt below minimum length
    TOO_SHORT_AUDIO = "too_short_audio"     # voiceover.mp3 duration < 20s
    TTS_FAILED = "tts_failed"               # TTS provider error
    RENDER_FAILED = "render_failed"         # FFmpeg render failed
    READY = "ready"                         # Has all files, ready to render
    RENDERED = "rendered"                   # Has final.mp4


# Required files for a shorts folder (before TTS)
SHORTS_REQUIRED_FILES: list[str] = [
    "voiceover.txt",
    "shotlist.csv",
    "captions.srt",
    "metadata.json",
    "sources.md",
    "assets_needed.md",
]


# ============================================================================
# Voiceover Length Limits
# ============================================================================

# Minimum voiceover text length (below this = too_short_voiceover)
VOICEOVER_MIN_ZH_CHARS: int = 120      # Chinese characters
VOICEOVER_MIN_EN_WORDS: int = 80       # English words

# Target voiceover text length for 30-45 second videos
VOICEOVER_TARGET_ZH_CHARS: tuple[int, int] = (140, 220)  # Chinese: 140-220 chars
VOICEOVER_TARGET_EN_WORDS: tuple[int, int] = (90, 140)   # English: 90-140 words

# Minimum audio duration (below this = too_short_audio)
VOICEOVER_MIN_DURATION_SECONDS: float = 20.0

# Target audio duration
VOICEOVER_TARGET_DURATION_SECONDS: tuple[int, int] = (30, 45)


# ============================================================================
# Video Rendering Specifications
# ============================================================================

RENDER_WIDTH: int = 1080
RENDER_HEIGHT: int = 1920
RENDER_FPS: int = 30
RENDER_VIDEO_CODEC: str = "libx264"
RENDER_AUDIO_CODEC: str = "aac"
RENDER_DEFAULT_BG_COLOR: str = "black"


# FFmpeg subtitle style for burn-in
# FontSize=22 with margins to prevent overflow on 1080px width
RENDER_SUBTITLE_STYLE: str = (
    "FontSize=22,"
    "PrimaryColour=&HFFFFFF,"
    "OutlineColour=&H000000,"
    "Outline=2,"
    "Alignment=2,"
    "MarginV=80,"
    "MarginL=40,"
    "MarginR=40"
)


def determine_shorts_status(folder_path: "Path") -> ShortsStatus:
    """
    Determine the production status of a shorts folder.
    
    Args:
        folder_path: Path to the shorts folder (e.g., outputs/shorts/42/)
    
    Returns:
        ShortsStatus enum value
    """
    from pathlib import Path
    
    folder = Path(folder_path) if not isinstance(folder_path, Path) else folder_path
    
    # Check if folder exists
    if not folder.exists():
        return ShortsStatus.NEEDS_EXPORT
    
    # Check if all 6 required files exist
    for filename in SHORTS_REQUIRED_FILES:
        if not (folder / filename).exists():
            return ShortsStatus.NEEDS_EXPORT
    
    # Check if voiceover.mp3 exists
    if not (folder / "voiceover.mp3").exists():
        return ShortsStatus.NEEDS_TTS
    
    # Check if final.mp4 exists (rendered)
    if (folder / "final.mp4").exists():
        return ShortsStatus.RENDERED
    
    # All files present, ready to render
    return ShortsStatus.READY
