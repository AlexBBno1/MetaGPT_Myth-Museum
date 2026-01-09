"""
Myth Museum - Configuration

Load configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from core.constants import SourceTypeEnum
from core.logging import get_logger
from core.models import LLMConfig, SourceConfig

logger = get_logger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Default config file
DEFAULT_CONFIG_FILE = CONFIG_DIR / "sources.yaml"


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config/sources.yaml)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return _default_config()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    logger.info(f"Loaded config from {config_path}")
    return config


def _default_config() -> dict[str, Any]:
    """Return default configuration."""
    return {
        "llm": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-4-turbo",
        },
        "sources": [],
        "wiki_topics": [
            "health myths",
            "medical misconceptions",
            "historical myths",
            "science myths",
        ],
        "database": {
            "path": str(DATA_DIR / "myth_museum.sqlite"),
        },
        "output": {
            "path": str(OUTPUT_DIR / "packets"),
        },
    }


def get_llm_config(config: Optional[dict[str, Any]] = None) -> LLMConfig:
    """
    Get LLM configuration from config dict and environment variables.
    Environment variables take precedence.
    
    Env vars:
        MYTH_LLM_BASE_URL
        MYTH_LLM_API_KEY
        MYTH_LLM_MODEL
    
    Args:
        config: Configuration dictionary (loads from file if None)
    
    Returns:
        LLMConfig instance
    """
    if config is None:
        config = load_config()
    
    llm_config = config.get("llm", {})
    
    return LLMConfig(
        base_url=os.environ.get("MYTH_LLM_BASE_URL", llm_config.get("base_url", "https://api.openai.com/v1")),
        api_key=os.environ.get("MYTH_LLM_API_KEY", llm_config.get("api_key", "")),
        model=os.environ.get("MYTH_LLM_MODEL", llm_config.get("model", "gpt-4-turbo")),
        temperature=float(llm_config.get("temperature", 0.7)),
        max_tokens=int(llm_config.get("max_tokens", 4096)),
        timeout=int(llm_config.get("timeout", 60)),
    )


def get_sources(config: Optional[dict[str, Any]] = None) -> list[SourceConfig]:
    """
    Get list of source configurations.
    
    Args:
        config: Configuration dictionary (loads from file if None)
    
    Returns:
        List of SourceConfig instances
    """
    if config is None:
        config = load_config()
    
    sources = []
    for idx, src in enumerate(config.get("sources", [])):
        source_type = src.get("type", "rss")
        try:
            source_type_enum = SourceTypeEnum(source_type)
        except ValueError:
            logger.warning(f"Unknown source type: {source_type}, defaulting to RSS")
            source_type_enum = SourceTypeEnum.RSS
        
        sources.append(SourceConfig(
            id=idx + 1,
            name=src.get("name", f"source_{idx}"),
            type=source_type_enum,
            config_json=src.get("config", {}),
            enabled=src.get("enabled", True),
        ))
    
    return sources


def get_wiki_topics(config: Optional[dict[str, Any]] = None) -> list[str]:
    """
    Get list of Wikipedia topic keywords.
    
    Args:
        config: Configuration dictionary (loads from file if None)
    
    Returns:
        List of topic keywords
    """
    if config is None:
        config = load_config()
    
    return config.get("wiki_topics", [])


def get_db_path(config: Optional[dict[str, Any]] = None) -> Path:
    """
    Get database file path.
    
    Args:
        config: Configuration dictionary (loads from file if None)
    
    Returns:
        Path to database file
    """
    if config is None:
        config = load_config()
    
    db_config = config.get("database", {})
    db_path = db_config.get("path", str(DATA_DIR / "myth_museum.sqlite"))
    return Path(db_path)


def get_output_path(config: Optional[dict[str, Any]] = None) -> Path:
    """
    Get output directory path.
    
    Args:
        config: Configuration dictionary (loads from file if None)
    
    Returns:
        Path to output directory
    """
    if config is None:
        config = load_config()
    
    output_config = config.get("output", {})
    output_path = output_config.get("path", str(OUTPUT_DIR / "packets"))
    return Path(output_path)


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for directory in [CONFIG_DIR, DATA_DIR, OUTPUT_DIR, LOGS_DIR, OUTPUT_DIR / "packets"]:
        directory.mkdir(parents=True, exist_ok=True)
