"""
Myth Museum - Vertex AI Configuration

Loads Vertex AI configuration from environment variables and service account JSON.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "imagegeneration@006"  # Imagen 3


# ============================================================================
# Configuration Data Class
# ============================================================================


@dataclass
class VertexConfig:
    """Vertex AI configuration."""
    project_id: str
    location: str
    model: str
    credentials_path: Optional[str]
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        if not self.project_id:
            return False
        if not self.location:
            return False
        if self.credentials_path and not Path(self.credentials_path).exists():
            logger.warning(f"Credentials file not found: {self.credentials_path}")
            return False
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model": self.model,
            "credentials_path": self.credentials_path,
        }


# ============================================================================
# Configuration Loader
# ============================================================================


def load_vertex_config() -> VertexConfig:
    """
    Load Vertex AI configuration from environment variables.
    
    Environment Variables:
        VERTEX_PROJECT_ID: Google Cloud project ID
        VERTEX_LOCATION: Region (default: us-central1)
        VERTEX_MODEL: Model name (default: imagegeneration@006)
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    
    Returns:
        VertexConfig object
    """
    project_id = os.getenv("VERTEX_PROJECT_ID", "")
    location = os.getenv("VERTEX_LOCATION", DEFAULT_LOCATION)
    model = os.getenv("VERTEX_MODEL", DEFAULT_MODEL)
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # Try to infer project ID from credentials file if not set
    if not project_id and credentials_path:
        project_id = _extract_project_from_credentials(credentials_path)
    
    config = VertexConfig(
        project_id=project_id,
        location=location,
        model=model,
        credentials_path=credentials_path if credentials_path else None,
    )
    
    if config.is_valid():
        logger.info(f"Vertex AI config loaded: project={project_id}, location={location}, model={model}")
    else:
        logger.warning("Vertex AI config incomplete - check environment variables")
    
    return config


def _extract_project_from_credentials(credentials_path: str) -> str:
    """
    Extract project ID from service account JSON file.
    
    Args:
        credentials_path: Path to service account JSON
    
    Returns:
        Project ID or empty string
    """
    import json
    
    try:
        path = Path(credentials_path)
        if not path.exists():
            return ""
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("project_id", "")
    except Exception as e:
        logger.warning(f"Failed to extract project ID from credentials: {e}")
        return ""


def get_vertex_endpoint(config: VertexConfig) -> str:
    """
    Build Vertex AI endpoint URL.
    
    Args:
        config: Vertex configuration
    
    Returns:
        Full endpoint URL
    """
    return (
        f"https://{config.location}-aiplatform.googleapis.com/v1/"
        f"projects/{config.project_id}/locations/{config.location}/"
        f"publishers/google/models/{config.model}"
    )


def validate_vertex_setup() -> tuple[bool, str]:
    """
    Validate Vertex AI setup is complete.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    config = load_vertex_config()
    
    if not config.project_id:
        return False, "VERTEX_PROJECT_ID not set"
    
    if not config.credentials_path:
        return False, "GOOGLE_APPLICATION_CREDENTIALS not set"
    
    if not Path(config.credentials_path).exists():
        return False, f"Credentials file not found: {config.credentials_path}"
    
    return True, "Vertex AI configuration valid"


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    config = load_vertex_config()
    print(f"Project ID: {config.project_id}")
    print(f"Location: {config.location}")
    print(f"Model: {config.model}")
    print(f"Credentials: {config.credentials_path}")
    print(f"Valid: {config.is_valid()}")
    print(f"Endpoint: {get_vertex_endpoint(config)}")
