"""
Myth Museum - Folder Naming Utilities

Standardized folder naming for video outputs.
Format: {series_slug}_{topic_slug}/
"""

import re
import unicodedata
from pathlib import Path


def slugify(text: str, max_length: int = 30) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify
        max_length: Maximum length of slug
    
    Returns:
        Lowercase slug with hyphens
    
    Examples:
        >>> slugify("Napoleon Height Myth")
        'napoleon-height-myth'
        >>> slugify("The Aztec Civilization!")
        'aztec-civilization'
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    words = text.split()
    words = [w for w in words if w not in stopwords]
    text = ' '.join(words)
    
    # Replace non-alphanumeric with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    # Collapse multiple hyphens
    text = re.sub(r'-+', '-', text)
    
    # Truncate to max length, but don't cut words
    if len(text) > max_length:
        text = text[:max_length].rsplit('-', 1)[0]
    
    return text or 'untitled'


def series_to_slug(series_name: str) -> str:
    """
    Convert series name to slug.
    
    Args:
        series_name: Series name (e.g., "Greek Myths", "History Lies")
    
    Returns:
        Slug (e.g., "greek-myths", "history-lies")
    """
    return slugify(series_name, max_length=20)


def generate_folder_name(
    series_name: str,
    topic: str,
    video_type: str = "",
) -> str:
    """
    Generate standardized folder name.
    
    Format: {series_slug}_{topic_slug} or {series_slug}_{topic_slug}_{type}
    
    Args:
        series_name: Series name
        topic: Video topic
        video_type: Optional type suffix ("image", "video", or empty for no suffix)
    
    Returns:
        Folder name
    
    Examples:
        >>> generate_folder_name("Greek Myths", "Hades the god")
        'greek-myths_hades-god'
        >>> generate_folder_name("History Lies", "Napoleon Height Myth", "video")
        'history-lies_napoleon-height-myth_video'
        >>> generate_folder_name("Lost Civs", "Aztec", "image")
        'lost-civs_aztec_image'
    """
    series_slug = series_to_slug(series_name)
    topic_slug = slugify(topic, max_length=30)
    
    base_name = f"{series_slug}_{topic_slug}"
    
    if video_type in ("image", "video"):
        return f"{base_name}_{video_type}"
    
    return base_name


def parse_folder_name(folder_name: str) -> tuple[str, str, str]:
    """
    Parse folder name back into series, topic, and type.
    
    Args:
        folder_name: Folder name to parse
    
    Returns:
        Tuple of (series_slug, topic_slug, video_type)
        video_type is "image", "video", or "" if not specified
    """
    video_type = ""
    
    # Check for type suffix
    if folder_name.endswith("_video"):
        video_type = "video"
        folder_name = folder_name[:-6]
    elif folder_name.endswith("_image"):
        video_type = "image"
        folder_name = folder_name[:-6]
    
    if '_' in folder_name:
        parts = folder_name.split('_', 1)
        return parts[0], parts[1] if len(parts) > 1 else '', video_type
    else:
        return 'unknown', folder_name, video_type


def get_video_type(folder_name: str) -> str:
    """
    Extract video type from folder name.
    
    Args:
        folder_name: Folder name
    
    Returns:
        "image", "video", or "" if not specified
    """
    if folder_name.endswith("_video"):
        return "video"
    elif folder_name.endswith("_image"):
        return "image"
    return ""


def infer_series_from_folder(folder_name: str) -> str:
    """
    Try to infer series name from existing folder name.
    
    Used for migration of legacy folders.
    
    Args:
        folder_name: Existing folder name
    
    Returns:
        Inferred series name
    """
    folder_lower = folder_name.lower()
    
    # Known topic -> series mappings
    mappings = {
        'hades': 'Greek Myths',
        'zeus': 'Greek Myths',
        'poseidon': 'Greek Myths',
        'athena': 'Greek Myths',
        'odysseus': 'Greek Myths',
        'odyssey': 'Greek Myths',
        'calypso': 'Greek Myths',
        'myth': 'Greek Myths',
        'aztec': 'Lost Civs',
        'maya': 'Lost Civs',
        'inca': 'Lost Civs',
        'egypt': 'Lost Civs',
        'napoleon': 'History Lies',
        'lincoln': 'History Lies',
        'vangogh': 'History Lies',
        'van_gogh': 'History Lies',
        'colosseum': 'Empire Files',
        'rome': 'Empire Files',
        'caesar': 'Empire Files',
        'war': 'War Myths',
        'battle': 'War Myths',
    }
    
    for keyword, series in mappings.items():
        if keyword in folder_lower:
            return series
    
    # Check if it's a numeric folder (legacy claim-based)
    if folder_name.isdigit():
        return 'Myth Museum'
    
    return 'Myth Museum'


def suggest_new_folder_name(old_name: str, series_name: str = None) -> str:
    """
    Suggest new folder name for migration.
    
    Args:
        old_name: Current folder name
        series_name: Optional series name override
    
    Returns:
        Suggested new folder name
    """
    # Infer series if not provided
    if not series_name:
        series_name = infer_series_from_folder(old_name)
    
    # Clean up old name for topic
    topic = old_name
    
    # Remove common suffixes
    for suffix in ['_demo', '_v2', '_v3', '_test', '_old']:
        if topic.endswith(suffix):
            topic = topic[:-len(suffix)]
    
    # Handle numeric folders
    if topic.isdigit():
        topic = f"claim-{topic}"
    
    return generate_folder_name(series_name, topic)


# ============================================================================
# Migration Helpers
# ============================================================================

def get_migration_plan(shorts_dir: Path) -> list[dict]:
    """
    Generate migration plan for existing folders.
    
    Args:
        shorts_dir: Path to shorts output directory
    
    Returns:
        List of migration actions
    """
    if not shorts_dir.exists():
        return []
    
    plan = []
    
    for folder in sorted(shorts_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        old_name = folder.name
        
        # Skip if already in new format
        if '_' in old_name and not old_name.endswith('_demo'):
            # Check if it looks like series_topic format
            series_slug, topic_slug = parse_folder_name(old_name)
            if series_slug in ['greek-myths', 'history-lies', 'lost-civs', 'war-myths', 
                               'empire-files', 'myth-museum', 'mind-tricks', 'science-myths']:
                continue
        
        new_name = suggest_new_folder_name(old_name)
        
        if old_name != new_name:
            plan.append({
                'old_path': folder,
                'new_path': shorts_dir / new_name,
                'old_name': old_name,
                'new_name': new_name,
                'series': infer_series_from_folder(old_name),
            })
    
    return plan


def execute_migration(plan: list[dict], dry_run: bool = True) -> list[dict]:
    """
    Execute folder migration.
    
    Args:
        plan: Migration plan from get_migration_plan()
        dry_run: If True, only simulate the migration
    
    Returns:
        List of results
    """
    results = []
    
    for action in plan:
        old_path = action['old_path']
        new_path = action['new_path']
        
        result = {
            'old_name': action['old_name'],
            'new_name': action['new_name'],
            'success': False,
            'error': None,
        }
        
        if new_path.exists():
            result['error'] = f"Target already exists: {new_path}"
            results.append(result)
            continue
        
        if dry_run:
            result['success'] = True
            result['dry_run'] = True
        else:
            try:
                old_path.rename(new_path)
                result['success'] = True
            except Exception as e:
                result['error'] = str(e)
        
        results.append(result)
    
    return results
