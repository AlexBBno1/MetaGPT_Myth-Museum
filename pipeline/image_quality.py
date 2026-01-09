"""
Myth Museum - Image Quality Validator

Validates generated images for quality issues:
- Brightness (too dark / too bright)
- Contrast (washed out)
- Sharpness (blurry)
- Color cast (green/gray tint)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Constants - Thresholds
# ============================================================================

# Brightness: 0.0 (black) to 1.0 (white)
BRIGHTNESS_MIN = 0.15
BRIGHTNESS_MAX = 0.85

# Contrast: standard deviation of pixel values
CONTRAST_MIN = 0.3

# Sharpness: Laplacian variance
SHARPNESS_MIN = 80.0

# Color cast: max ratio of any channel to others
COLOR_CAST_RATIO_MAX = 1.25  # If G > 1.25 * max(R, B), it's green-tinted


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class QualityMetrics:
    """Image quality metrics."""
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    color_cast: str = "none"  # none, green, red, blue, gray
    rgb_balance: dict = field(default_factory=lambda: {"r": 0.0, "g": 0.0, "b": 0.0})
    
    def to_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "sharpness": round(self.sharpness, 1),
            "color_cast": self.color_cast,
            "rgb_balance": {k: round(v, 3) for k, v in self.rgb_balance.items()},
        }


@dataclass
class QualityResult:
    """Result of quality validation."""
    passed: bool
    metrics: QualityMetrics
    failures: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "failures": self.failures,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Quality Checks
# ============================================================================


def calculate_brightness(img_array: np.ndarray) -> float:
    """
    Calculate image brightness.
    
    Args:
        img_array: RGB image as numpy array (H, W, 3)
    
    Returns:
        Brightness value 0.0-1.0
    """
    # Convert to grayscale using luminosity method
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    return float(np.mean(gray) / 255.0)


def calculate_contrast(img_array: np.ndarray) -> float:
    """
    Calculate image contrast using standard deviation.
    
    Args:
        img_array: RGB image as numpy array
    
    Returns:
        Contrast value (normalized std dev)
    """
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    return float(np.std(gray) / 255.0)


def calculate_sharpness(img_array: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    
    Higher values = sharper image.
    
    Args:
        img_array: RGB image as numpy array
    
    Returns:
        Sharpness score (Laplacian variance)
    """
    try:
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
        
    except ImportError:
        # Fallback without cv2: simple gradient-based sharpness
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        
        # Calculate gradients
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        
        # Sharpness as mean gradient magnitude
        sharpness = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
        
        # Scale to match Laplacian variance roughly
        return float(sharpness * 10)


def detect_color_cast(img_array: np.ndarray) -> tuple[str, dict]:
    """
    Detect color cast in image.
    
    Args:
        img_array: RGB image as numpy array
    
    Returns:
        Tuple of (cast_type, rgb_balance)
        cast_type: "none", "green", "red", "blue", "gray"
    """
    # Calculate mean of each channel
    r_mean = float(np.mean(img_array[:, :, 0]))
    g_mean = float(np.mean(img_array[:, :, 1]))
    b_mean = float(np.mean(img_array[:, :, 2]))
    
    total = r_mean + g_mean + b_mean
    if total == 0:
        return "gray", {"r": 0.33, "g": 0.33, "b": 0.33}
    
    # Normalize to ratios
    rgb_balance = {
        "r": r_mean / total,
        "g": g_mean / total,
        "b": b_mean / total,
    }
    
    # Check for color cast
    channels = {"r": r_mean, "g": g_mean, "b": b_mean}
    max_channel = max(channels, key=channels.get)
    max_value = channels[max_channel]
    
    # Get max of other channels
    other_values = [v for k, v in channels.items() if k != max_channel]
    max_other = max(other_values) if other_values else max_value
    
    # Check if dominant channel exceeds threshold
    if max_other > 0 and max_value / max_other > COLOR_CAST_RATIO_MAX:
        cast_map = {"r": "red", "g": "green", "b": "blue"}
        return cast_map[max_channel], rgb_balance
    
    # Check for gray/washed out (low saturation)
    # If all channels are very similar AND low contrast
    channel_range = max(channels.values()) - min(channels.values())
    if channel_range < 20:  # Very similar channels
        avg_brightness = total / 3 / 255
        if avg_brightness > 0.7 or avg_brightness < 0.3:
            return "gray", rgb_balance
    
    return "none", rgb_balance


# ============================================================================
# Image Quality Validator
# ============================================================================


class ImageQualityValidator:
    """
    Validate image quality for storyboard backgrounds.
    
    Checks brightness, contrast, sharpness, and color cast.
    """
    
    def __init__(
        self,
        brightness_range: tuple[float, float] = (BRIGHTNESS_MIN, BRIGHTNESS_MAX),
        contrast_min: float = CONTRAST_MIN,
        sharpness_min: float = SHARPNESS_MIN,
        color_cast_ratio_max: float = COLOR_CAST_RATIO_MAX,
    ):
        """
        Initialize validator with thresholds.
        
        Args:
            brightness_range: (min, max) brightness
            contrast_min: Minimum contrast
            sharpness_min: Minimum sharpness
            color_cast_ratio_max: Maximum color channel ratio
        """
        self.brightness_min, self.brightness_max = brightness_range
        self.contrast_min = contrast_min
        self.sharpness_min = sharpness_min
        self.color_cast_ratio_max = color_cast_ratio_max
    
    def validate(self, image_path: Path) -> QualityResult:
        """
        Validate an image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            QualityResult with metrics and pass/fail status
        """
        metrics = QualityMetrics()
        failures = []
        recommendations = []
        
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            
            # Calculate metrics
            metrics.brightness = calculate_brightness(img_array)
            metrics.contrast = calculate_contrast(img_array)
            metrics.sharpness = calculate_sharpness(img_array)
            metrics.color_cast, metrics.rgb_balance = detect_color_cast(img_array)
            
            # Check brightness
            if metrics.brightness < self.brightness_min:
                failures.append(f"Too dark: brightness={metrics.brightness:.2f} < {self.brightness_min}")
                recommendations.append("Add lighting keywords: 'bright lighting', 'well-lit'")
            elif metrics.brightness > self.brightness_max:
                failures.append(f"Too bright: brightness={metrics.brightness:.2f} > {self.brightness_max}")
                recommendations.append("Reduce exposure: 'soft lighting', 'not overexposed'")
            
            # Check contrast
            if metrics.contrast < self.contrast_min:
                failures.append(f"Low contrast: contrast={metrics.contrast:.2f} < {self.contrast_min}")
                recommendations.append("Add contrast: 'high contrast', 'dramatic lighting'")
            
            # Check sharpness
            if metrics.sharpness < self.sharpness_min:
                failures.append(f"Blurry: sharpness={metrics.sharpness:.1f} < {self.sharpness_min}")
                recommendations.append("Add sharpness: 'sharp focus', 'detailed', '8k'")
            
            # Check color cast
            if metrics.color_cast != "none":
                failures.append(f"Color cast detected: {metrics.color_cast}")
                color_fixes = {
                    "green": "Add: 'natural colors', 'no green tint'",
                    "red": "Add: 'natural colors', 'no red tint'",
                    "blue": "Add: 'warm lighting', 'no blue tint'",
                    "gray": "Add: 'vibrant colors', 'colorful', 'not washed out'",
                }
                recommendations.append(color_fixes.get(metrics.color_cast, "Adjust colors"))
            
            passed = len(failures) == 0
            
            return QualityResult(
                passed=passed,
                metrics=metrics,
                failures=failures,
                recommendations=recommendations,
            )
            
        except Exception as e:
            logger.error(f"Failed to validate image {image_path}: {e}")
            return QualityResult(
                passed=False,
                metrics=metrics,
                failures=[f"Error loading image: {str(e)}"],
                recommendations=["Check image file is valid"],
            )
    
    def validate_batch(self, image_paths: list[Path]) -> list[QualityResult]:
        """
        Validate multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of QualityResult
        """
        return [self.validate(path) for path in image_paths]


# ============================================================================
# Prompt Rewriter
# ============================================================================


def generate_improved_prompt(
    original_prompt: str,
    quality_result: QualityResult,
) -> str:
    """
    Generate improved prompt based on quality failures.
    
    Args:
        original_prompt: Original image prompt
        quality_result: Quality validation result
    
    Returns:
        Improved prompt with fixes
    """
    improved = original_prompt
    
    # Add fixes based on failures
    fixes = []
    
    for failure in quality_result.failures:
        if "dark" in failure.lower():
            fixes.append("bright lighting")
        elif "bright" in failure.lower():
            fixes.append("soft natural lighting")
        elif "contrast" in failure.lower():
            fixes.append("high contrast")
        elif "blurry" in failure.lower():
            fixes.append("sharp focus, highly detailed")
        elif "green" in failure.lower():
            fixes.append("natural colors, no green tint")
        elif "red" in failure.lower():
            fixes.append("natural colors, balanced white balance")
        elif "blue" in failure.lower():
            fixes.append("warm lighting, natural colors")
        elif "gray" in failure.lower():
            fixes.append("vibrant saturated colors")
    
    if fixes:
        fix_str = ", ".join(fixes)
        improved = f"{original_prompt}, {fix_str}"
    
    return improved


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Quality Validator")
    parser.add_argument("images", nargs="+", help="Image files to validate")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    validator = ImageQualityValidator()
    
    results = []
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"File not found: {image_path}")
            continue
        
        result = validator.validate(path)
        results.append({"path": str(path), **result.to_dict()})
        
        if not args.json:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"\n{status} {path.name}")
            print(f"  Brightness: {result.metrics.brightness:.2f}")
            print(f"  Contrast: {result.metrics.contrast:.2f}")
            print(f"  Sharpness: {result.metrics.sharpness:.1f}")
            print(f"  Color Cast: {result.metrics.color_cast}")
            
            if result.failures:
                print(f"  Failures: {result.failures}")
            if result.recommendations:
                print(f"  Recommendations: {result.recommendations}")
    
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
