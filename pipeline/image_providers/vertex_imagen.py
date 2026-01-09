"""
Myth Museum - Multi-Provider Image Generation

Generate images using multiple AI providers with automatic fallback:
1. Vertex AI Imagen 3 (Google Cloud) - Highest quality
2. DALL-E 3 (OpenAI) - High quality fallback
3. Vertex AI Imagen @006 (Google Cloud) - Standard quality
4. Pexels API (stock photos)
5. Lorem Picsum (placeholder)

Quality Modes:
- "high": Imagen 3 -> DALL-E 3 -> Imagen @006 -> Pexels -> Picsum
- "standard": Imagen @006 -> DALL-E 3 -> Pexels -> Picsum
- "fallback": Pexels -> Picsum (no AI generation)

Environment Variables:
- IMAGE_PROVIDER: "vertex" | "openai" | "auto" (default: auto)
- IMAGE_QUALITY: "high" | "standard" | "fallback" (default: high)
- VERTEX_PROJECT_ID, VERTEX_LOCATION
- OPENAI_API_KEY (for DALL-E 3)
- PEXELS_API_KEY (for stock photo fallback)
"""

import asyncio
import base64
import json
import os
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

from core.logging import get_logger

logger = get_logger(__name__)

# Provider selection
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "auto")  # vertex, openai, auto
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "high")  # high, standard, fallback

# Model names
IMAGEN3_MODEL = "imagen-3.0-generate-002"
IMAGEN_STANDARD_MODEL = "imagegeneration@006"

# Imagen 3 rate limiting (quota is 1 request per minute)
IMAGEN3_DELAY_SECONDS = 65  # Wait 65 seconds between requests to be safe


@dataclass
class ImageGenerationResult:
    """Result of image generation."""
    success: bool = False
    local_path: Optional[Path] = None
    prompt: str = ""
    seed: Optional[int] = None
    error: Optional[str] = None
    latency_ms: int = 0
    retries: int = 0
    source: str = ""  # "vertex", "pexels", or "picsum"
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "local_path": str(self.local_path) if self.local_path else None,
            "prompt": self.prompt,
            "seed": self.seed,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "source": self.source,
        }


class VertexImagenProvider:
    """
    Vertex AI Imagen image generation provider.
    
    Supports both Imagen 3 (high quality) and standard Imagen @006.
    
    Environment variables:
    - VERTEX_PROJECT_ID: Google Cloud project ID
    - VERTEX_LOCATION: Region (default: us-central1)
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    """
    
    def __init__(self):
        self.project_id = os.getenv("VERTEX_PROJECT_ID", "")
        self.location = os.getenv("VERTEX_LOCATION", "us-central1")
        
        # Try to find service account file
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            # Look for JSON file in current directory
            json_files = list(Path(".").glob("*-*.json"))
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)
                        if data.get("type") == "service_account":
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(jf)
                            self.project_id = self.project_id or data.get("project_id", "")
                            break
                except:
                    continue
        
        # Cache for different models
        self._models: dict = {}
        self._vertexai_initialized = False
        
        logger.info(f"Vertex AI config loaded: project={self.project_id}, location={self.location}")
    
    def _init_vertexai(self):
        """Initialize Vertex AI SDK."""
        if not self._vertexai_initialized:
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            self._vertexai_initialized = True
    
    def _get_model(self, model_name: str = IMAGEN_STANDARD_MODEL):
        """Lazy load Vertex AI model by name."""
        if model_name not in self._models:
            from vertexai.preview.vision_models import ImageGenerationModel
            
            self._init_vertexai()
            self._models[model_name] = ImageGenerationModel.from_pretrained(model_name)
            logger.info(f"Vertex AI Imagen initialized: {model_name}")
        
        return self._models[model_name]
    
    def is_available(self) -> bool:
        """Check if Vertex AI is configured."""
        return bool(self.project_id)
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: str = "9:16",
        model_name: str = IMAGEN_STANDARD_MODEL,
        num_retries: int = 3,
    ) -> ImageGenerationResult:
        """
        Generate an image using Vertex AI Imagen.
        
        Args:
            prompt: Image generation prompt
            output_path: Where to save the image
            aspect_ratio: Aspect ratio (default 9:16 for Shorts)
            model_name: Model to use (imagen-3.0-generate-002 or imagegeneration@006)
            num_retries: Number of retries on failure
        
        Returns:
            ImageGenerationResult
        """
        result = ImageGenerationResult(success=False, prompt=prompt, source="vertex")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(1, num_retries + 1):
            result.retries = attempt
            start_time = time.time()
            
            try:
                model = self._get_model(model_name)
                
                # Generate image
                images = model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_few",
                    person_generation="allow_adult",
                )
                
                # Handle different response formats
                image_list = []
                if hasattr(images, 'images'):
                    image_list = images.images
                elif hasattr(images, '__iter__'):
                    image_list = list(images)
                else:
                    image_list = [images]
                
                if image_list:
                    # Save image
                    image = image_list[0]
                    image.save(str(output_path))
                    
                    result.success = True
                    result.local_path = output_path
                    result.latency_ms = int((time.time() - start_time) * 1000)
                    
                    # Try to get seed if available
                    if hasattr(image, '_generation_parameters'):
                        result.seed = image._generation_parameters.get('seed')
                    
                    logger.info(f"Generated image: {output_path} (seed={result.seed}, {result.latency_ms}ms)")
                    return result
                else:
                    logger.warning(f"Attempt {attempt}: No images returned")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Imagen generation error: {error_msg}")
                logger.warning(f"Attempt {attempt} failed: {error_msg}")
                result.error = error_msg
                
                # Don't retry on permission errors
                if "403" in error_msg or "Permission" in error_msg:
                    break
                
                # Don't retry on content policy errors
                if "sensitive words" in error_msg.lower():
                    break
        
        result.error = result.error or "Failed after all attempts"
        return result
    
    async def generate_batch(
        self,
        prompts: list[dict],
        output_dir: Path,
    ) -> list[ImageGenerationResult]:
        """
        Generate multiple images.
        
        Args:
            prompts: List of dicts with 'segment' and 'prompt' keys
            output_dir: Output directory
        
        Returns:
            List of ImageGenerationResult
        """
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for p in prompts:
            segment = p.get('segment', 'image')
            prompt = p.get('prompt', '')
            output_path = output_dir / f"{segment}.jpg"
            
            result = await self.generate_image(prompt, output_path)
            results.append(result)
        
        return results


# ============================================================================
# DALL-E 3 Provider (OpenAI)
# ============================================================================

class DallE3Provider:
    """
    OpenAI DALL-E 3 image generation provider.
    
    Environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - DALLE_MODEL: Model name (default: dall-e-3)
    
    DALL-E 3 advantages:
    - Better prompt understanding
    - Higher quality output
    - Better text rendering (though we avoid text)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("DALLE_MODEL", "dall-e-3")
        self.base_url = "https://api.openai.com/v1/images/generations"
        
        if self.api_key:
            logger.info(f"DALL-E 3 configured: model={self.model}")
        else:
            logger.debug("DALL-E 3 not configured (no OPENAI_API_KEY)")
    
    def is_available(self) -> bool:
        """Check if DALL-E is configured."""
        return bool(self.api_key)
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        size: str = "1024x1792",  # Vertical for Shorts
        quality: str = "standard",  # or "hd"
        num_retries: int = 2,
    ) -> ImageGenerationResult:
        """
        Generate an image using DALL-E 3.
        
        Args:
            prompt: Image generation prompt
            output_path: Where to save the image
            size: Image size (1024x1024, 1024x1792, 1792x1024)
            quality: "standard" or "hd"
            num_retries: Number of retries on failure
        
        Returns:
            ImageGenerationResult
        """
        result = ImageGenerationResult(success=False, prompt=prompt, source="dalle3")
        
        if not self.api_key:
            result.error = "OPENAI_API_KEY not set"
            return result
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(1, num_retries + 1):
            result.retries = attempt
            start_time = time.time()
            
            try:
                # Prepare request
                request_data = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "quality": quality,
                    "response_format": "b64_json",  # Get base64 data directly
                }).encode('utf-8')
                
                request = urllib.request.Request(
                    self.base_url,
                    data=request_data,
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                    },
                    method='POST'
                )
                
                ctx = ssl.create_default_context()
                
                with urllib.request.urlopen(request, context=ctx, timeout=120) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    if 'data' in data and len(data['data']) > 0:
                        # Decode base64 image
                        b64_image = data['data'][0].get('b64_json', '')
                        if b64_image:
                            image_data = base64.b64decode(b64_image)
                            output_path.write_bytes(image_data)
                            
                            result.success = True
                            result.local_path = output_path
                            result.latency_ms = int((time.time() - start_time) * 1000)
                            
                            # Get revised prompt if available
                            revised = data['data'][0].get('revised_prompt', '')
                            if revised:
                                logger.debug(f"DALL-E revised prompt: {revised[:100]}...")
                            
                            logger.info(f"DALL-E 3 generated: {output_path} ({result.latency_ms}ms)")
                            return result
                    
                    logger.warning(f"DALL-E attempt {attempt}: No image data returned")
                    
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8') if e.fp else str(e)
                try:
                    error_json = json.loads(error_body)
                    error_msg = error_json.get('error', {}).get('message', str(e))
                except:
                    error_msg = error_body[:200]
                
                logger.error(f"DALL-E HTTP error: {e.code} - {error_msg}")
                result.error = error_msg
                
                # Don't retry on content policy violations
                if e.code == 400 and 'content_policy' in error_msg.lower():
                    break
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"DALL-E generation error: {error_msg}")
                result.error = error_msg
        
        result.error = result.error or "Failed after all attempts"
        return result


# ============================================================================
# Fallback Providers
# ============================================================================

class PexelsFallback:
    """
    Pexels API fallback for image generation.
    
    Requires PEXELS_API_KEY environment variable.
    Free tier: 200 requests/hour.
    """
    
    def __init__(self):
        self.api_key = os.getenv("PEXELS_API_KEY", "")
        self.base_url = "https://api.pexels.com/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def download_image(
        self,
        keyword: str,
        output_path: Path,
        index: int = 0,
    ) -> ImageGenerationResult:
        """Download image from Pexels by keyword."""
        result = ImageGenerationResult(success=False, prompt=keyword, source="pexels")
        start_time = time.time()
        
        if not self.api_key:
            result.error = "PEXELS_API_KEY not set"
            return result
        
        try:
            url = f"{self.base_url}/search?query={keyword.replace(' ', '+')}&orientation=portrait&per_page=5"
            
            request = urllib.request.Request(url, headers={
                'Authorization': self.api_key,
                'User-Agent': 'Mozilla/5.0'
            })
            
            ctx = ssl.create_default_context()
            
            with urllib.request.urlopen(request, context=ctx, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                photos = data.get('photos', [])
                if photos and len(photos) > index:
                    photo = photos[index % len(photos)]
                    img_url = photo['src']['portrait']
                    
                    # Download the image
                    img_request = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(img_request, context=ctx, timeout=30) as img_response:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(img_response.read())
                        
                        result.success = True
                        result.local_path = output_path
                        result.latency_ms = int((time.time() - start_time) * 1000)
                        logger.info(f"Pexels download: {output_path} ({result.latency_ms}ms)")
                        return result
                
                result.error = "No photos found"
                
        except Exception as e:
            result.error = str(e)
            logger.warning(f"Pexels fallback failed: {e}")
        
        return result


class PicsumFallback:
    """
    Lorem Picsum fallback - always works, no API key required.
    
    Returns random high-quality stock photos.
    """
    
    async def download_image(
        self,
        output_path: Path,
        seed: int = 0,
        width: int = 1080,
        height: int = 1920,
    ) -> ImageGenerationResult:
        """Download random image from Lorem Picsum."""
        result = ImageGenerationResult(success=False, source="picsum")
        start_time = time.time()
        
        try:
            # Use seed for consistent images
            url = f"https://picsum.photos/seed/{seed + 100}/{width}/{height}"
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(request, context=ctx, timeout=30) as response:
                data = response.read()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                
                result.success = True
                result.local_path = output_path
                result.latency_ms = int((time.time() - start_time) * 1000)
                result.seed = seed
                logger.info(f"Picsum download: {output_path} (seed={seed}, {result.latency_ms}ms)")
                return result
                
        except Exception as e:
            result.error = str(e)
            logger.warning(f"Picsum fallback failed: {e}")
        
        return result


# ============================================================================
# Unified Provider with Fallbacks
# ============================================================================

class ImageProviderWithFallback:
    """
    Unified image provider with automatic fallbacks.
    
    Quality-based fallback chain:
    - "high": Imagen 3 -> DALL-E 3 -> Imagen @006 -> Pexels -> Picsum
    - "standard": Imagen @006 -> DALL-E 3 -> Pexels -> Picsum
    - "fallback": Pexels -> Picsum (no AI generation)
    
    Environment Variables:
    - IMAGE_PROVIDER: "vertex" | "openai" | "auto" (default: auto)
    - IMAGE_QUALITY: "high" | "standard" | "fallback" (default: high)
    """
    
    def __init__(self):
        self.vertex = VertexImagenProvider()
        self.dalle = DallE3Provider()
        self.pexels = PexelsFallback()
        self.picsum = PicsumFallback()
        self._fallback_index = 0
        self._last_imagen3_request = 0  # Timestamp for Imagen 3 rate limiting
        
        # Log available providers
        providers = []
        if self.vertex.is_available():
            providers.append("Vertex AI (Imagen 3 + @006)")
        if self.dalle.is_available():
            providers.append("DALL-E 3")
        if self.pexels.is_available():
            providers.append("Pexels")
        providers.append("Picsum")
        
        logger.info(f"Image providers available: {', '.join(providers)}")
        logger.info(f"Mode: {IMAGE_PROVIDER}, Quality: {IMAGE_QUALITY}")
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        fallback_keyword: str = "",
        aspect_ratio: str = "9:16",
        quality: str = "",  # Override IMAGE_QUALITY if specified
    ) -> ImageGenerationResult:
        """
        Generate image with automatic fallbacks based on quality setting.
        
        Fallback chain by quality:
        - "high": Imagen 3 -> DALL-E 3 -> Imagen @006 -> Pexels -> Picsum
        - "standard": Imagen @006 -> DALL-E 3 -> Pexels -> Picsum
        - "fallback": Pexels -> Picsum (no AI generation)
        
        Args:
            prompt: AI generation prompt
            output_path: Where to save the image
            fallback_keyword: Keyword for fallback providers (Pexels search)
            aspect_ratio: Aspect ratio (for Vertex AI)
            quality: Override quality setting ("high", "standard", "fallback")
        
        Returns:
            ImageGenerationResult with source field indicating provider used
        """
        output_path = Path(output_path)
        quality = quality or IMAGE_QUALITY
        
        # Fallback-only mode (no AI generation)
        if quality == "fallback":
            return await self._try_stock_fallbacks(output_path, fallback_keyword)
        
        # High quality mode: Try Imagen 3 first
        if quality == "high" and self.vertex.is_available():
            logger.info("Trying Imagen 3 (high quality)...")
            result = await self._try_imagen3(prompt, output_path, aspect_ratio)
            if result.success:
                return result
            logger.info("Imagen 3 failed, trying fallbacks...")
        
        # Standard quality or Imagen 3 failed: Try standard providers
        # Determine provider order based on IMAGE_PROVIDER setting
        if IMAGE_PROVIDER == "openai":
            # DALL-E first, then Vertex as fallback
            if self.dalle.is_available():
                result = await self._try_dalle(prompt, output_path)
                if result.success:
                    return result
            
            if self.vertex.is_available():
                result = await self._try_vertex_standard(prompt, output_path, aspect_ratio)
                if result.success:
                    return result
                
        elif IMAGE_PROVIDER == "vertex":
            # Vertex only (no DALL-E)
            if self.vertex.is_available():
                result = await self._try_vertex_standard(prompt, output_path, aspect_ratio)
                if result.success:
                    return result
                
        else:  # "auto" - try both
            # Try Vertex AI @006 first (usually faster, no per-image cost)
            if self.vertex.is_available():
                result = await self._try_vertex_standard(prompt, output_path, aspect_ratio)
                if result.success:
                    return result
            
            # Try DALL-E 3 as AI fallback
            if self.dalle.is_available():
                logger.info("Vertex AI failed, trying DALL-E 3...")
                result = await self._try_dalle(prompt, output_path)
                if result.success:
                    return result
        
        # Stock photo fallbacks
        return await self._try_stock_fallbacks(output_path, fallback_keyword)
    
    async def _try_stock_fallbacks(
        self,
        output_path: Path,
        fallback_keyword: str,
    ) -> ImageGenerationResult:
        """Try stock photo fallbacks (Pexels -> Picsum)."""
        logger.info(f"AI generation failed, trying stock photos for: {fallback_keyword or 'image'}")
        
        # Try Pexels if API key is available
        if self.pexels.is_available() and fallback_keyword:
            result = await self.pexels.download_image(
                keyword=fallback_keyword,
                output_path=output_path,
                index=self._fallback_index,
            )
            
            if result.success:
                self._fallback_index += 1
                return result
        
        # Fall back to Picsum (always works)
        result = await self.picsum.download_image(
            output_path=output_path,
            seed=self._fallback_index,
        )
        
        self._fallback_index += 1
        return result
    
    async def _wait_for_imagen3_quota(self) -> None:
        """
        Wait for Imagen 3 quota if necessary.
        
        Imagen 3 has a quota of 1 request per minute in us-central1.
        This method ensures we don't exceed this rate limit.
        """
        elapsed = time.time() - self._last_imagen3_request
        if self._last_imagen3_request > 0 and elapsed < IMAGEN3_DELAY_SECONDS:
            wait_time = IMAGEN3_DELAY_SECONDS - elapsed
            logger.info(f"Waiting {wait_time:.0f}s for Imagen 3 quota...")
            await asyncio.sleep(wait_time)
        self._last_imagen3_request = time.time()
    
    async def _try_imagen3(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: str,
    ) -> ImageGenerationResult:
        """Try Vertex AI Imagen 3 (highest quality)."""
        # Wait for Imagen 3 quota
        await self._wait_for_imagen3_quota()
        
        result = await self.vertex.generate_image(
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            model_name=IMAGEN3_MODEL,
            num_retries=2,
        )
        if result.success:
            result.source = "imagen3"
        return result
    
    async def _try_vertex_standard(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: str,
    ) -> ImageGenerationResult:
        """Try Vertex AI Imagen @006 (standard quality)."""
        result = await self.vertex.generate_image(
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            model_name=IMAGEN_STANDARD_MODEL,
            num_retries=2,
        )
        if result.success:
            result.source = "vertex"
        return result
    
    async def _try_dalle(
        self,
        prompt: str,
        output_path: Path,
    ) -> ImageGenerationResult:
        """Try DALL-E 3."""
        result = await self.dalle.generate_image(
            prompt=prompt,
            output_path=output_path,
            size="1024x1792",  # Vertical for Shorts
            quality="standard",
            num_retries=2,
        )
        if result.success:
            result.source = "dalle3"
        return result
    
    async def generate_batch(
        self,
        prompts: list[dict],
        output_dir: Path,
        quality: str = "",  # Override quality for batch
    ) -> list[ImageGenerationResult]:
        """
        Generate multiple images with fallbacks.
        
        Args:
            prompts: List of dicts with 'segment', 'prompt', and optional 'fallback_keyword'
            output_dir: Output directory
            quality: Override quality setting for entire batch
        
        Returns:
            List of ImageGenerationResult
        """
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for p in prompts:
            segment = p.get('segment', 'image')
            prompt = p.get('prompt', '')
            fallback_keyword = p.get('fallback_keyword', segment)
            output_path = output_dir / f"{segment}.jpg"
            
            result = await self.generate_image(
                prompt=prompt,
                output_path=output_path,
                fallback_keyword=fallback_keyword,
                quality=quality,
            )
            results.append(result)
        
        return results


# ============================================================================
# Convenience function for quick access
# ============================================================================

def get_image_provider() -> ImageProviderWithFallback:
    """Get a configured image provider with fallback support."""
    return ImageProviderWithFallback()
