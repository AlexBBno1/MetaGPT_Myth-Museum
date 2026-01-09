"""
Myth Museum - LLM Client

OpenAI-compatible LLM client with retry logic.
"""

import asyncio
import json
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_llm_config, load_config
from core.logging import get_logger
from core.models import LLMConfig

logger = get_logger(__name__)


class LLMClient:
    """
    OpenAI-compatible LLM client.
    
    Supports any OpenAI-compatible API by configuring base_url.
    Includes retry logic with exponential backoff.
    """
    
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: API base URL (OpenAI-compatible)
            api_key: API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize async client
        self._client: Optional[AsyncOpenAI] = None
    
    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "LLMClient":
        """
        Create client from configuration.
        
        Args:
            config: Config dict (loads from file if None)
        
        Returns:
            LLMClient instance
        """
        if config is None:
            config = load_config()
        
        llm_config = get_llm_config(config)
        
        return cls(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.timeout,
        )
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client
    
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != "YOUR_API_KEY")
    
    @retry(
        retry=retry_if_exception_type(OpenAIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
        
        Returns:
            Response content string
        """
        if not self.is_configured():
            logger.warning("LLM not configured, returning empty response")
            return ""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM response: {content[:100]}...")
            return content
            
        except OpenAIError as e:
            logger.error(f"LLM API error: {e}")
            raise
    
    async def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Send chat completion and parse JSON response.
        
        Args:
            messages: List of message dicts
            schema: Optional Pydantic model for validation
            temperature: Override default temperature (lower for JSON)
        
        Returns:
            Parsed JSON dict
        """
        # Add JSON instruction to system message
        json_instruction = "\nRespond with valid JSON only. No markdown, no code blocks, just raw JSON."
        
        # Modify or add system message
        modified_messages = []
        has_system = False
        
        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                modified_messages.append({
                    "role": "system",
                    "content": msg["content"] + json_instruction,
                })
            else:
                modified_messages.append(msg)
        
        if not has_system:
            modified_messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant." + json_instruction,
            })
        
        # Use lower temperature for JSON
        temp = temperature or 0.3
        
        response = await self.chat(modified_messages, temperature=temp)
        
        # Try to parse JSON
        try:
            # Clean up response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Validate against schema if provided
            if schema:
                validated = schema(**data)
                data = validated.model_dump()
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {}
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            return {}
    
    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None


# Convenience function for simple usage
async def quick_chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    config: Optional[dict] = None,
) -> str:
    """
    Quick one-off chat completion.
    
    Args:
        prompt: User prompt
        system: System message
        config: Config dict
    
    Returns:
        Response string
    """
    client = LLMClient.from_config(config)
    
    if not client.is_configured():
        return ""
    
    try:
        return await client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ])
    finally:
        await client.close()
