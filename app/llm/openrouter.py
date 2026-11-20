"""OpenRouter LLM integration for RealtyAI."""
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from openai import OpenAI
from openai import (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    AuthenticationError,
    NotFoundError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_settings


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class OpenRouterClient:
    """OpenRouter API client with retry logic and fallback support."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 fallback_model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None):
        settings = get_settings()
        
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        self.fallback_model = fallback_model or settings.OPENROUTER_FALLBACK_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.timeout = timeout or settings.LLM_TIMEOUT
        
        self.site_url = settings.OPENROUTER_SITE_URL
        self.app_name = settings.OPENROUTER_APP_NAME
        
        self.logger = logger.bind(component="OpenRouterClient")
        
        # Initialize client
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=self.timeout
            )
        else:
            self.client = None
            self.logger.warning("OpenRouter API key not configured")
    
    def is_configured(self) -> bool:
        """Check if client is properly configured."""
        return self.client is not None and bool(self.api_key)
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
        )),
        reraise=True
    )
    def _call_api(self, model: str, messages: List[Dict[str, str]]) -> LLMResponse:
        """Make API call with retry logic."""
        if not self.client:
            raise RuntimeError("OpenRouter client not configured. Please set OPENROUTER_API_KEY.")
        
        self.logger.debug(f"Calling OpenRouter with model: {model}")
        
        try:
            # Prepare headers
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                extra_headers["X-Title"] = self.app_name
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=extra_headers if extra_headers else None
            )
            
            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            self.logger.debug(f"OpenRouter response: model={model}, tokens={usage}")
            
            return LLMResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {e}")
            raise
    
    def generate(self, 
                 messages: List[Dict[str, str]], 
                 use_fallback: bool = False) -> LLMResponse:
        """Generate response using primary model with fallback."""
        if not self.is_configured():
            raise RuntimeError("OpenRouter client not configured. Set OPENROUTER_API_KEY.")
        
        model = self.fallback_model if use_fallback else self.model
        
        try:
            return self._call_api(model, messages)
        except Exception as e:
            # If primary model fails and fallback is available, try fallback
            if not use_fallback and self.fallback_model and self.fallback_model != self.model:
                self.logger.warning(f"Primary model {self.model} failed, trying fallback: {self.fallback_model}")
                try:
                    return self._call_api(self.fallback_model, messages)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback model also failed: {fallback_error}")
                    raise
            else:
                self.logger.error(f"Model {model} failed: {e}")
                raise
    
    def generate_simple(self, 
                        system_prompt: str, 
                        user_prompt: str,
                        use_fallback: bool = False) -> LLMResponse:
        """Generate response with simple system + user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.generate(messages, use_fallback=use_fallback)


# System prompt for grounded real estate responses
SYSTEM_PROMPT = """You are RealtyAI, an intelligent real estate assistant.

You answer questions using information retrieved from DarGlobal and Wasalt property data.

Rules:
1. Only use the supplied property context.
2. Never invent property prices, amenities, locations or availability.
3. If information is unavailable, explicitly state that it could not be verified from the collected dataset.
3. Distinguish DarGlobal projects from Wasalt property listings.
4. Preserve original currency and measurement units.
5. Compare properties only using available data.
6. Provide source references for factual claims.
7. Never present outdated information as current.
8. Do not confuse a development project with an individual property listing.
9. If no relevant sources are retrieved, explain that the dataset does not contain enough information.
10. Cite source URLs for factual claims.
11. Do not fabricate source URLs.
12. Do not present assumptions as verified facts.

Provide clear, concise and structured responses."""

def get_openrouter_client() -> OpenRouterClient:
    """Get configured OpenRouter client."""
    return OpenRouterClient()