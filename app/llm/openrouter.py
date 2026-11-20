"""OpenRouter LLM integration for RealtyAI."""
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings


@dataclass
class LLMResponse:
    """Response from an LLM completion."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


# Only transient/service-capacity failures should trigger automatic retries/fallback.
TRANSIENT_API_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


class OpenRouterClient:
    """OpenRouter client with bounded retries and an explicit optional fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        self.fallback_model = (
            fallback_model if fallback_model is not None else settings.OPENROUTER_FALLBACK_MODEL
        )
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT
        self.site_url = settings.OPENROUTER_SITE_URL
        self.app_name = settings.OPENROUTER_APP_NAME
        self.logger = logger.bind(component="OpenRouterClient")

        self.client = (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=self.timeout,
            )
            if self.api_key
            else None
        )
        if self.client is None:
            self.logger.warning("OpenRouter API key not configured")

    def is_configured(self) -> bool:
        return self.client is not None and bool(self.api_key)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(TRANSIENT_API_ERRORS),
        reraise=True,
    )
    def _call_api(self, model: str, messages: List[Dict[str, str]]) -> LLMResponse:
        if self.client is None:
            raise RuntimeError("OpenRouter client not configured. Set OPENROUTER_API_KEY.")

        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_headers=headers or None,
        )
        if not response.choices:
            raise RuntimeError(f"OpenRouter returned no choices for model {model!r}")

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        use_fallback: bool = False,
    ) -> LLMResponse:
        """Call the selected model; fallback only after transient primary failures."""
        if not self.is_configured():
            raise RuntimeError("OpenRouter client not configured. Set OPENROUTER_API_KEY.")

        selected_model = self.fallback_model if use_fallback else self.model
        if not selected_model:
            raise ValueError("No OpenRouter model is configured.")

        try:
            return self._call_api(selected_model, messages)
        except TRANSIENT_API_ERRORS:
            if (
                use_fallback
                or not self.fallback_model
                or self.fallback_model == self.model
            ):
                raise
            self.logger.warning(
                "Primary model %s had a transient failure; trying configured fallback %s",
                self.model,
                self.fallback_model,
            )
            # Do not mask fallback errors: surface the actual fallback failure.
            return self._call_api(self.fallback_model, messages)

    def generate_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, use_fallback=use_fallback)


SYSTEM_PROMPT = """You are RealtyAI, an intelligent real estate assistant.

You answer questions using information retrieved from DarGlobal and Wasalt property data.

Rules:
1. Only use the supplied property context.
2. Never invent property prices, amenities, locations or availability.
3. If information is unavailable, explicitly state that it could not be verified from the collected dataset.
4. Distinguish DarGlobal projects from Wasalt property listings.
5. Preserve original currency and measurement units.
6. Compare properties only using available data.
7. Provide source references for factual claims.
8. Never present outdated information as current.
9. Do not confuse a development project with an individual property listing.
10. If no relevant sources are retrieved, explain that the dataset does not contain enough information.
11. Cite source URLs for factual claims.
12. Do not fabricate source URLs or present assumptions as verified facts.

Provide clear, concise and structured responses."""


def get_openrouter_client() -> OpenRouterClient:
    return OpenRouterClient()
