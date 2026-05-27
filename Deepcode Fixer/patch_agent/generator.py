import logging
import os
from typing import Dict, Tuple

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from .config import PatchAgentConfig
from .utils_retry import retry_with_backoff

logger = logging.getLogger(__name__)


class PatchLLM:
    """LLM wrapper that handles prompt submission and basic parsing."""

    def __init__(self, config: PatchAgentConfig):
        api_key = None
        for candidate in ("PATCH_AGENT_OPENAI_KEY", "OPENAI_API_KEY"):
            api_key = api_key or os.getenv(candidate)
        if not api_key:
            raise RuntimeError("OpenAI API key not provided.")
        self.client = OpenAI(api_key=api_key)
        self.config = config

    def generate_patch(self, prompt: str) -> Dict[str, str]:
        """Generate patch with retry logic for transient API failures."""
        retryable_exceptions = (
            APIError,
            APIConnectionError,
            APITimeoutError,
            Exception,  # Catch-all for other transient errors
        )

        def _call_api():
            return self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )

        response = retry_with_backoff(
            _call_api,
            max_retries=self.config.max_retries,
            initial_delay=self.config.retry_initial_delay,
            max_delay=self.config.retry_max_delay,
            retryable_exceptions=retryable_exceptions,
            operation_name=f"Patch generation ({self.config.llm_model})",
        )

        content = response.choices[0].message.content.strip()
        logger.debug("LLM raw response: %s", content[:2000])
        patch_text, rationale = self._split_response(content)
        return {"patch": patch_text, "rationale": rationale, "raw": content}

    @staticmethod
    def _split_response(content: str) -> Tuple[str, str]:
        lower = content.lower()
        marker = "patch rationale:"
        if marker in lower:
            idx = lower.index(marker)
            return content[:idx].strip(), content[idx:].strip()
        return content.strip(), ""


