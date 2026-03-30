"""Downstream LLM adapter: forwards screened prompts to an OpenAI-compatible API.

Local development
-----------------
When the downstream URL is unreachable (e.g. during unit tests or local dev
without a running LLM), the call fails and returns a descriptive mock string
instead of raising.  This keeps the proxy usable before a real LLM is wired in.

Production
----------
Point ``DOWNSTREAM_URL`` at any OpenAI-compatible endpoint:
  * OpenAI API  : https://api.openai.com/v1/chat/completions
  * Ollama       : http://127.0.0.1:11444/v1/chat/completions
  * LiteLLM proxy: http://127.0.0.1:4000/v1/chat/completions (or +10 → 4010)
"""

from __future__ import annotations

import httpx

from guardrail_proxy.config.settings import Settings


class DownstreamAdapter:
    """
    Thin HTTP adapter that wraps the downstream LLM chat-completions endpoint.

    Parameters
    ----------
    settings : Settings
        Application settings.  ``downstream_url`` is the only field consumed here.
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.downstream_url
        self._client = httpx.Client(timeout=30)

    def call(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Forward *prompt* to the downstream model and return the response text.

        Returns a descriptive fallback string instead of raising on network
        or API errors so proxy callers always receive a well-formed response.
        """
        try:
            response = self._client.post(
                self._url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"[downstream unavailable: {exc}]"
