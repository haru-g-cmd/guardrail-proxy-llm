"""LangChain runnable middleware that wraps the guardrail analysis pipeline.

Usage in a LangChain chain
--------------------------
::

    from guardrail_proxy.config.settings import Settings
    from guardrail_proxy.integrations.langchain_guardrail import build_guardrail_chain
    from langchain_core.runnables import RunnableLambda

    settings = Settings()
    chain = build_guardrail_chain(settings) | RunnableLambda(lambda p: call_llm(p))
    result = chain.invoke("What is the capital of France?")

Behaviour
---------
* ALLOW   → the original prompt is passed downstream unchanged.
* SANITIZE → the redacted prompt is passed downstream.
* BLOCK   → a ``ValueError`` is raised, stopping the chain.

This matches LangChain's convention of raising exceptions for guard failures
so that ``RunnableWithFallbacks`` or ``try/except`` blocks can handle them.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import ConfigDict

from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core.analyzer import PromptAnalyzer
from guardrail_proxy.models.contracts import GuardrailVerdict


class GuardrailRunnable(RunnableSerializable[str, str]):
    """
    A LangChain-compatible :class:`~langchain_core.runnables.RunnableSerializable`
    that screens a prompt in-chain before it reaches any downstream model.

    Fields
    ------
    settings : Settings
        Application settings, including detection thresholds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Screen *input* and return the effective prompt.

        Parameters
        ----------
        input : str
            The raw user prompt.
        config : RunnableConfig, optional
            Standard LangChain run configuration (tracing, tags, etc.).

        Returns
        -------
        str
            The original prompt (ALLOW) or the sanitised prompt (SANITIZE).

        Raises
        ------
        ValueError
            When the verdict is BLOCK.  The error message includes the reason
            so it can be surfaced to the caller.
        """
        analyzer = PromptAnalyzer(self.settings)
        result = analyzer.analyze(input)

        if result.verdict == GuardrailVerdict.BLOCK:
            raise ValueError(
                f"Prompt blocked by guardrail: {result.reason}"
            )

        if result.verdict == GuardrailVerdict.SANITIZE and result.sanitized_prompt:
            return result.sanitized_prompt

        return input


def build_guardrail_chain(settings: Settings) -> GuardrailRunnable:
    """
    Construct a ready-to-use :class:`GuardrailRunnable` for embedding in
    LangChain pipelines.

    Example
    -------
    ::

        chain = build_guardrail_chain(settings) | my_llm_runnable
    """
    return GuardrailRunnable(settings=settings)
