"""DistilBERT-backed maliciousness scorer with a rule-based fallback.

Architecture
------------
* On startup, the classifier tries to load a fine-tuned DistilBERT model from
  the path specified in ``Settings.maliciousness_model_path``.
* If the path does not exist and ``Settings.use_heuristic_fallback`` is True,
  the heuristic scorer is used instead.  This keeps the service operational out
  of the box before a model artifact has been trained.
* When a trained model IS present, the full HuggingFace inference path is used
  and the heuristic is never called.

Heuristic design
----------------
Each marker has an individual weight so the aggregate score is proportional to
the *compound* adversarial intent of a prompt rather than a flat accumulator.
Two strong markers (e.g. "ignore all previous" + "reveal system prompt") will
push the score above the default 0.6 threshold; a single ambiguous word will not.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardrail_proxy.config.settings import Settings

# Weighted prompt-injection / jailbreak markers.
# Format: (substring, weight-added-to-score)
_MARKERS: list[tuple[str, float]] = [
    ("ignore previous instructions",       0.35),
    ("ignore all previous",                0.30),
    ("disregard safety",                   0.35),
    ("disregard all",                      0.25),
    ("forget your instructions",           0.30),
    ("forget all previous",                0.30),
    ("reveal system prompt",               0.35),
    ("reveal the system prompt",           0.35),
    ("show system prompt",                 0.30),
    ("show the system prompt",             0.30),
    ("print system prompt",                0.30),
    ("print the system prompt",            0.30),
    ("leak system prompt",                 0.35),
    ("leak the system prompt",             0.35),
    ("system prompt",                      0.20),
    ("bypass guardrail",                   0.40),
    ("bypass all",                         0.25),
    ("developer mode",                     0.25),
    ("jailbreak",                          0.35),
    ("do anything now",                    0.30),
    ("act as dan",                         0.40),
    ("you are now unrestricted",           0.40),
    ("pretend you are an unrestricted",    0.35),
    ("return all secrets",                 0.35),
    ("exfiltrate",                         0.40),
    ("execute arbitrary",                  0.35),
]

# Structural injection tokens that commonly appear in template attacks
_STRUCTURAL: tuple[str, ...] = ("{{", "}}", "<system>", "</system>", "###instruction")


class DistilBERTClassifier:
    """
    Wraps a locally trained DistilBERT binary classifier.

    Label encoding expected from the trained model:
      0 → benign
      1 → malicious / adversarial
    """

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.maliciousness_threshold
        self._use_fallback = settings.use_heuristic_fallback
        model_path = Path(settings.maliciousness_model_path)

        self._tokenizer = None
        self._model = None
        self.source: str = "heuristic-fallback"

        if model_path.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )
            self._model.eval()
            self.source = str(model_path)

    # ── Public API ────────────────────────────────────────────────────────

    def score(self, prompt: str) -> float:
        """
        Return the probability that *prompt* is malicious (0.0 – 1.0).

        Uses the trained DistilBERT model when available; falls back to the
        heuristic when the model path does not exist.
        """
        if self._model is None or self._tokenizer is None:
            if not self._use_fallback:
                raise RuntimeError(
                    f"No trained model found at '{self.source}'. "
                    "Train the model first or set USE_HEURISTIC_FALLBACK=true."
                )
            return self._heuristic(prompt)

        enc = self._tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        # If single-logit output, treat it as P(malicious); otherwise use label-1.
        return float(probs[-1].item()) if probs.numel() > 1 else float(probs.item())

    # ── Heuristic fallback ────────────────────────────────────────────────

    def _heuristic(self, prompt: str) -> float:
        """
        Rule-based approximate maliciousness scorer.

        Starting score of 0.05 (slight prior toward benign).
        Each matched marker adds its individual weight.
        Structural injection tokens add a flat 0.20.
        Result is capped at 0.99 to avoid false certainty.
        """
        lowered = prompt.lower()
        score = 0.05

        for marker, weight in _MARKERS:
            if marker in lowered:
                score += weight

        if any(tok in lowered for tok in _STRUCTURAL):
            score += 0.20

        # Long prompts with many newlines can indicate multi-shot injections
        if prompt.count("\n") > 20:
            score += 0.10

        return round(min(score, 0.99), 4)
