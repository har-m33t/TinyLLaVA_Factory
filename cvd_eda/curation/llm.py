"""LLM-backed relevance classifier for borderline samples.

The keyword net (:mod:`cvd_eda.curation.keywords`) handles the two easy
cases: strongly-worded CVD samples get called "yes" without an LLM call, and
samples with no CVD-adjacent vocabulary at all get called "no" without an LLM
call. Everything in between — a mention of "cardiac", "aortic", "hypertension"
etc. that could plausibly be either a disease study or a healthy-tissue
reference — lands here.

We use Anthropic Messages with ``claude-haiku-4-5`` by default: cheap enough
to run over the ambiguous set (typically ~10-20% of samples) and strong enough
for a three-way relevance classification.

Outputs are structured JSON so downstream code doesn't need to parse prose::

    {
      "relevance": "yes" | "no" | "uncertain",
      "confidence": 0.0-1.0,
      "reasoning": "1-3 sentence justification anchored in the input text"
    }

An on-disk cache keyed by ``sha256(model || prompt)`` prevents duplicate spend
across reruns and across the two datasets (samples with identical metadata get
classified once).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOG = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You are a biomedical curator triaging RNA-seq sample metadata for cardiovascular-disease (CVD) relevance.

You will receive one sample at a time. The metadata is free-text pulled from GEO/SRA/GTEx/TCGA and may be terse or noisy.

Classify the sample as:
- "yes": the sample is from a subject with a cardiovascular disease, from diseased cardiovascular tissue, or from an experiment whose primary phenotype is a CVD condition. Healthy controls run *as part of a CVD study* count as "yes".
- "no": the sample has no CVD phenotype. Cardiac / vascular tissue from a non-CVD study (e.g. GTEx reference tissue, developmental biology, pan-cancer atlas without cardiac disease focus) is "no" unless the metadata specifically calls out disease.
- "uncertain": there is genuine ambiguity — the text mentions CVD-adjacent anatomy or physiology but doesn't disclose whether disease is present. Do not use "uncertain" as a hedge; use it when a human curator would also need to look up the series.

Return ONLY a JSON object with keys: relevance, confidence (0-1 float), reasoning (1-3 sentences quoting the specific phrase you relied on).
"""


USER_TEMPLATE = """Sample id: {sample_id}
Series id: {series_id}
Matched keywords (from initial filter): {matched}

Metadata text:
\"\"\"
{text}
\"\"\"
{series_block}
Classify this sample."""


SERIES_BLOCK_TEMPLATE = """
Series-level description (fetched from GEO for extra context):
\"\"\"
{series_text}
\"\"\"
"""


@dataclass
class LLMResult:
    relevance: str  # "yes" | "no" | "uncertain"
    confidence: float
    reasoning: str
    model: str
    cached: bool


class LLMClassifier:
    """Anthropic-backed classifier with an on-disk JSON cache.

    Kept thin on purpose — this module only knows how to classify one sample;
    the orchestrator in :mod:`cvd_eda.curation.curate` decides which samples
    to send.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        max_tokens: int = 400,
    ):
        try:
            import anthropic  # noqa: F401
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "The `anthropic` package is required for LLM classification. "
                "Install with: uv pip install anthropic"
            ) from exc

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Task 3 requires an LLM pass for "
                "borderline samples — set the key, or explicitly disable the LLM "
                "stage in the CLI (--disable-llm), which will mark every "
                "ambiguous sample 'uncertain' and force human review."
            )

        from anthropic import Anthropic

        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        # TinyLLaVA pins httpx==0.24.0 for Gradio; anthropic>=0.40 builds its
        # default httpx.Client with HTTPTransport(socket_options=...) which is
        # a newer-httpx-only kwarg. Passing a pre-built bare httpx.Client
        # sidesteps that code path (see anthropic/_base_client.py:916) so
        # neither pin has to move.
        self._client = Anthropic(http_client=httpx.Client(timeout=60.0))

        self._cache_path = None
        self._cache: dict[str, dict] = {}
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / f"llm_cache_{_safe_slug(model)}.json"
            if self._cache_path.exists():
                try:
                    self._cache = json.loads(self._cache_path.read_text())
                except json.JSONDecodeError:
                    LOG.warning("LLM cache at %s is corrupt; starting fresh", self._cache_path)
                    self._cache = {}

        self.call_count = 0
        self.cache_hit_count = 0

    # ------------------------------------------------------------------

    def classify(
        self,
        *,
        sample_id: str,
        series_id: str,
        matched: Iterable[str],
        text: str,
        series_text: str = "",
    ) -> LLMResult:
        matched_str = ", ".join(sorted(matched)) or "(none)"
        series_block = (
            SERIES_BLOCK_TEMPLATE.format(series_text=series_text.strip())
            if series_text
            else ""
        )
        prompt = USER_TEMPLATE.format(
            sample_id=sample_id,
            series_id=series_id or "(unknown)",
            matched=matched_str,
            text=text.strip() or "(empty)",
            series_block=series_block,
        )
        cache_key = _hash(self.model, SYSTEM_PROMPT, prompt)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self.cache_hit_count += 1
            return LLMResult(
                relevance=cached["relevance"],
                confidence=cached["confidence"],
                reasoning=cached["reasoning"],
                model=self.model,
                cached=True,
            )

        parsed = self._call_with_retry(prompt)
        self._cache[cache_key] = parsed
        self._persist_cache()
        self.call_count += 1
        return LLMResult(
            relevance=parsed["relevance"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            model=self.model,
            cached=False,
        )

    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = _extract_text(message)
                return _parse_json_response(raw)
            except Exception as exc:  # broad: SDK exceptions + JSON parse
                last_exc = exc
                if attempt < self.max_retries:
                    delay = 2 ** (attempt - 1)
                    LOG.warning(
                        "LLM call failed (attempt %s/%s): %s; retrying in %ss",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"LLM classification failed after {self.max_retries} attempts"
        ) from last_exc

    def _persist_cache(self) -> None:
        if self._cache_path is None:
            return
        tmp = self._cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        tmp.replace(self._cache_path)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _extract_text(message) -> str:
    """Pull the text content out of an anthropic.messages response."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(f"No text block in Anthropic response: {message}")


def _parse_json_response(raw: str) -> dict:
    """Parse the JSON object out of the model's response.

    Tolerates responses wrapped in a ```json fence or with a leading sentence,
    even though the system prompt asks for pure JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in response: {raw!r}")
    obj = json.loads(text[start : end + 1])

    relevance = str(obj.get("relevance", "")).strip().lower()
    if relevance not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Unexpected relevance value: {obj!r}")
    confidence = float(obj.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(obj.get("reasoning", "")).strip()
    return {"relevance": relevance, "confidence": confidence, "reasoning": reasoning}
