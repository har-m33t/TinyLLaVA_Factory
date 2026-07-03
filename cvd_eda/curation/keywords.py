"""CVD keyword net used as a high-recall first-pass filter.

Two tiers:

* ``STRONG_KEYWORDS`` — phrases that are unambiguous disease signals. A hit
  here bypasses the LLM: the sample is called CVD-relevant on the strength of
  the keyword alone.
* ``AMBIGUOUS_KEYWORDS`` — single tokens (mostly anatomical or physiological)
  that co-occur with healthy-tissue studies as often as with disease studies.
  Hits here go to the LLM for adjudication.

Anything that matches neither list is called not-CVD-relevant without an LLM
call.

The lists are intentionally over-inclusive on recall — the LLM stage in
:mod:`cvd_eda.curation.llm` is responsible for precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


STRONG_KEYWORDS: tuple[str, ...] = (
    # Chronic disease phrases
    r"heart failure",
    r"congestive heart failure",
    r"chf",
    r"myocardial infarct\w*",
    r"acute coronary syndrome",
    r"coronary artery disease",
    r"coronary heart disease",
    r"ischemic heart disease",
    r"ischaemic heart disease",
    r"cardiomyopath\w*",
    r"dilated cardiomyopathy",
    r"hypertrophic cardiomyopathy",
    r"atrial fibrillation",
    r"ventricular fibrillation",
    r"ventricular tachycardia",
    r"atherosclero\w*",
    r"aortic aneurysm",
    r"aortic stenosis",
    r"aortic dissection",
    r"peripheral artery disease",
    r"peripheral arterial disease",
    r"cardiac arrest",
    r"sudden cardiac death",
    r"pulmonary hypertension",
    r"pulmonary embolism",
    r"deep vein thrombosis",
    r"cerebrovascular\s+(disease|accident)",
    r"cardiovascular disease",
    r"cardiovascular\s+event\w*",
    r"cardiomyocyte injury",
    r"cardiac\s+(hypertrophy|fibrosis|remodel\w*|injury|damage)",
    # Common study-shorthand acronyms
    r"\bmi\b",
    r"\bhfref\b",
    r"\bhfpef\b",
    r"\bhcm\b",
    r"\bdcm\b",
    r"\bcad\b",
    r"\bchd\b",
    r"\bacs\b",
    r"\bafib\b",
    r"\bcvd\b",
)


AMBIGUOUS_KEYWORDS: tuple[str, ...] = (
    r"cardiac",
    r"cardio",
    r"cardiovascular",
    r"heart",
    r"coronary",
    r"myocardial",
    r"myocardium",
    r"ventric\w*",
    r"atrial",
    r"atrium",
    r"aortic",
    r"aorta",
    r"vascular",
    r"arterial",
    r"arrhythmi\w*",
    r"hypertens\w*",
    r"ischemi\w*",
    r"ischaemi\w*",
    r"angina",
    r"stenosis",
    r"thrombosis",
    r"embol\w*",
)


@dataclass(frozen=True)
class KeywordMatch:
    """Result of matching a single text blob against the keyword net."""

    strength: str  # "strong" | "ambiguous" | "none"
    matched: tuple[str, ...]  # normalized surface forms of the hits

    @property
    def primary(self) -> str:
        """One representative keyword to store in the output CSV."""
        return self.matched[0] if self.matched else ""


def _compile(patterns: Iterable[str]) -> re.Pattern[str]:
    parts = []
    for p in patterns:
        if p.startswith("\\b") or p.endswith("\\b"):
            parts.append(p)
        else:
            parts.append(rf"\b{p}\b")
    return re.compile("|".join(f"(?:{p})" for p in parts), flags=re.IGNORECASE)


_STRONG_RE = _compile(STRONG_KEYWORDS)
_AMBIGUOUS_RE = _compile(AMBIGUOUS_KEYWORDS)


def match(text: str) -> KeywordMatch:
    """Classify a text blob against the keyword net.

    Strong hits win: even if the text contains ambiguous tokens too, a strong
    hit escalates the whole sample to ``"strong"`` so we skip the LLM.
    """
    if not text:
        return KeywordMatch("none", ())

    strong_hits = tuple(sorted({m.group(0).lower() for m in _STRONG_RE.finditer(text)}))
    if strong_hits:
        return KeywordMatch("strong", strong_hits)

    ambiguous_hits = tuple(
        sorted({m.group(0).lower() for m in _AMBIGUOUS_RE.finditer(text)})
    )
    if ambiguous_hits:
        return KeywordMatch("ambiguous", ambiguous_hits)

    return KeywordMatch("none", ())
