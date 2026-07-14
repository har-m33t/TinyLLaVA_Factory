"""
label.py — Task 1: weak whole-corpus CVD label.

Applies the broad ARCHS4 CVD keyword net (locked in `.claude/elastic_net_todo.md`)
as a case-insensitive regex against three per-sample metadata fields
(`title`, `source_name_ch1`, `characteristics_ch1`). Every sample in the
corpus gets label=1 if any keyword hits any of those fields, else 0.

Why keyword-on-whole-corpus and not a curated subset
----------------------------------------------------
This stage deliberately replaces the earlier "build a clean CVD cohort"
subset-selection stage. The elastic net is doing double duty: finding which
genes predict CVD-relatedness *and* distinguishing CVD samples from every
other tissue/condition/cell-line in the corpus. Label noise (family-history
mentions, cardiac gene names in unrelated cancer studies, etc.) is an
accepted tradeoff, documented in the write-up (task 11).

Base-rate reporting is not optional — it drives the negative-subsample
ratio in step 2 and is the first thing a reviewer will ask about.

Outputs
-------
label_summary.json
    total_samples, n_positive, positive_pct, keyword_list, fields_searched,
    per_field_positive_counts (auditability: sanity-check which field lit up
    the most positives).
labels.npy
    int8 vector of length n_samples: 1 = CVD-matched, 0 = not.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eda.dataset import io as archs4_io

logger = logging.getLogger(__name__)


# The broad CVD net — kept as an exact list because it's a methods-critical
# input to the whole-corpus label. Do not narrow to the HF-only subset from
# the earlier discussion; that list was for cohort curation, not corpus label.
CVD_KEYWORDS: tuple[str, ...] = (
    "cardiovasc",
    "cardiac",
    "heart failure",
    "myocardial infarct",
    "coronary artery",
    "atherosclerosis",
    "cardiomyopathy",
    "arrhythmia",
    "atrial fibrillation",
    "hypertension",
    "ischemic heart",
    "aortic",
    "vascular disease",
    "congestive heart",
    "cardiac hypertrophy",
    "cardiac fibrosis",
)

FIELDS_SEARCHED: tuple[str, ...] = ("title", "source_name_ch1", "characteristics_ch1")


def _compile_regex(keywords: tuple[str, ...]) -> re.Pattern:
    """Combine keywords into one case-insensitive regex.

    Escaping each keyword individually avoids surprises if a term ever
    contains a regex metacharacter (currently none do, but the list is
    editable per-run and we shouldn't rely on that staying true).
    """
    return re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)


def label_field(values: np.ndarray, pattern: re.Pattern) -> np.ndarray:
    """Return a bool mask marking which entries match the pattern.

    `None`/absent field entries are already replaced with empty strings by
    the loader below; treating them as "no match" is the safe default.
    """
    return np.array([bool(pattern.search(str(v))) for v in values], dtype=bool)


def run(h5_path: Path, outdir: Path) -> Path:
    out = outdir / "label"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    pattern = _compile_regex(CVD_KEYWORDS)

    with archs4_io.open_h5(h5_path) as h5:
        n_samples = archs4_io.get_shape(h5).n_samples
        field_hits: dict[str, np.ndarray] = {}
        for field in FIELDS_SEARCHED:
            values = archs4_io.read_sample_field(h5, field)
            if values is None:
                logger.warning("field %r absent from H5; treating as no-match", field)
                field_hits[field] = np.zeros(n_samples, dtype=bool)
                continue
            field_hits[field] = label_field(values, pattern)
            logger.info("field %r: %d / %d matched", field, int(field_hits[field].sum()), n_samples)

    combined = np.zeros(n_samples, dtype=bool)
    for mask in field_hits.values():
        combined |= mask
    labels = combined.astype(np.int8)
    n_positive = int(labels.sum())
    positive_pct = round(100.0 * n_positive / n_samples, 4) if n_samples else 0.0

    np.save(out / "labels.npy", labels)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(n_samples),
        "n_positive": n_positive,
        "positive_pct": positive_pct,
        "keyword_list": list(CVD_KEYWORDS),
        "fields_searched": list(FIELDS_SEARCHED),
        "per_field_positive_counts": {
            field: int(mask.sum()) for field, mask in field_hits.items()
        },
        "note": (
            "Weak whole-corpus label — keyword match on metadata only, no manual "
            "curation, no negation rules. Label noise is expected; see writeup."
        ),
    }
    with open(out / "label_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "labelled %d / %d samples as CVD-related (%.3f%%)",
        n_positive, n_samples, positive_pct,
    )
    return out
