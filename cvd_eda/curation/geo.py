"""Optional NCBI GEO series-description fetcher.

When a per-sample title/characteristics blob is too sparse to judge on its own
(e.g. "Sample 12, RNA-seq"), we can pull the parent GSE's title + summary from
GEO and hand that to the LLM as extra context.

Opt-in via ``--use-geo-fetch``; off by default because:

* NCBI E-utilities is rate-limited (3 req/s without an API key, 10 with).
* Not every ARCHS4 ``series_id`` is a GSE — some are internal. Failures are
  swallowed and cached as ``""`` rather than aborting the run.
* Task 3 works fine without it in most cases: sample-level metadata is
  usually informative enough. Task 5 (labeling) is the stage that really
  benefits from GEO fetches, and it has its own fetcher (:mod:`cvd_eda.labeling`).

Fetches are cached on disk per series id so reruns are free.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class GEOSeriesFetcher:
    def __init__(
        self,
        cache_dir: str | Path,
        email: str | None = None,
        api_key: str | None = None,
        min_interval_s: float = 0.4,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email or "cvd-eda@example.org"
        self.api_key = api_key
        self.min_interval_s = min_interval_s
        self._last_request_at = 0.0

    def fetch(self, series_id: str) -> str:
        """Return "title | summary | overalldesign" for a GEO series, or "" on failure."""
        if not series_id:
            return ""
        cached = self._read_cache(series_id)
        if cached is not None:
            return cached

        try:
            uid = self._esearch_gse_uid(series_id)
            text = self._esummary_text(uid) if uid else ""
        except Exception as exc:  # network / parsing errors → soft-fail
            LOG.warning("GEO fetch failed for %s: %s", series_id, exc)
            text = ""

        self._write_cache(series_id, text)
        return text

    # ------------------------------------------------------------------

    def _esearch_gse_uid(self, series_id: str) -> str | None:
        params = {
            "db": "gds",
            "term": f"{series_id}[Accession] AND gse[EntryType]",
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        payload = self._get_json(ESEARCH_URL, params)
        ids = payload.get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else None

    def _esummary_text(self, uid: str) -> str:
        params = {"db": "gds", "id": uid, "retmode": "json", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        payload = self._get_json(ESUMMARY_URL, params)
        record = payload.get("result", {}).get(uid, {})
        parts = [
            record.get("title", "").strip(),
            record.get("summary", "").strip(),
            record.get("overalldesign", "").strip(),
        ]
        return " | ".join(p for p in parts if p)

    # ------------------------------------------------------------------

    def _get_json(self, base: str, params: dict) -> dict:
        self._throttle()
        url = f"{base}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = resp.read()
        self._last_request_at = time.time()
        return json.loads(data)

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def _cache_file(self, series_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in series_id)
        return self.cache_dir / f"{safe}.txt"

    def _read_cache(self, series_id: str) -> str | None:
        p = self._cache_file(series_id)
        return p.read_text() if p.exists() else None

    def _write_cache(self, series_id: str, text: str) -> None:
        self._cache_file(series_id).write_text(text)
