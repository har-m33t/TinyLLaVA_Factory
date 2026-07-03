"""Structured JSON log accumulator for Task 4 processing runs.

The Reporting Agent (Task 7) is only as good as the audit trail Task 4
leaves behind. Every step of :mod:`cvd_eda.task4_processing.processing`
returns a small dataclass report; :class:`ProcessingLog` collects those,
plus the config and environment, and dumps a single JSON file at the end.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class ProcessingLog:
    dataset: str
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    config: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_step(self, name: str, report: Any) -> None:
        if hasattr(report, "__dataclass_fields__"):
            report = asdict(report)
        self.steps[name] = report

    def add_warning(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        log.error(message)
        self.errors.append(message)

    def finalize(self, output_path: Path) -> None:
        self.finished_at = _now_iso()
        self.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        log.info("Wrote processing log → %s", output_path)
