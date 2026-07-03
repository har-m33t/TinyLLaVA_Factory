"""CLI entrypoint for Task 7 (Reporting Agent).

Usage::

    source .venv/bin/activate

    python -m cvd_eda.reporting.run \\
        --inputs-dir cvd_eda/logs \\
        --output     cvd_eda/logs/cvd_eda_report.md

Add ``--disable-llm`` to skip the executive-summary paragraph when no
``ANTHROPIC_API_KEY`` is available. The deterministic sections of the report
are unchanged.

See :file:`cvd_eda/reporting/README.md` for the decision rubric.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from cvd_eda.reporting import inputs as inputs_mod
from cvd_eda.reporting.llm import DEFAULT_MODEL, NarrativeError, synthesize_narrative
from cvd_eda.reporting.report import build_payload, render_markdown
from cvd_eda.reporting.schema import DecisionVerdict


LOG = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cvd_eda.reporting.run",
        description="Task 7: aggregate Task 1–6 outputs into cvd_eda_report.md "
                    "with a go/no-go recommendation.",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("cvd_eda/logs"),
        help="Directory holding the upstream logs / CSVs (default: cvd_eda/logs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the Markdown report (default: <inputs-dir>/cvd_eda_report.md).",
    )
    parser.add_argument(
        "--json-payload",
        type=Path,
        default=None,
        help="Optional path to also dump the aggregator payload as JSON. Useful "
             "when a downstream dashboard wants to consume the same numbers.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Skip the LLM executive-summary. The deterministic report is unaffected.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model id for the executive summary (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--exit-code-on-no-go",
        type=int,
        default=0,
        help="Exit non-zero when the verdict is 'no-go'. Off by default so a CI "
             "pipeline can decide policy separately. Set to e.g. 3 to have Task 7 "
             "block the next stage.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inputs_dir: Path = args.inputs_dir.resolve()
    if not inputs_dir.exists():
        print(f"error: inputs dir not found: {inputs_dir}", file=sys.stderr)
        return 2

    output: Path = (args.output or (inputs_dir / "cvd_eda_report.md")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("loading upstream artifacts from %s", inputs_dir)
    loaded = inputs_mod.load_all(inputs_dir)
    payload = build_payload(loaded, inputs_dir)

    if args.disable_llm:
        payload.narrative_source = "disabled"
    else:
        try:
            payload.narrative = synthesize_narrative(payload, model=args.model)
            payload.narrative_source = "llm"
        except NarrativeError as exc:
            LOG.warning("LLM narrative disabled: %s", exc)
            payload.narrative_source = f"error:{exc}"

    output.write_text(render_markdown(payload))
    LOG.info("wrote %s", output)

    if args.json_payload is not None:
        args.json_payload.parent.mkdir(parents=True, exist_ok=True)
        args.json_payload.write_text(json.dumps(_serializable(payload), indent=2))
        LOG.info("wrote %s", args.json_payload)

    verdict = payload.decision.verdict
    print(f"verdict: {verdict}")
    if verdict == DecisionVerdict.NO_GO and args.exit_code_on_no_go:
        return args.exit_code_on_no_go
    return 0


def _serializable(payload) -> dict:
    """Turn the payload into JSON-safe primitives.

    ``asdict`` handles the nested dataclasses; we only need to promise the
    top-level structure the same way.
    """
    return asdict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
