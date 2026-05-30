#!/usr/bin/env python3
"""Step 6: Compute metrics and generate visualization charts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import load_config
from kds_slm.evaluation.compare import render_markdown_report, run_evaluation
from kds_slm.logging_utils import get_logger
from kds_slm.viz.plots import generate_all_figures

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize KD comparison results")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_df, summary = run_evaluation(cfg)

    figures_dir = cfg.resolve(cfg.figures_dir)
    log_path = cfg.resolve(cfg.training_logs_path)
    paths = generate_all_figures(summary, metrics_df, figures_dir, log_path)
    for path in paths:
        logger.info("Saved figure: %s", path)

    report_path = cfg.resolve(cfg.qa_results_dir) / "report.md"
    render_markdown_report(summary, report_path)
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
